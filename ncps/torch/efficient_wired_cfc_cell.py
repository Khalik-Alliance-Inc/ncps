# Copyright 2022-2024 Mathias Lechner
# Efficient Sparse Implementation by Khalik Alliance Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Optional, Tuple

import numpy as np
import torch
from torch import nn

from .efficient_cfc_cell import EfficientCfCCell


class EfficientWiredCfCCell(nn.Module):
    """An efficient sparse implementation of the Wired CfC cell that only allocates
    parameters for connections that actually exist in the wiring.

    This implementation provides the same functionality as WiredCfCCell but with
    significant memory and computation savings when using sparse connectivity patterns.
    """

    def __init__(
        self,
        input_size: Optional[int],
        wiring,
        mode: str = "default",
    ):
        """Initialize an efficient wired CfC cell.

        Args:
            input_size: Number of input features (can be None if wiring is already built)
            wiring: A Wiring object that defines the network structure
            mode: CfC mode - "default", "pure", or "no_gate"
        """
        super(EfficientWiredCfCCell, self).__init__()

        if input_size is not None:
            wiring.build(input_size)
        if not wiring.is_built():
            raise ValueError(
                "Wiring error! Unknown number of input features. Please pass the parameter 'input_size' or call the 'wiring.build()'."
            )
        self._wiring = wiring

        self._layers = []
        in_features = wiring.input_dim

        # Count total parameters across all layers for reporting
        self.total_possible_params = 0
        self.total_actual_params = 0

        for l in range(wiring.num_layers):
            hidden_units_indices = self._wiring.get_neurons_of_layer(l)
            num_hidden_units_in_layer = len(hidden_units_indices)

            # Extract adjacency matrices for this layer
            if l == 0:
                # First layer: sensory inputs to inter neurons
                # Shape: (sensory_input_dim, num_hidden_units_in_layer)
                cell_sensory_adj = self._wiring.sensory_adjacency_matrix[
                    :, hidden_units_indices
                ]
            else:
                # Later layers: previous layer to current layer
                prev_layer_neurons_indices = self._wiring.get_neurons_of_layer(l - 1)
                # Shape: (num_prev_layer_hidden_units, num_hidden_units_in_layer)
                inter_neuron_adj = self._wiring.adjacency_matrix[
                    np.ix_(prev_layer_neurons_indices, hidden_units_indices)
                ]
                cell_sensory_adj = inter_neuron_adj

            # Recurrent connections within the current layer's hidden units.
            # The original WiredCfCCell implicitly adds full recurrence.
            # For EfficientCfCCell, this is the 'adjacency_matrix' parameter.
            # Shape: (num_hidden_units_in_layer, num_hidden_units_in_layer)
            cell_recurrent_adj = np.ones(
                (num_hidden_units_in_layer, num_hidden_units_in_layer)
            )

            # Create efficient CfC cell for this layer
            rnn_cell = EfficientCfCCell(
                in_features,  # Number of inputs to this cell (either sensory or from prev layer)
                num_hidden_units_in_layer,  # Number of hidden units in this cell/layer
                mode=mode,
                backbone_activation="lecun_tanh",
                backbone_units=0,  # No backbone in wired mode per layer by default
                backbone_layers=0,
                backbone_dropout=0.0,
                adjacency_matrix=cell_recurrent_adj,
                sensory_adjacency_matrix=cell_sensory_adj,
            )

            # Register as a submodule
            self.register_module(f"layer_{l}", rnn_cell)
            self._layers.append(rnn_cell)

            # Update parameter counts
            if hasattr(rnn_cell, "sparsity_info"):
                info = rnn_cell.sparsity_info
                self.total_possible_params += info["total_possible_connections"]
                self.total_actual_params += info["actual_connections"]

            in_features = (
                num_hidden_units_in_layer  # Output of this layer is input to next
            )

    @property
    def state_size(self):
        return self._wiring.units

    @property
    def layer_sizes(self):
        return [
            len(self._wiring.get_neurons_of_layer(i))
            for i in range(self._wiring.num_layers)
        ]

    @property
    def num_layers(self):
        return self._wiring.num_layers

    @property
    def sensory_size(self):
        return self._wiring.input_dim

    @property
    def motor_size(self):
        return self._wiring.output_dim

    @property
    def output_size(self):
        # This should align with what CfC wrapper expects for its projection layer.
        # Typically, this is self.motor_size or the size of the last layer's output.
        # Using self.motor_size is more consistent with wiring definition.
        return self.motor_size

    @property
    def synapse_count(self):
        """Count of actual synapses (non-zero connections) in the network."""
        return np.sum(
            np.abs(self._wiring.adjacency_matrix) > 1e-5
        )  # Use epsilon for float

    @property
    def sensory_synapse_count(self):
        """Count of actual sensory synapses (non-zero connections) from inputs."""
        return np.sum(
            np.abs(self._wiring.sensory_adjacency_matrix) > 1e-5
        )  # Use epsilon

    @property
    def sparsity_info(self):
        """Get overall sparsity information for the entire wired network."""
        layer_infos = []
        for i, layer in enumerate(self._layers):
            if hasattr(layer, "sparsity_info"):
                info = layer.sparsity_info
                info["layer"] = i
                layer_infos.append(info)

        return {
            "total_possible_parameters": self.total_possible_params,
            "total_actual_parameters": self.total_actual_params,
            "overall_sparsity": (
                1.0 - (self.total_actual_params / self.total_possible_params)
                if self.total_possible_params > 0
                else 0.0
            ),
            "overall_memory_savings": (
                f"{self.total_possible_params / self.total_actual_params:.2f}x"
                if self.total_actual_params > 0 and self.total_possible_params > 0
                else "N/A"
            ),
            "total_parameter_reduction": self.total_possible_params
            - self.total_actual_params,
            "layer_details": layer_infos,
            "synapse_count": self.synapse_count,
            "sensory_synapse_count": self.sensory_synapse_count,
        }

    def forward(
        self, input: torch.Tensor, hx: torch.Tensor, timespans: torch.Tensor = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through the efficient wired CfC network.

        Args:
            input: Input tensor [batch_size, input_size]
            hx: Hidden state [batch_size, state_size]
            timespans: Time spans [batch_size] or scalar (default: 1.0)

        Returns:
            (output, new_hidden): Output from motor neurons and updated hidden state
        """
        if timespans is None:
            # Create a default timespan tensor on the same device as input
            # Ensure it's a scalar tensor if not batched, or shaped for batch if input is batched
            if input.dim() > 1 and input.size(0) > 1:  # Batched input
                timespans = torch.full(
                    (input.size(0),), 1.0, device=input.device, dtype=input.dtype
                )
            else:  # Unbatched or single item batch
                timespans = torch.tensor(1.0, device=input.device, dtype=input.dtype)

        # Split hidden state according to layer sizes
        h_state_per_layer = torch.split(hx, self.layer_sizes, dim=1)

        new_h_state_list = []
        current_layer_input = input
        last_layer_output = None  # To store the output of the very last layer

        for i in range(self.num_layers):
            # For timespans, if it's a batched tensor, it should already match.
            # If it's a scalar, EfficientCfCCell will handle broadcasting.
            layer_ts = timespans
            if (
                timespans.dim() == 1
                and self._layers[i].hidden_size > 1
                and timespans.size(0) == input.size(0)
            ):
                # If timespans is per-batch-item, pass it directly
                pass
            elif timespans.dim() == 0:  # Scalar timespan
                pass  # EfficientCfCCell will unsqueeze if needed
            else:  # Fallback for safety or if timespans is misshaped for cell
                layer_ts = (
                    timespans[0]
                    if timespans.numel() > 0
                    else torch.tensor(1.0, device=input.device, dtype=input.dtype)
                )

            h_output_current_layer, _ = self._layers[i].forward(
                current_layer_input, h_state_per_layer[i], layer_ts
            )
            current_layer_input = (
                h_output_current_layer  # Output of this layer becomes input to next
            )
            new_h_state_list.append(h_output_current_layer)
            if i == self.num_layers - 1:
                last_layer_output = h_output_current_layer

        # Concatenate all hidden states
        concatenated_new_h_state = torch.cat(new_h_state_list, dim=1)

        # Determine the "primary" output based on wiring's output neurons
        # This ensures the first returned value matches self.output_size (motor_size)
        if (
            self._wiring.output_dim > 0
            and self._wiring.output_neuron_indices is not None
        ):
            output_indices = torch.tensor(
                self._wiring.output_neuron_indices,
                device=concatenated_new_h_state.device,
                dtype=torch.long,
            )
            final_output_to_return = concatenated_new_h_state.index_select(
                1, output_indices
            )
        elif (
            last_layer_output is not None
        ):  # Fallback if no specific output neurons, use last layer's output
            final_output_to_return = last_layer_output
        else:  # Should not happen in a valid network
            final_output_to_return = torch.empty(
                (input.size(0), 0), device=input.device, dtype=input.dtype
            )

        return final_output_to_return, concatenated_new_h_state
