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

import numpy as np
import torch
from torch import nn
from typing import Optional, List, Tuple

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
            hidden_units = self._wiring.get_neurons_of_layer(l)
            
            # Extract adjacency matrices for this layer
            # Following the original implementation pattern
            if l == 0:
                # First layer: sensory inputs to inter neurons
                input_adjacency = self._wiring.sensory_adjacency_matrix[:, hidden_units]
            else:
                # Later layers: previous layer to current layer
                prev_layer_neurons = self._wiring.get_neurons_of_layer(l - 1)
                input_adjacency = self._wiring.adjacency_matrix[:, hidden_units]
                input_adjacency = input_adjacency[prev_layer_neurons, :]
            
            # The original adds full recurrence within layer (identity matrix)
            # So we create a combined adjacency that includes input connections and full recurrence
            combined_adjacency = np.zeros((in_features + len(hidden_units), len(hidden_units)))
            combined_adjacency[:in_features, :] = input_adjacency
            combined_adjacency[in_features:, :] = np.ones((len(hidden_units), len(hidden_units)))  # Full recurrence
            
            # For efficient cell, we pass this as the recurrent adjacency
            recurrent_adjacency = combined_adjacency
            sensory_adjacency = None  # Everything is in the combined adjacency

            # Create efficient CfC cell for this layer
            rnn_cell = EfficientCfCCell(
                in_features,
                len(hidden_units),
                mode=mode,
                backbone_activation="lecun_tanh",
                backbone_units=0,  # No backbone in wired mode
                backbone_layers=0,
                backbone_dropout=0.0,
                adjacency_matrix=recurrent_adjacency,
                sensory_adjacency_matrix=sensory_adjacency if l == 0 else None,
            )
            
            # Register as a submodule
            self.register_module(f"layer_{l}", rnn_cell)
            self._layers.append(rnn_cell)
            
            # Update parameter counts
            if hasattr(rnn_cell, 'sparsity_info'):
                info = rnn_cell.sparsity_info
                self.total_possible_params += info['total_possible_connections']
                self.total_actual_params += info['actual_connections']
            
            in_features = len(hidden_units)

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
        return self.motor_size

    @property
    def synapse_count(self):
        """Count of actual synapses (non-zero connections) in the network."""
        return np.sum(np.abs(self._wiring.adjacency_matrix) > 0)

    @property
    def sensory_synapse_count(self):
        """Count of actual sensory synapses (non-zero connections) from inputs."""
        return np.sum(np.abs(self._wiring.sensory_adjacency_matrix) > 0)
    
    @property
    def sparsity_info(self):
        """Get overall sparsity information for the entire wired network."""
        layer_infos = []
        for i, layer in enumerate(self._layers):
            if hasattr(layer, 'sparsity_info'):
                info = layer.sparsity_info
                info['layer'] = i
                layer_infos.append(info)
        
        return {
            "total_possible_parameters": self.total_possible_params,
            "total_actual_parameters": self.total_actual_params,
            "overall_sparsity": 1.0 - (self.total_actual_params / self.total_possible_params) if self.total_possible_params > 0 else 0.0,
            "overall_memory_savings": f"{self.total_possible_params / self.total_actual_params:.2f}x" if self.total_actual_params > 0 else "N/A",
            "total_parameter_reduction": self.total_possible_params - self.total_actual_params,
            "layer_details": layer_infos,
            "synapse_count": self.synapse_count,
            "sensory_synapse_count": self.sensory_synapse_count,
        }

    def forward(self, input: torch.Tensor, hx: torch.Tensor, timespans: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through the efficient wired CfC network.
        
        Args:
            input: Input tensor [batch_size, input_size]
            hx: Hidden state [batch_size, state_size]
            timespans: Time spans [batch_size] or scalar (default: 1.0)
            
        Returns:
            (output, new_hidden): Output from motor neurons and updated hidden state
        """
        if timespans is None:
            timespans = torch.tensor(1.0, device=input.device)
        
        # Split hidden state according to layer sizes
        h_state = torch.split(hx, self.layer_sizes, dim=1)

        new_h_state = []
        inputs = input
        
        for i in range(self.num_layers):
            h, _ = self._layers[i].forward(inputs, h_state[i], timespans)
            inputs = h  # Output of this layer becomes input to next
            new_h_state.append(h)

        # Concatenate all hidden states
        new_h_state = torch.cat(new_h_state, dim=1)
        
        # Extract motor neuron outputs (they are the first neurons in the state)
        motor_output = new_h_state[:, :self.motor_size]
        
        return motor_output, new_h_state
