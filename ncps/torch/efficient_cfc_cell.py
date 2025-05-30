# Copyright 2022-2024 Mathias Lechner and Ramin Hasani
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

from typing import Optional, Tuple  # Added Optional

import numpy as np
import torch
import torch.nn as nn

from .cfc_cell import LeCun  # Assuming cfc_cell.py is in the same directory


class EfficientCfCCell(nn.Module):
    """An efficient sparse implementation of the CfC cell that only allocates
    parameters for connections that actually exist in the wiring.

    This implementation provides the same functionality as CfCCell but with
    significant memory and computation savings when using sparse connectivity.
    """

    def __init__(
        self,
        input_size,
        hidden_size,
        mode="default",
        backbone_activation="lecun_tanh",
        backbone_units=128,
        backbone_layers=1,
        backbone_dropout=0.0,
        adjacency_matrix: Optional[np.ndarray] = None,  # recurrent (h->h)
        sensory_adjacency_matrix: Optional[np.ndarray] = None,  # input (x->h)
    ):
        """Initialize an efficient CfC cell with sparse connectivity.

        Args:
            input_size: Number of input features
            hidden_size: Number of hidden units
            mode: CfC mode - "default", "pure", or "no_gate"
            backbone_activation: Activation function for backbone layers
            backbone_units: Number of units in backbone layers
            backbone_layers: Number of backbone layers
            backbone_dropout: Dropout rate for backbone layers
            adjacency_matrix: Sparse adjacency matrix for recurrent connections [hidden_size, hidden_size]
            sensory_adjacency_matrix: Sparse adjacency matrix for input connections [input_size, hidden_size]
        """
        super(EfficientCfCCell, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.mode = mode
        self.backbone_layers = backbone_layers

        # Determine if sparse connections are used
        # Sparse mode is active if any adjacency matrix is provided and non-empty.
        use_sparse_def = (
            sensory_adjacency_matrix is not None and sensory_adjacency_matrix.size > 0
        ) or (adjacency_matrix is not None and adjacency_matrix.size > 0)

        if use_sparse_def:
            full_adjacency = torch.zeros(
                input_size + hidden_size, hidden_size, dtype=torch.float32
            )

            if (
                sensory_adjacency_matrix is not None
                and sensory_adjacency_matrix.shape == (input_size, hidden_size)
            ):
                if not isinstance(sensory_adjacency_matrix, np.ndarray):
                    sensory_adjacency_matrix = np.array(
                        sensory_adjacency_matrix, dtype=np.float32
                    )
                full_adjacency[:input_size] = torch.from_numpy(
                    sensory_adjacency_matrix.astype(np.float32)
                )

            if adjacency_matrix is not None and adjacency_matrix.shape == (
                hidden_size,
                hidden_size,
            ):
                if not isinstance(adjacency_matrix, np.ndarray):
                    adjacency_matrix = np.array(adjacency_matrix, dtype=np.float32)
                full_adjacency[input_size:] = torch.from_numpy(
                    adjacency_matrix.astype(np.float32)
                )

            # Find all non-zero connections using a small epsilon for float comparisons
            self.edges = torch.nonzero(full_adjacency.abs() > 1e-5)  # Epsilon for float
            self.num_edges = len(self.edges)

            if self.num_edges > 0:
                self.has_sparse_connections = True
                self.register_buffer("src_indices", self.edges[:, 0])
                self.register_buffer("dst_indices", self.edges[:, 1])
                # Store signs if you need them for specific operations, otherwise magnitudes
                self.register_buffer(
                    "edge_signs",
                    full_adjacency[self.edges[:, 0], self.edges[:, 1]].sign(),
                )
            else:  # No edges found despite matrices provided (e.g., all zero matrices)
                self.has_sparse_connections = False
                self.num_edges = (
                    input_size + hidden_size
                ) * hidden_size  # Fallback to dense count
        else:
            # Fall back to dense connections if no adjacency matrix provided or they are empty
            self.has_sparse_connections = False
            self.num_edges = (
                input_size + hidden_size
            ) * hidden_size  # for dense fallback calculations

        # Set up activation function (same as your original)
        if backbone_activation == "silu":
            self.backbone_act_fn = nn.SiLU
        elif backbone_activation == "relu":
            self.backbone_act_fn = nn.ReLU
        elif backbone_activation == "tanh":
            self.backbone_act_fn = nn.Tanh
        elif backbone_activation == "gelu":
            self.backbone_act_fn = nn.GELU
        elif backbone_activation == "lecun_tanh":
            self.backbone_act_fn = LeCun
        else:
            raise ValueError(f"Unknown activation {backbone_activation}")

        # Backbone network
        self.backbone = None
        if backbone_layers > 0:
            layer_list = [
                nn.Linear(input_size + hidden_size, backbone_units),
                self.backbone_act_fn(),
            ]
            for i in range(1, backbone_layers):
                layer_list.append(nn.Linear(backbone_units, backbone_units))
                layer_list.append(self.backbone_act_fn())
                if backbone_dropout > 0.0:
                    layer_list.append(torch.nn.Dropout(backbone_dropout))
            self.backbone = nn.Sequential(*layer_list)

        # CfC-specific layers
        # cat_shape is the dimension of the input to ff1/ff2/time_a/time_b
        # If backbone, it's backbone_units. Otherwise, it's (input_size + hidden_size).
        cat_shape = (
            backbone_units if backbone_layers > 0 else (input_size + hidden_size)
        )

        # Parameters for transformations
        if self.has_sparse_connections and self.backbone is None:
            # Efficient sparse parameters only if no backbone and sparse connections exist
            self.ff1_weights = nn.Parameter(
                torch.randn(self.num_edges)
                / np.sqrt(cat_shape)  # Fan-in from cat_shape
            )
            if mode != "pure":
                self.ff2_weights = nn.Parameter(
                    torch.randn(self.num_edges) / np.sqrt(cat_shape)
                )
                self.time_a_weights = nn.Parameter(
                    torch.randn(self.num_edges) / np.sqrt(cat_shape)
                )
                self.time_b_weights = nn.Parameter(
                    torch.randn(self.num_edges) / np.sqrt(cat_shape)
                )
        else:
            # Dense layers if backbone is present OR if not using sparse connections
            self.ff1 = nn.Linear(cat_shape, hidden_size)
            if mode != "pure":
                self.ff2 = nn.Linear(cat_shape, hidden_size)
                self.time_a = nn.Linear(cat_shape, hidden_size)
                self.time_b = nn.Linear(cat_shape, hidden_size)

        # Bias terms (always dense, applied per hidden unit)
        self.ff1_bias = nn.Parameter(torch.zeros(hidden_size))
        if mode != "pure":
            self.ff2_bias = nn.Parameter(torch.zeros(hidden_size))
            self.time_a_bias = nn.Parameter(torch.zeros(hidden_size))
            self.time_b_bias = nn.Parameter(torch.zeros(hidden_size))

        # Mode-specific parameters (same as your original)
        if mode == "pure":
            self.w_tau = nn.Parameter(torch.zeros(1, hidden_size))
            self.A = nn.Parameter(torch.ones(1, hidden_size))

        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()

        self.init_weights()

    def init_weights(self):
        # Initialize dense parameters
        for name, param in self.named_parameters():
            if param.dim() == 2 and param.requires_grad:  # Dense weight matrices
                torch.nn.init.xavier_uniform_(param)
            elif (
                "weights" in name
                and param.dim() == 1
                and self.has_sparse_connections
                and self.backbone is None
            ):
                # Initialize sparse weight vectors for ff1_weights, ff2_weights etc.
                # The fan_in for these sparse weights depends on the number of incoming edges to a typical hidden unit,
                # which is not directly self.num_edges. A simpler proxy is sqrt of input to that layer.
                # If self.backbone is None, cat_shape is (input_size + hidden_size)
                cat_shape = self.input_size + self.hidden_size
                torch.nn.init.uniform_(
                    param, -1.0 / np.sqrt(cat_shape), 1.0 / np.sqrt(cat_shape)
                )
            elif "bias" in name and param.requires_grad:  # Bias terms
                torch.nn.init.zeros_(param)

        # Specific init for pure mode (if needed, but usually zeros/ones is fine)
        if self.mode == "pure":
            if hasattr(self, "w_tau"):
                torch.nn.init.zeros_(self.w_tau)
            if hasattr(self, "A"):
                torch.nn.init.ones_(self.A)

    def _sparse_linear(
        self, x: torch.Tensor, weights: torch.Tensor, bias: torch.Tensor
    ) -> torch.Tensor:
        """Compute sparse matrix multiplication using edge list representation."""
        batch_size = x.shape[0]
        # Ensure output tensor has the correct dtype and device
        output = torch.zeros(
            batch_size, self.hidden_size, device=x.device, dtype=x.dtype
        )

        # Gather source values: x has shape [batch, input_size + hidden_size]
        # self.src_indices contains indices into the concatenated dimension
        src_values = x.gather(
            1, self.src_indices.unsqueeze(0).expand(batch_size, -1)
        )  # [batch, num_edges]

        # Apply weights and edge signs (if edge_signs are meant to flip weights)
        # Assuming weights already incorporate any necessary sign from adjacency.
        # If edge_signs are +1/-1 from original adjacency, and weights are learned magnitudes:
        # weighted_values = src_values * weights * self.edge_signs
        # If weights are learned directly, and signs are just for info:
        weighted_values = src_values * weights  # [batch, num_edges]

        # Scatter-add to destination indices
        # self.dst_indices are for the hidden_size dimension
        output.scatter_add_(
            1, self.dst_indices.unsqueeze(0).expand(batch_size, -1), weighted_values
        )

        return output + bias.unsqueeze(0)  # Add bias (broadcast over batch)

    def forward(
        self, input: torch.Tensor, hx: torch.Tensor, ts: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass of the efficient CfC cell.

        Args:
            input: Input tensor [batch_size, input_size]
            hx: Hidden state [batch_size, hidden_size]
            ts: Time step [batch_size] or scalar

        Returns:
            (output, new_hidden): Both tensors of shape [batch_size, hidden_size]
        """
        batch_size = input.shape[0]

        # Concatenate input and hidden state
        x_concat = torch.cat([input, hx], dim=1)  # [batch, input_size + hidden_size]

        # Apply backbone if present
        processed_x = self.backbone(x_concat) if self.backbone is not None else x_concat

        # Compute ff1
        if self.has_sparse_connections and self.backbone is None:
            ff1 = self._sparse_linear(processed_x, self.ff1_weights, self.ff1_bias)
        else:
            ff1 = self.ff1(processed_x)  # Will use self.ff1.bias internally

        if self.mode == "pure":
            # Pure CfC solution
            # Ensure ts has the right shape for broadcasting with ff1 (Batch, Hidden) or (Batch, 1)
            if ts.dim() == 0:  # scalar
                ts_b = ts
            elif ts.dim() == 1 and ts.shape[0] == batch_size:  # (Batch_size)
                ts_b = ts.unsqueeze(1)  # -> (Batch_size, 1)
            else:  # Assume ts is already (Batch_size, 1) or compatible
                ts_b = ts

            new_hidden = (
                -self.A
                * torch.exp(-ts_b * (torch.abs(self.w_tau) + torch.abs(ff1)))
                * ff1
                + self.A
            )
        else:
            # Standard CfC
            if self.has_sparse_connections and self.backbone is None:
                ff2 = self._sparse_linear(processed_x, self.ff2_weights, self.ff2_bias)
                t_a = self._sparse_linear(
                    processed_x, self.time_a_weights, self.time_a_bias
                )
                t_b = self._sparse_linear(
                    processed_x, self.time_b_weights, self.time_b_bias
                )
            else:
                ff2 = self.ff2(processed_x)  # Uses self.ff2.bias
                t_a = self.time_a(processed_x)  # Uses self.time_a.bias
                t_b = self.time_b(processed_x)  # Uses self.time_b.bias

            ff1_act = self.tanh(ff1)  # Apply tanh after ff1 (if not pure)
            ff2_act = self.tanh(ff2)

            # Ensure ts has the right shape for broadcasting with t_a, t_b [Batch, Hidden]
            if ts.dim() == 0:  # scalar
                ts_interp = ts
            elif ts.dim() == 1 and ts.shape[0] == batch_size:  # (Batch_size)
                ts_interp = ts.unsqueeze(
                    1
                )  # -> (Batch_size, 1) for broadcasting with (B,H)
            elif (
                ts.dim() == 2 and ts.shape[0] == batch_size and ts.shape[1] == 1
            ):  # (Batch_size, 1)
                ts_interp = ts
            else:  # Fallback or error for unexpected ts shape
                # This case should ideally be handled or raise an error.
                # For now, assume it can broadcast or take the first element if it's a batch of scalars.
                if ts.numel() == batch_size:
                    ts_interp = ts.unsqueeze(1)
                else:
                    ts_interp = (
                        ts[0]
                        if ts.numel() > 0
                        else torch.tensor(1.0, device=input.device, dtype=input.dtype)
                    )

            t_interp = self.sigmoid(t_a * ts_interp + t_b)

            if self.mode == "no_gate":
                new_hidden = ff1_act + t_interp * ff2_act
            else:  # "default" mode
                new_hidden = ff1_act * (1.0 - t_interp) + t_interp * ff2_act

        return new_hidden, new_hidden

    @property
    def sparsity_info(self):
        """Return information about the sparsity of this cell."""
        if self.has_sparse_connections:
            # This is the total possible if it were dense BEFORE any backbone
            total_possible_pre_backbone = (
                self.input_size + self.hidden_size
            ) * self.hidden_size
            actual_conn = self.num_edges  # This is num_edges for the sparse part

            # If backbone exists, ff1 etc. are dense, so actual connections for those parts are cat_shape * hidden_size
            if self.backbone is not None:
                cat_shape = (
                    self.backbone.in_features
                    if hasattr(self.backbone, "in_features")
                    else (
                        self.backbone[0].in_features
                        if isinstance(self.backbone, nn.Sequential)
                        and len(self.backbone) > 0
                        else (self.input_size + self.hidden_size)
                    )
                )  # fallback for cat_shape
                # Approximation: count params for ff1, ff2, time_a, time_b if they are dense
                num_dense_layers = 1  # ff1
                if self.mode != "pure":
                    num_dense_layers += 3  # ff2, time_a, time_b
                actual_conn_dense_part = num_dense_layers * (
                    cat_shape * self.hidden_size + self.hidden_size
                )  # weights + bias

                # This info becomes more complex: part sparse (potentially, if edges defined), part dense
                return {
                    "message": "Mixed sparsity due to backbone. ff-layers are dense.",
                    "total_possible_connections_pre_backbone": total_possible_pre_backbone,
                    "actual_connections_in_dense_ff_layers": actual_conn_dense_part,
                    "sparsity_level": "N/A (mixed)",
                    "memory_savings": "N/A (mixed)",
                    "parameter_reduction": "N/A (mixed)",
                }

            # If no backbone and sparse connections are truly used
            sparsity_level = (
                1.0 - (actual_conn / total_possible_pre_backbone)
                if total_possible_pre_backbone > 0
                else 0.0
            )
            memory_savings_factor = (
                total_possible_pre_backbone / actual_conn
                if actual_conn > 0
                else float("inf")
            )

            return {
                "total_possible_connections": total_possible_pre_backbone,
                "actual_connections": actual_conn,
                "sparsity_level": sparsity_level,
                "memory_savings": (
                    f"{memory_savings_factor:.2f}x"
                    if actual_conn > 0
                    else "Infinite (no connections)"
                ),
                "parameter_reduction": total_possible_pre_backbone - actual_conn,
            }
        else:  # Fully dense case (no sparse matrices provided or backbone makes it dense)
            cat_shape = (
                self.backbone[-2].out_features
                if self.backbone is not None
                and len(self.backbone) > 1
                and isinstance(self.backbone[-2], nn.Linear)
                else (
                    self.backbone[0].out_features
                    if self.backbone is not None
                    and isinstance(self.backbone[0], nn.Linear)
                    else (self.input_size + self.hidden_size)
                )
            )
            total_dense_connections = cat_shape * self.hidden_size
            num_dense_layers = 1  # ff1
            if self.mode != "pure":
                num_dense_layers += 3  # ff2, time_a, time_b

            # total actual parameters for the ff layers (weights + biases)
            actual_params = num_dense_layers * (
                total_dense_connections + self.hidden_size
            )

            return {
                "total_possible_connections": actual_params,  # In dense mode, possible = actual for these layers
                "actual_connections": actual_params,
                "sparsity_level": 0.0,
                "memory_savings": "1.00x (dense)",
                "parameter_reduction": 0,
            }
