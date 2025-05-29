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

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from typing import Optional, Union, Tuple
from .cfc_cell import LeCun


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
        adjacency_matrix=None,
        sensory_adjacency_matrix=None,
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
        
        # Extract connectivity structure
        if adjacency_matrix is not None:
            # Combine sensory and recurrent adjacency matrices
            full_adjacency = torch.zeros(input_size + hidden_size, hidden_size)
            if sensory_adjacency_matrix is not None:
                full_adjacency[:input_size] = torch.from_numpy(sensory_adjacency_matrix).float()
            full_adjacency[input_size:] = torch.from_numpy(adjacency_matrix).float()
            
            # Find all non-zero connections
            self.edges = torch.nonzero(full_adjacency.abs() > 0)
            self.num_edges = len(self.edges)
            
            # Store edge information
            self.register_buffer('src_indices', self.edges[:, 0])
            self.register_buffer('dst_indices', self.edges[:, 1])
            self.register_buffer('edge_signs', full_adjacency[self.edges[:, 0], self.edges[:, 1]].sign())
            
            # Create parameters only for existing connections
            self.has_sparse_connections = True
        else:
            # Fall back to dense connections if no adjacency matrix provided
            self.has_sparse_connections = False
            self.num_edges = (input_size + hidden_size) * hidden_size
        
        # Set up activation function
        if backbone_activation == "silu":
            backbone_activation = nn.SiLU
        elif backbone_activation == "relu":
            backbone_activation = nn.ReLU
        elif backbone_activation == "tanh":
            backbone_activation = nn.Tanh
        elif backbone_activation == "gelu":
            backbone_activation = nn.GELU
        elif backbone_activation == "lecun_tanh":
            backbone_activation = LeCun
        else:
            raise ValueError(f"Unknown activation {backbone_activation}")
        
        # Backbone network
        self.backbone = None
        if backbone_layers > 0:
            layer_list = [
                nn.Linear(input_size + hidden_size, backbone_units),
                backbone_activation(),
            ]
            for i in range(1, backbone_layers):
                layer_list.append(nn.Linear(backbone_units, backbone_units))
                layer_list.append(backbone_activation())
                if backbone_dropout > 0.0:
                    layer_list.append(torch.nn.Dropout(backbone_dropout))
            self.backbone = nn.Sequential(*layer_list)
        
        # CfC-specific layers
        cat_shape = backbone_units if backbone_layers > 0 else (input_size + hidden_size)
        
        if self.has_sparse_connections:
            # Efficient sparse parameters
            if self.backbone is None:
                # Direct sparse connections
                self.ff1_weights = nn.Parameter(torch.randn(self.num_edges) / np.sqrt(cat_shape))
                if mode != "pure":
                    self.ff2_weights = nn.Parameter(torch.randn(self.num_edges) / np.sqrt(cat_shape))
                    
                    # Time interpolation parameters per hidden unit
                    self.time_a_weights = nn.Parameter(torch.randn(self.num_edges) / np.sqrt(cat_shape))
                    self.time_b_weights = nn.Parameter(torch.randn(self.num_edges) / np.sqrt(cat_shape))
            else:
                # Backbone output to hidden - dense since backbone is dense
                self.ff1 = nn.Linear(cat_shape, hidden_size)
                if mode != "pure":
                    self.ff2 = nn.Linear(cat_shape, hidden_size)
                    self.time_a = nn.Linear(cat_shape, hidden_size)
                    self.time_b = nn.Linear(cat_shape, hidden_size)
        else:
            # Fall back to dense layers
            self.ff1 = nn.Linear(cat_shape, hidden_size)
            if mode != "pure":
                self.ff2 = nn.Linear(cat_shape, hidden_size)
                self.time_a = nn.Linear(cat_shape, hidden_size)
                self.time_b = nn.Linear(cat_shape, hidden_size)
        
        # Bias terms (always dense)
        self.ff1_bias = nn.Parameter(torch.zeros(hidden_size))
        if mode != "pure":
            self.ff2_bias = nn.Parameter(torch.zeros(hidden_size))
            self.time_a_bias = nn.Parameter(torch.zeros(hidden_size))
            self.time_b_bias = nn.Parameter(torch.zeros(hidden_size))
        
        # Mode-specific parameters
        if mode == "pure":
            self.w_tau = nn.Parameter(torch.zeros(1, hidden_size))
            self.A = nn.Parameter(torch.ones(1, hidden_size))
        
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        
        self.init_weights()
    
    def init_weights(self):
        # Initialize dense parameters
        for name, param in self.named_parameters():
            if param.dim() == 2 and param.requires_grad:
                torch.nn.init.xavier_uniform_(param)
            elif 'weights' in name and param.dim() == 1:
                # Initialize sparse weight vectors
                fan_in = self.input_size + self.hidden_size
                torch.nn.init.uniform_(param, -1/np.sqrt(fan_in), 1/np.sqrt(fan_in))
    
    def _sparse_linear(self, x: torch.Tensor, weights: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
        """Compute sparse matrix multiplication using edge list representation."""
        batch_size = x.shape[0]
        output = torch.zeros(batch_size, self.hidden_size, device=x.device, dtype=x.dtype)
        
        # Gather source values
        src_values = x[:, self.src_indices]  # [batch, num_edges]
        
        # Apply weights and edge signs
        weighted_values = src_values * weights * self.edge_signs  # [batch, num_edges]
        
        # Scatter to destination indices
        # We need to handle the case where multiple edges go to the same destination
        output.scatter_add_(1, self.dst_indices.unsqueeze(0).expand(batch_size, -1), weighted_values)
        
        return output + bias
    
    def forward(self, input: torch.Tensor, hx: torch.Tensor, ts: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
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
        x = torch.cat([input, hx], dim=1)  # [batch, input_size + hidden_size]
        
        # Apply backbone if present
        if self.backbone is not None:
            x = self.backbone(x)
        
        # Compute ff1
        if self.has_sparse_connections and self.backbone is None:
            ff1 = self._sparse_linear(x, self.ff1_weights, self.ff1_bias)
        else:
            ff1 = self.ff1(x)
        
        if self.mode == "pure":
            # Pure CfC solution
            new_hidden = (
                -self.A * torch.exp(-ts * (torch.abs(self.w_tau) + torch.abs(ff1))) * ff1 + self.A
            )
        else:
            # Standard CfC
            if self.has_sparse_connections and self.backbone is None:
                ff2 = self._sparse_linear(x, self.ff2_weights, self.ff2_bias)
                t_a = self._sparse_linear(x, self.time_a_weights, self.time_a_bias)
                t_b = self._sparse_linear(x, self.time_b_weights, self.time_b_bias)
            else:
                ff2 = self.ff2(x)
                t_a = self.time_a(x)
                t_b = self.time_b(x)
            
            ff1 = self.tanh(ff1)
            ff2 = self.tanh(ff2)
            
            # Ensure ts has the right shape
            if ts.dim() == 0:
                ts = ts.unsqueeze(0).expand(batch_size)
            elif ts.dim() == 1 and ts.shape[0] == 1:
                ts = ts.expand(batch_size)
            
            t_interp = self.sigmoid(t_a * ts.unsqueeze(1) + t_b)
            
            if self.mode == "no_gate":
                new_hidden = ff1 + t_interp * ff2
            else:
                new_hidden = ff1 * (1.0 - t_interp) + t_interp * ff2
        
        return new_hidden, new_hidden
    
    @property
    def sparsity_info(self):
        """Return information about the sparsity of this cell."""
        if self.has_sparse_connections:
            total_possible = (self.input_size + self.hidden_size) * self.hidden_size
            return {
                "total_possible_connections": total_possible,
                "actual_connections": self.num_edges,
                "sparsity_level": 1.0 - (self.num_edges / total_possible),
                "memory_savings": f"{total_possible / self.num_edges:.2f}x",
                "parameter_reduction": total_possible - self.num_edges
            }
        else:
            return {
                "total_possible_connections": self.num_edges,
                "actual_connections": self.num_edges,
                "sparsity_level": 0.0,
                "memory_savings": "1x (dense)",
                "parameter_reduction": 0
            }
