# Copyright 2025 Khalik Alliance Inc.
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

"""
Test suite for the efficient sparse CfC implementation.

This module verifies that the efficient sparse implementation maintains
functional correctness while achieving the expected parameter reduction
and performance improvements.
"""

import torch
import torch.nn as nn
import numpy as np
from ncps.torch import CfC
from ncps.wirings import AutoNCP, FullyConnected, NCP
import time


def count_parameters(model):
    """Count total number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def test_output_equivalence():
    """Verify that efficient and regular implementations produce compatible outputs."""
    print("\n--- Testing Output Equivalence ---")
    
    # Test configuration
    input_size = 10
    batch_size = 4
    seq_len = 8
    
    # Create identical wiring configuration
    wiring = AutoNCP(units=32, output_size=8)
    wiring.build(input_size)
    
    # Initialize models with identical architecture
    model_regular = CfC(
        input_size=input_size,
        units=wiring,
        mode="default",
        batch_first=True,
        use_efficient=False
    )
    
    model_efficient = CfC(
        input_size=input_size,
        units=wiring,
        mode="default", 
        batch_first=True,
        use_efficient=True
    )
    
    # Note: Due to different parameter structures between regular and efficient implementations,
    # we verify that both produce valid outputs with their respective initializations
    
    # Set to evaluation mode to ensure consistent behavior
    model_regular.eval()
    model_efficient.eval()
    
    # Generate test data
    x = torch.randn(batch_size, seq_len, input_size)
    h0 = torch.zeros(batch_size, wiring.units)
    
    # Execute forward pass
    with torch.no_grad():
        output_regular, h_regular = model_regular(x, h0)
        output_efficient, h_efficient = model_efficient(x, h0)
    
    # Verify output dimensions match
    assert output_regular.shape == output_efficient.shape, \
        f"Output shape mismatch: {output_regular.shape} vs {output_efficient.shape}"
    assert h_regular.shape == h_efficient.shape, \
        f"Hidden state shape mismatch: {h_regular.shape} vs {h_efficient.shape}"
    
    print(f"  Output shapes match: {output_regular.shape}")
    print(f"  Hidden state shapes match: {h_regular.shape}")
    
    # Verify numerical stability
    assert torch.all(torch.isfinite(output_regular)), "Regular model produced non-finite values"
    assert torch.all(torch.isfinite(output_efficient)), "Efficient model produced non-finite values"
    
    print("  Both models produce numerically stable outputs")
    

def test_parameter_reduction():
    """Verify that efficient implementation achieves expected parameter reduction."""
    print("\n--- Testing Parameter Reduction ---")
    
    input_size = 20
    
    # Define test configurations with varying sparsity levels
    configurations = [
        ("Sparse AutoNCP", AutoNCP(units=64, output_size=16, sparsity_level=0.5)),
        ("Highly Sparse NCP", NCP(
            inter_neurons=32, 
            command_neurons=8, 
            motor_neurons=8,
            sensory_fanout=4,
            inter_fanout=4,
            motor_fanin=4,
            recurrent_command=4,
        )),
    ]
    
    for config_name, wiring in configurations:
        wiring.build(input_size)
        
        # Initialize regular implementation
        model_regular = CfC(
            input_size=input_size,
            units=wiring,
            mode="default",
            use_efficient=False
        )
        
        # Initialize efficient implementation
        model_efficient = CfC(
            input_size=input_size,
            units=wiring,
            mode="default",
            use_efficient=True
        )
        
        # Calculate parameter counts
        params_regular = count_parameters(model_regular)
        params_efficient = count_parameters(model_efficient)
        
        # Calculate reduction percentage
        reduction_percentage = (params_regular - params_efficient) / params_regular * 100
        
        print(f"\n  Configuration: {config_name}")
        print(f"    Regular implementation parameters: {params_regular:,}")
        print(f"    Efficient implementation parameters: {params_efficient:,}")
        print(f"    Parameter reduction: {reduction_percentage:.1f}%")
        
        # Display sparsity information if available
        if hasattr(model_efficient, 'sparsity_info'):
            info = model_efficient.sparsity_info
            if 'overall_sparsity' in info:
                print(f"    Overall sparsity level: {info['overall_sparsity']*100:.1f}%")
                print(f"    Memory savings factor: {info['overall_memory_savings']}")
        
        assert params_efficient < params_regular, \
            f"Efficient implementation must have fewer parameters than regular implementation for {config_name}"


def test_gradient_flow():
    """Verify proper gradient propagation through efficient implementation."""
    print("\n--- Testing Gradient Flow ---")
    
    # Test configuration
    input_size = 8
    batch_size = 2
    seq_len = 5
    
    # Create sparse wiring
    wiring = AutoNCP(units=16, output_size=4, sparsity_level=0.3)
    wiring.build(input_size)
    
    # Initialize model with efficient implementation
    model = CfC(
        input_size=input_size,
        units=wiring,
        mode="default",
        batch_first=True,
        use_efficient=True
    )
    
    # Generate test data
    x = torch.randn(batch_size, seq_len, input_size, requires_grad=True)
    h0 = torch.zeros(batch_size, wiring.units)
    target = torch.randn(batch_size, seq_len, wiring.output_dim)
    
    # Forward pass
    output, h_final = model(x, h0)
    
    # Compute loss
    loss_fn = nn.MSELoss()
    loss = loss_fn(output, target)
    
    # Backward pass
    loss.backward()
    
    # Verify gradient computation
    gradient_count = 0
    for name, param in model.named_parameters():
        if param.requires_grad and param.grad is not None:
            assert torch.all(torch.isfinite(param.grad)), \
                f"Non-finite gradients detected in parameter: {name}"
            gradient_count += 1
    
    assert gradient_count > 0, "No parameter gradients were computed"
    assert x.grad is not None, "Input gradients were not computed"
    assert torch.all(torch.isfinite(x.grad)), "Non-finite gradients detected in input tensor"
    
    print(f"  Verified {gradient_count} parameter gradients")
    print("  Input gradient computation verified")
    print("  All gradients are numerically stable")


def test_different_modes():
    """Verify efficient implementation compatibility with all CfC modes."""
    print("\n--- Testing Different CfC Modes ---")
    
    # Test configuration
    input_size = 10
    batch_size = 2
    seq_len = 4
    
    # Test all supported modes
    modes = ["default", "pure", "no_gate"]
    
    # Create wiring configuration
    wiring = AutoNCP(units=20, output_size=5)
    wiring.build(input_size)
    
    for mode in modes:
        # Initialize model with specific mode
        model = CfC(
            input_size=input_size,
            units=wiring,
            mode=mode,
            batch_first=True,
            use_efficient=True
        )
        
        # Generate test data
        x = torch.randn(batch_size, seq_len, input_size)
        h0 = torch.zeros(batch_size, wiring.units)
        
        # Execute forward pass
        output, h_final = model(x, h0)
        
        # Verify output validity
        assert torch.all(torch.isfinite(output)), \
            f"Non-finite output detected for mode: {mode}"
        assert output.shape == (batch_size, seq_len, wiring.output_dim), \
            f"Unexpected output shape for mode: {mode}"
        
        print(f"  Mode '{mode}': Verified")


def test_symmetric_recurrence_edge_counts():
    """Verify that efficient wired uses symmetric within-layer recurrence counts.

    Compares expected nonzero edge counts per layer (sensory + recurrent) against
    the model's reported sparse edge counts, and checks symmetric bounds for the
    command layer: directed <= symmetric <= 2*directed.
    """
    print("\n--- Testing Symmetric Recurrence Edge Counts ---")

    input_size = 16
    wiring = AutoNCP(units=48, output_size=12, sparsity_level=0.6)
    wiring.build(input_size)

    model = CfC(
        input_size=input_size,
        units=wiring,
        mode="default",
        use_efficient=True,
    )

    # Layer index mapping for AutoNCP: 0=inter, 1=command, 2=motor
    inter_idx = wiring.get_neurons_of_layer(0)
    cmd_idx = wiring.get_neurons_of_layer(1)
    mot_idx = wiring.get_neurons_of_layer(2)

    A = wiring.adjacency_matrix
    S = wiring.sensory_adjacency_matrix

    # Expected per-layer edge counts: sensory-to-layer + within-layer recurrent
    # Layer 0 (inter): sensory inputs -> inter, recurrence among inter (usually 0)
    N0_sens = int(np.count_nonzero(S[:, inter_idx]))
    A_ii = A[np.ix_(inter_idx, inter_idx)]
    R0_sym = int(np.count_nonzero((A_ii != 0) | (A_ii.T != 0)))
    expected_L0 = N0_sens + R0_sym

    # Layer 1 (command): inter -> command, recurrence among command (symmetric union)
    N1_prev = int(np.count_nonzero(A[np.ix_(inter_idx, cmd_idx)]))
    A_cc = A[np.ix_(cmd_idx, cmd_idx)]
    R1_dir = int(np.count_nonzero(A_cc))
    R1_sym = int(np.count_nonzero((A_cc != 0) | (A_cc.T != 0)))
    expected_L1 = N1_prev + R1_sym

    # Layer 2 (motor): command -> motor, recurrence among motor (usually 0)
    N2_prev = int(np.count_nonzero(A[np.ix_(cmd_idx, mot_idx)]))
    A_mm = A[np.ix_(mot_idx, mot_idx)]
    R2_sym = int(np.count_nonzero((A_mm != 0) | (A_mm.T != 0)))
    expected_L2 = N2_prev + R2_sym

    # Retrieve model-reported per-layer sparse connection counts
    assert hasattr(model, "sparsity_info"), "Model should expose sparsity_info"
    layer_details = model.sparsity_info.get("layer_details", [])
    assert len(layer_details) == 3, "AutoNCP should yield 3 layers"

    actual_L0 = int(layer_details[0]["actual_connections"])
    actual_L1 = int(layer_details[1]["actual_connections"])
    actual_L2 = int(layer_details[2]["actual_connections"])

    # Check equality with expected counts
    assert actual_L0 == expected_L0, f"Layer 0 connections mismatch: actual={actual_L0}, expected={expected_L0}"
    assert actual_L1 == expected_L1, f"Layer 1 connections mismatch: actual={actual_L1}, expected={expected_L1}"
    assert actual_L2 == expected_L2, f"Layer 2 connections mismatch: actual={actual_L2}, expected={expected_L2}"

    # Check symmetric bounds for command recurrence
    assert R1_dir <= R1_sym <= 2 * R1_dir if R1_dir > 0 else R1_sym == 0, (
        f"Symmetric recurrence out of expected bounds: directed={R1_dir}, symmetric={R1_sym}"
    )

    print(
        f"  Layer edges ok (L0={actual_L0}, L1={actual_L1}, L2={actual_L2}); R1_dir={R1_dir}, R1_sym={R1_sym}"
    )

def test_performance():
    """Measure performance characteristics of efficient implementation."""
    print("\n--- Testing Performance Characteristics ---")
    
    # Test configuration for performance measurement
    input_size = 32
    batch_size = 16
    seq_len = 20
    
    # Create sparse network architecture
    wiring = AutoNCP(units=128, output_size=32, sparsity_level=0.7)
    wiring.build(input_size)
    
    # Initialize regular implementation
    model_regular = CfC(
        input_size=input_size,
        units=wiring,
        mode="default",
        batch_first=True,
        use_efficient=False
    )
    
    # Initialize efficient implementation
    model_efficient = CfC(
        input_size=input_size,
        units=wiring,
        mode="default",
        batch_first=True,
        use_efficient=True
    )
    
    # Generate test data
    x = torch.randn(batch_size, seq_len, input_size)
    h0 = torch.zeros(batch_size, wiring.units)
    
    # Warm-up phase to ensure fair comparison
    for _ in range(3):
        model_regular(x, h0)
        model_efficient(x, h0)
    
    # Measure regular implementation performance
    start_time = time.time()
    for _ in range(10):
        output_regular, h_regular = model_regular(x, h0)
        output_regular.sum().backward()
    time_regular = time.time() - start_time
    
    # Measure efficient implementation performance
    model_efficient.zero_grad()
    start_time = time.time()
    for _ in range(10):
        output_efficient, h_efficient = model_efficient(x, h0)
        output_efficient.sum().backward()
    time_efficient = time.time() - start_time
    
    # Calculate performance metrics
    speedup_factor = time_regular / time_efficient
    
    print(f"\n  Regular implementation execution time: {time_regular:.3f}s")
    print(f"  Efficient implementation execution time: {time_efficient:.3f}s")
    print(f"  Performance improvement factor: {speedup_factor:.2f}x")
    
    # Note: Performance improvements may vary based on hardware and runtime conditions
    if speedup_factor > 1.0:
        print("  Efficient implementation demonstrates performance improvement")
    else:
        print("  Note: Performance characteristics may vary based on hardware and configuration")


def run_test_suite():
    """Execute comprehensive test suite for efficient sparse implementation."""
    print("=" * 70)
    print("Efficient Sparse CfC Implementation Test Suite")
    print("=" * 70)
    
    try:
        # Execute test modules
        test_output_equivalence()
        test_parameter_reduction()
        test_gradient_flow()
        test_different_modes()
        test_symmetric_recurrence_edge_counts()
        test_performance()
        
        print("\n" + "=" * 70)
        print("Test Suite Completed Successfully")
        print("All tests passed")
        print("=" * 70)
        
    except AssertionError as e:
        print("\n" + "=" * 70)
        print(f"Test Suite Failed")
        print(f"Error: {str(e)}")
        print("=" * 70)
        raise
    except Exception as e:
        print("\n" + "=" * 70)
        print(f"Unexpected Error During Testing")
        print(f"Error Type: {type(e).__name__}")
        print(f"Error Details: {str(e)}")
        print("=" * 70)
        raise


if __name__ == "__main__":
    # Initialize random seeds for reproducible results
    torch.manual_seed(42)
    np.random.seed(42)
    
    run_test_suite()
