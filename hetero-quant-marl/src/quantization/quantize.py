"""
Post-training quantization utilities for neural networks.
Simulates quantization effects by quantizing weights to target bit-width.
"""

import torch
import torch.nn as nn
import copy


def quantize_tensor(tensor, bits=8, symmetric=True):
    """
    Quantize a tensor to specified bit-width.
    
    Args:
        tensor: Input tensor (weights)
        bits: Target bit-width (32, 8, 4, 2)
        symmetric: If True, use symmetric quantization
    
    Returns:
        Quantized tensor (still float, but with reduced precision)
    """
    if bits >= 32:
        return tensor  # No quantization
    
    with torch.no_grad():
        if symmetric:
            # Symmetric quantization: [-qmax, qmax]
            qmax = 2**(bits-1) - 1
            scale = tensor.abs().max() / qmax if tensor.abs().max() > 0 else 1.0
            
            # Quantize
            tensor_q = torch.clamp(torch.round(tensor / scale), -qmax, qmax)
            # Dequantize
            tensor_dq = tensor_q * scale
        else:
            # Asymmetric quantization: [0, qmax]
            qmax = 2**bits - 1
            t_min, t_max = tensor.min(), tensor.max()
            scale = (t_max - t_min) / qmax if (t_max - t_min) > 0 else 1.0
            
            # Quantize
            tensor_q = torch.clamp(torch.round((tensor - t_min) / scale), 0, qmax)
            # Dequantize
            tensor_dq = tensor_q * scale + t_min
        
        return tensor_dq


def quantize_model(model, bits=8):
    """
    Apply post-training quantization to all Linear layers.
    
    Args:
        model: PyTorch model
        bits: Target bit-width (32=FP32, 8=INT8, 4=INT4, 2=INT2)
    
    Returns:
        Quantized model (new copy)
    """
    quantized_model = copy.deepcopy(model)
    
    for name, module in quantized_model.named_modules():
        if isinstance(module, nn.Linear):
            module.weight.data = quantize_tensor(module.weight.data, bits)
            if module.bias is not None:
                module.bias.data = quantize_tensor(module.bias.data, bits)
    
    return quantized_model


def quantize_team_heterogeneous(networks_dict, precision_assignment):
    """
    Apply heterogeneous quantization to a team of agents.
    
    Args:
        networks_dict: Dict mapping agent_id -> network
        precision_assignment: Dict mapping agent_id -> bits
            e.g., {"agent_0": 8, "agent_1": 4, "agent_2": 4}
    
    Returns:
        New dict with heterogeneous quantization applied
    """
    new_networks = {}
    for agent_id, net in networks_dict.items():
        bits = precision_assignment.get(agent_id, 32)
        new_networks[agent_id] = quantize_model(net, bits)
    return new_networks


def get_model_size_bits(model, bits=32):
    """Calculate model size in bits at given precision."""
    total_params = 0
    for param in model.parameters():
        total_params += param.numel()
    return total_params * bits


def count_unique_weights(model):
    """Count unique weight values (sanity check for quantization)."""
    all_weights = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            all_weights.append(module.weight.data.flatten())
    if all_weights:
        all_weights = torch.cat(all_weights)
        return len(torch.unique(all_weights))
    return 0