#!/usr/bin/env python3
"""Debug script to compare affine transform output."""

import torch
import triton_augment as ta
import triton_augment.functional as F

try:
    import torchvision.transforms.v2.functional as tvF
except ImportError:
    import torchvision.transforms.functional as tvF

def debug_rotate():
    print("=" * 60)
    print("Debug: Rotate 90 degrees")
    print("=" * 60)
    
    # Simple test image
    img = torch.rand(1, 3, 8, 8, device='cuda')
    angle = 90.0
    
    print(f"\nInput image shape: {img.shape}")
    print(f"Input image[0, 0, :4, :4]:\n{img[0, 0, :4, :4]}")
    
    # Torchvision
    tv_result = tvF.rotate(img, angle, interpolation=tvF.InterpolationMode.BILINEAR)
    print(f"\nTorchvision result[0, 0, :4, :4]:\n{tv_result[0, 0, :4, :4]}")
    print(f"Torchvision result stats: min={tv_result.min():.4f}, max={tv_result.max():.4f}, mean={tv_result.mean():.4f}")
    
    # Triton
    ta_result = F.rotate(img, angle, interpolation=ta.InterpolationMode.BILINEAR)
    print(f"\nTriton result[0, 0, :4, :4]:\n{ta_result[0, 0, :4, :4]}")
    print(f"Triton result stats: min={ta_result.min():.4f}, max={ta_result.max():.4f}, mean={ta_result.mean():.4f}")
    
    # Check if Triton result is all zeros or fill value
    print(f"\nTriton result all zeros? {torch.all(ta_result == 0).item()}")
    print(f"Triton result unique values count: {len(torch.unique(ta_result))}")
    
    # Difference
    diff = torch.abs(ta_result - tv_result)
    print(f"\nMax difference: {diff.max():.6f}")
    print(f"Mean difference: {diff.mean():.6f}")

def debug_matrix():
    print("\n" + "=" * 60)
    print("Debug: Check matrix values")
    print("=" * 60)
    
    batch_size = 1
    height, width = 8, 8
    angle = 90.0
    
    # Our matrix calculation
    center_tensor = torch.zeros(batch_size, 2, device='cuda', dtype=torch.float32)
    angle_tensor = torch.tensor([angle], device='cuda', dtype=torch.float32)
    translate_tensor = torch.zeros(batch_size, 2, device='cuda', dtype=torch.float32)
    scale_tensor = torch.ones(batch_size, device='cuda', dtype=torch.float32)
    shear_tensor = torch.zeros(batch_size, 2, device='cuda', dtype=torch.float32)
    
    matrix = F._get_inverse_affine_matrix(
        center_tensor, angle_tensor, translate_tensor, scale_tensor, shear_tensor
    )
    print(f"\nTriton matrix (center=[0,0]): {matrix}")
    
    # Torchvision's matrix for comparison
    from torchvision.transforms.functional import _get_inverse_affine_matrix as tv_get_matrix
    tv_matrix = tv_get_matrix([0.0, 0.0], -angle, [0.0, 0.0], 1.0, [0.0, 0.0])
    print(f"Torchvision matrix (center=[0,0], -angle): {tv_matrix}")

def debug_identity():
    print("\n" + "=" * 60)
    print("Debug: Identity transform (angle=0)")
    print("=" * 60)
    
    img = torch.rand(1, 3, 8, 8, device='cuda')
    
    # Torchvision
    tv_result = tvF.rotate(img, 0, interpolation=tvF.InterpolationMode.BILINEAR)
    
    # Triton
    ta_result = F.rotate(img, 0, interpolation=ta.InterpolationMode.BILINEAR)
    
    diff = torch.abs(ta_result - tv_result)
    print(f"Max difference for identity: {diff.max():.6f}")
    print(f"Input == TV output? {torch.allclose(img, tv_result, atol=1e-5)}")
    print(f"Input == TA output? {torch.allclose(img, ta_result, atol=1e-5)}")

if __name__ == "__main__":
    debug_identity()
    debug_matrix()
    debug_rotate()

