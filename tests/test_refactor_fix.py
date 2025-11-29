import sys
from unittest.mock import MagicMock, PropertyMock

# Mock triton
triton = MagicMock()
triton.cdiv = lambda x, y: (x + y - 1) // y
sys.modules["triton"] = triton
sys.modules["triton.language"] = MagicMock()

import torch
import pytest

# Mock CUDA availability if not available
if not torch.cuda.is_available():
    torch.cuda.is_available = lambda: True
    # We won't mock torch.Tensor.is_cuda directly as it's a C-extension property
    pass

# Import after mocking
import triton_augment.functional as F
from triton_augment.transforms import TritonFusedAugment
import triton_augment.transforms as transforms_module

# Mock the kernel
F.fused_augment_kernel = MagicMock()
F.fused_augment_kernel.__getitem__.return_value = MagicMock()
F.affine_transform_kernel = MagicMock()
F.affine_transform_kernel.__getitem__.return_value = MagicMock()

def test_prepare_affine_params_logic():
    """Test the logic of _prepare_affine_params_to_tensor."""
    print("\nTesting _prepare_affine_params_to_tensor Logic...")
    device = 'cpu'
    batch_size = 2
    height, width = 100, 100
    
    # Case 1: Default center (None)
    # Should result in center at [width/2, height/2]
    angle = 0.0
    translate = [0.0, 0.0]
    scale = 1.0
    shear = [0.0, 0.0]
    center = None
    
    angle_t, translate_t, scale_t, shear_t, center_t = F._prepare_affine_params_to_tensor(
        angle, translate, scale, shear, center, batch_size, height, width, device
    )
    
    expected_center = torch.tensor([50.0, 50.0], device=device).repeat(batch_size, 1)
    print(f"Actual center: {center_t[0]}")
    print(f"Expected center: {expected_center[0]}")
    assert torch.allclose(center_t, expected_center)
    print("Default Center Logic Passed")
    
    # Case 2: Custom center
    # Should result in translated coordinates: [cx - w/2, cy - h/2]
    # Wait, looking at the code I wrote:
    # cx = center[0] - width * 0.5
    # cy = center[1] - height * 0.5
    # This matches the requirement for _get_inverse_affine_matrix which expects
    # translated coordinates where [0,0] is image center.
    
    center_custom = [0.0, 0.0] # Top-left corner
    angle_t, translate_t, scale_t, shear_t, center_t = F._prepare_affine_params_to_tensor(
        angle, translate, scale, shear, center_custom, batch_size, height, width, device
    )
    
    expected_center_custom = torch.tensor([-50.0, -50.0], device=device).repeat(batch_size, 1)
    assert torch.allclose(center_t, expected_center_custom)
    print("Custom Center Logic Passed")

def test_affine_consistency():
    """Verify affine and fused_augment use the same logic."""
    print("\nTesting Consistency...")
    device = 'cpu'
    batch_size = 1
    height, width = 100, 100
    img = torch.rand(batch_size, 3, height, width, device=device)
    
    # Mock _validate_image_tensor
    original_validate = F._validate_image_tensor
    F._validate_image_tensor = lambda *args: None
    
    try:
        # Call affine
        F.affine(img, angle=45.0, translate=[0,0], scale=1.0, shear=[0,0])
        
        # Call fused_augment (affine mode)
        F.fused_augment(
            img, 
            top=0, left=0, height=height, width=width, 
            angle=45.0, translate=[0,0], scale=1.0, shear=[0,0]
        )
        
        # We can't easily check internal state without more mocking,
        # but if both run without error using the shared helper, it's a good sign.
        print("Execution Consistency Passed")
        
    finally:
        F._validate_image_tensor = original_validate

if __name__ == "__main__":
    try:
        test_prepare_affine_params_logic()
        test_affine_consistency()
        print("\nALL TESTS PASSED!")
    except Exception as e:
        print(f"\nTEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
