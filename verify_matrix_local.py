
import sys
from unittest.mock import MagicMock
import torch

# Mock triton
sys.modules["triton"] = MagicMock()
sys.modules["triton.language"] = MagicMock()

# Now we can import functional
# We need to make sure we don't hit other import errors
# The functional module imports kernels, which import triton.
# Since we mocked triton, the kernel decorators @triton.jit might fail if not mocked properly?
# MagicMock should handle it.

try:
    from triton_augment.functional import _get_inverse_affine_matrix
except ImportError as e:
    print(f"Import failed: {e}")
    # It might fail if it tries to import from .kernels which might have other issues
    # Let's try to see if we can just import the function if it was isolated.
    # But it is not.
    # Let's hope the mock works.
    sys.exit(1)

try:
    from torchvision.transforms.functional import _get_inverse_affine_matrix as tv_get_inverse_affine_matrix
except ImportError:
    try:
        from torchvision.transforms.v2.functional import _get_inverse_affine_matrix as tv_get_inverse_affine_matrix
    except ImportError:
        print("Torchvision not found")
        sys.exit(1)

def test_matrix():
    print("Testing matrix calculation...")

    # Test cases matching test_affine.py exactly
    test_cases = [
        (0.0, (0.0, 0.0), 1.0, (0.0, 0.0), (0.0, 0.0)),  # Identity
        (90.0, (0.0, 0.0), 1.0, (0.0, 0.0), (112.0, 112.0)),  # 90 deg rotation
        (45.0, (10.0, 20.0), 1.2, (5.0, 5.0), (100.0, 100.0)),  # Complex
        (-30.0, (-5.0, 5.0), 0.8, (10.0, 0.0), (50.0, 50.0)),  # Negative angle/shear
    ]

    for i, (angle, translate, scale, shear, center) in enumerate(test_cases):
        print(f"\nTest case {i}: angle={angle}, translate={translate}, scale={scale}, shear={shear}, center={center}")

        # Triton input (batched)
        center_t = torch.tensor([center], device='cpu')
        angle_t = torch.tensor([angle], device='cpu')
        translate_t = torch.tensor([translate], device='cpu')
        scale_t = torch.tensor([scale], device='cpu')
        shear_t = torch.tensor([shear], device='cpu')

        triton_matrix = _get_inverse_affine_matrix(
            center_t, angle_t, translate_t, scale_t, shear_t
        )

        # Torchvision input
        tv_matrix = tv_get_inverse_affine_matrix(
            list(center), float(angle), list(translate), float(scale), list(shear)
        )
        tv_matrix_t = torch.tensor(tv_matrix, dtype=torch.float32).unsqueeze(0)

        # Compare
        if torch.allclose(triton_matrix, tv_matrix_t, atol=1e-4, rtol=1e-4):
            print(f"  SUCCESS: Matrices match!")
        else:
            print(f"  FAILURE: Matrices do not match!")
            print(f"  Triton: {triton_matrix.flatten().tolist()}")
            print(f"  TV:     {tv_matrix_t.flatten().tolist()}")
            diff = torch.abs(triton_matrix - tv_matrix_t)
            print(f"  Max diff: {diff.max().item()}")

if __name__ == "__main__":
    test_matrix()
