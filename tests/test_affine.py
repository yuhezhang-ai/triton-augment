"""
Tests for affine and rotation transforms.
"""

import pytest
import torch
import triton_augment as ta
import triton_augment.functional as F

try:
    import torchvision.transforms.v2.functional as tvF
except ImportError:
    tvF = None

try:
    from torchvision.transforms.functional import _get_inverse_affine_matrix as tv_get_inverse_affine_matrix
except ImportError:
    try:
        from torchvision.transforms.v2.functional import _get_inverse_affine_matrix as tv_get_inverse_affine_matrix
    except ImportError:
        tv_get_inverse_affine_matrix = None



class TestAffineCorrectness:
    """Test affine operations match torchvision exactly (or reasonably close)."""
    
    @pytest.mark.parametrize("angle", [0, 90, 180, 45, -30])
    @pytest.mark.parametrize("interpolation", ["bilinear", "nearest"])
    def test_rotate_matches_torchvision(self, angle, interpolation):
        """Test that F.rotate matches torchvision for both interpolation modes."""
        img = torch.rand(2, 3, 224, 224, device='cuda')

        # Get interpolation mode enum
        if interpolation == "bilinear":
            tv_interp = tvF.InterpolationMode.BILINEAR
            ta_interp = ta.InterpolationMode.BILINEAR
        else:
            tv_interp = tvF.InterpolationMode.NEAREST
            ta_interp = ta.InterpolationMode.NEAREST

        # Torchvision rotate
        tv_result = tvF.rotate(img, angle, interpolation=tv_interp)

        # Triton rotate
        ta_result = F.rotate(img, angle, interpolation=ta_interp)

        # Allow small tolerance due to float precision and interpolation differences
        # Triton uses float32 accumulation, might differ slightly
        torch.testing.assert_close(ta_result, tv_result, atol=1e-3, rtol=1e-3)

    def test_affine_identity(self):
        """Test that identity affine (no transformation) does nothing."""
        img = torch.rand(2, 3, 224, 224, device='cuda')
        
        # Identity: angle=0, no translate, scale=1, no shear
        result = F.affine(img, angle=0, translate=[0, 0], scale=1.0, shear=[0, 0])
        
        torch.testing.assert_close(result, img, atol=1e-5, rtol=1e-5)

    @pytest.mark.parametrize("batch_size,height,width", [
        (1, 224, 224),   # Single image, square
        (2, 224, 224),   # Small batch, square
        (4, 320, 240),   # Batch, non-square
        (8, 111, 100),   # Larger batch, smaller image
    ])
    @pytest.mark.parametrize("angle,translate,scale,shear", [
        (0.0, [0.0, 0.0], 1.0, [0.0, 0.0]),           # Identity
        (30.0, [10.0, 20.0], 1.2, [5.0, 10.0]),       # Complex transform
        (90.0, [0.0, 0.0], 1.0, [0.0, 0.0]),          # 90 degree rotation
        (-45.0, [15.0, -10.0], 0.8, [0.0, 0.0]),      # Negative angle, scale down
        (15.0, [5.0, 5.0], 1.0, [10.0, -5.0]),        # Shear only
        (0.0, [20.0, 30.0], 1.5, [0.0, 0.0]),         # Translation + scale
    ])
    @pytest.mark.parametrize("interpolation", ["bilinear", "nearest"])
    def test_affine_matches_torchvision(self, batch_size, height, width, angle, translate, scale, shear, interpolation):
        """Test that F.affine matches torchvision with various shapes, parameters, and interpolation modes."""
        img = torch.rand(batch_size, 3, height, width, device='cuda')

        # Get interpolation mode enum
        if interpolation == "bilinear":
            tv_interp = tvF.InterpolationMode.BILINEAR
            ta_interp = ta.InterpolationMode.BILINEAR
        else:
            tv_interp = tvF.InterpolationMode.NEAREST
            ta_interp = ta.InterpolationMode.NEAREST

        # Torchvision affine
        tv_result = tvF.affine(
            img, angle=angle, translate=translate, scale=scale, shear=shear,
            interpolation=tv_interp
        )

        # Triton affine
        ta_result = F.affine(
            img, angle=angle, translate=translate, scale=scale, shear=shear,
            interpolation=ta_interp
        )

        # Allow small tolerance
        torch.testing.assert_close(ta_result, tv_result, atol=1e-3, rtol=1e-3)

class TestTransformClasses:
    """Test RandomAffine and RandomRotation transform classes."""
    
    def test_random_rotation_deterministic(self):
        """Test that RandomRotation produces same results with same seed."""
        transform = ta.TritonRandomRotation(degrees=90)
        img = torch.rand(4, 3, 224, 224, device='cuda')
        
        # Set seed and run
        torch.manual_seed(42)
        result1 = transform(img)
        
        # Reset seed and run again
        torch.manual_seed(42)
        result2 = transform(img)
        
        # Should be identical
        torch.testing.assert_close(result1, result2)
    
    def test_random_affine_deterministic(self):
        """Test that RandomAffine produces same results with same seed."""
        transform = ta.TritonRandomAffine(
            degrees=15,
            translate=(0.1, 0.1),
            scale=(0.8, 1.2),
            shear=10
        )
        img = torch.rand(4, 3, 224, 224, device='cuda')
        
        # Set seed and run
        torch.manual_seed(42)
        result1 = transform(img)
        
        # Reset seed and run again
        torch.manual_seed(42)
        result2 = transform(img)
        
        # Should be identical
        torch.testing.assert_close(result1, result2)
    
    def test_random_rotation_per_image_randomness(self):
        """Test that different images get different rotations when same_on_batch=False."""
        transform = ta.TritonRandomRotation(degrees=180, same_on_batch=False)
        
        # Create batch with SAME image repeated (to test per-image randomness)
        single_img = torch.rand(1, 3, 100, 100, device='cuda')
        img = single_img.repeat(4, 1, 1, 1)  # Same image 4 times
        
        # Apply transform - each image should get different rotation
        torch.manual_seed(42)
        result = transform(img)
        
        # Check that different images in batch got different transforms
        # Compare first image with others
        different_count = 0
        for i in range(1, 4):
            if not torch.allclose(result[0], result[i], atol=1e-5):
                different_count += 1
        
        # At least some images should be different (with high probability for 180 degree range)
        assert different_count > 0, "With same_on_batch=False, different images should get different rotations"
    
    def test_random_affine_same_on_batch(self):
        """Test that same_on_batch=True applies same transform to all images."""
        transform = ta.TritonRandomAffine(
            degrees=45,
            translate=(0.1, 0.1),
            same_on_batch=True
        )
        
        # Create batch with SAME image repeated (to verify same transform applied)
        single_img = torch.rand(1, 3, 100, 100, device='cuda')
        img = single_img.repeat(4, 1, 1, 1)  # Same image 4 times
        
        torch.manual_seed(42)
        result = transform(img)
        
        # Since same_on_batch=True and input is same, all outputs should be identical
        for i in range(1, 4):
            torch.testing.assert_close(result[0], result[i], atol=1e-6, rtol=1e-6)
    
    def test_random_rotation_output_range(self):
        """Test that rotation output values are reasonable."""
        transform = ta.TritonRandomRotation(degrees=45, fill=0.0)
        img = torch.rand(2, 3, 224, 224, device='cuda')
        
        result = transform(img)
        
        # Output should be in valid range [0, 1] (assuming input is)
        # Some pixels might be fill value (0.0)
        assert result.min() >= 0.0
        assert result.max() <= 1.0
    
    def test_random_affine_parameter_validation(self):
        """Test that invalid parameters raise errors."""
        # Negative degrees should raise error
        with pytest.raises(ValueError):
            ta.TritonRandomAffine(degrees=-10)
        
        # Invalid translate range
        with pytest.raises(ValueError):
            ta.TritonRandomAffine(degrees=0, translate=(1.5, 0.5))
        
        # Invalid scale range
        with pytest.raises(ValueError):
            ta.TritonRandomAffine(degrees=0, scale=(1.2, 0.8))  # max < min
    
    def test_random_rotation_3d_input(self):
        """Test that transforms handle 3D input (C, H, W)."""
        transform = ta.TritonRandomRotation(degrees=45)
        img = torch.rand(3, 224, 224, device='cuda')
        
        result = transform(img)
        
        # Should return 3D output
        assert result.shape == img.shape
        assert result.ndim == 3
    
    def test_random_affine_5d_input(self):
        """Test that transforms handle 5D input (N, T, C, H, W) for video."""
        transform = ta.TritonRandomAffine(
            degrees=15,
            translate=(0.1, 0.1),
            same_on_frame=True
        )
        img = torch.rand(2, 8, 3, 112, 112, device='cuda')  # 2 videos, 8 frames each
        
        result = transform(img)
        
        # Should return 5D output
        assert result.shape == img.shape
        assert result.ndim == 5
        


class TestAffineMatrixCorrectness:
    """Test _get_inverse_affine_matrix matches torchvision."""
    
    def test_matrix_calculation_matches_torchvision(self):
        """Test that our matrix calculation matches torchvision for various params."""
        if tv_get_inverse_affine_matrix is None:
            pytest.skip("torchvision internal function not available")
            
        # Test cases: (angle, translate, scale, shear, center)
        test_cases = [
            (0.0, (0.0, 0.0), 1.0, (0.0, 0.0), (0.0, 0.0)),  # Identity
            (90.0, (0.0, 0.0), 1.0, (0.0, 0.0), (112.0, 112.0)),  # 90 deg rotation
            (45.0, (10.0, 20.0), 1.2, (5.0, 5.0), (100.0, 100.0)),  # Complex
            (-30.0, (-5.0, 5.0), 0.8, (10.0, 0.0), (50.0, 50.0)),  # Negative angle/shear
        ]
        
        for angle, translate, scale, shear, center in test_cases:
            # Prepare inputs for Triton (batched)
            # We use batch_size=1 for direct comparison
            center_t = torch.tensor([center], device='cuda')
            angle_t = torch.tensor([angle], device='cuda')
            translate_t = torch.tensor([translate], device='cuda')
            scale_t = torch.tensor([scale], device='cuda')
            shear_t = torch.tensor([shear], device='cuda')
            
            # Calculate Triton matrix
            triton_matrix = F._get_inverse_affine_matrix(
                center_t, angle_t, translate_t, scale_t, shear_t
            )
            
            # Prepare inputs for Torchvision (scalar/list)
            # torchvision expects lists for translate/shear/center
            tv_center = list(center)
            tv_translate = list(translate)
            tv_shear = list(shear)
            tv_scale = float(scale)
            tv_angle = float(angle)
            
            # Calculate Torchvision matrix
            # Note: torchvision returns [a, b, c, d, e, f]
            tv_matrix = tv_get_inverse_affine_matrix(
                tv_center, tv_angle, tv_translate, tv_scale, tv_shear
            )
            tv_matrix_t = torch.tensor(tv_matrix, device='cuda', dtype=torch.float32).unsqueeze(0)
            
            # Compare
            torch.testing.assert_close(triton_matrix, tv_matrix_t, atol=1e-4, rtol=1e-4)

    def test_batched_matrix_calculation(self):
        """Test that batched calculation produces same results as individual calls."""
        if tv_get_inverse_affine_matrix is None:
            pytest.skip("torchvision internal function not available")
            
        batch_size = 4
        
        # Random parameters
        angles = torch.rand(batch_size, device='cuda') * 360
        translates = torch.rand(batch_size, 2, device='cuda') * 50
        scales = torch.rand(batch_size, device='cuda') + 0.5
        shears = torch.rand(batch_size, 2, device='cuda') * 30
        centers = torch.rand(batch_size, 2, device='cuda') * 224
        
        # Calculate batched
        triton_matrices = F._get_inverse_affine_matrix(
            centers, angles, translates, scales, shears
        )
        
        # Calculate individual using torchvision
        for i in range(batch_size):
            tv_matrix = tv_get_inverse_affine_matrix(
                centers[i].tolist(),
                angles[i].item(),
                translates[i].tolist(),
                scales[i].item(),
                shears[i].tolist()
            )
            tv_matrix_t = torch.tensor(tv_matrix, device='cuda', dtype=torch.float32)
            
            torch.testing.assert_close(triton_matrices[i], tv_matrix_t, atol=1e-4, rtol=1e-4)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
