"""
Tests for individual color operations (functional + transform classes).

Compares Triton-Augment with torchvision to ensure correctness.
Tests both functional API (F.adjust_*) and transform classes (Triton*).
"""

import pytest
import torch
import triton_augment as ta
import triton_augment.functional as F

try:
    import torchvision.transforms.v2.functional as tvF
except ImportError:
    tvF = None


class TestBrightnessCorrectness:
    """Test brightness adjustment correctness against torchvision."""
    
    @pytest.mark.parametrize("brightness_factor", [0.0, 0.5, 1.0, 1.5, 2.0])
    @pytest.mark.parametrize("shape", [(2, 3, 64, 64), (1, 3, 224, 224)])
    def test_brightness_functional_matches_torchvision(self, brightness_factor, shape):
        """Test that F.adjust_brightness matches torchvision exactly."""
        img = torch.rand(*shape, device='cuda', dtype=torch.float32)
        
        tv_result = tvF.adjust_brightness(img, brightness_factor)
        ta_result = F.adjust_brightness(img, brightness_factor)
        
        torch.testing.assert_close(
            ta_result,
            tv_result,
            msg=f"Brightness mismatch for factor={brightness_factor}, shape={shape}"
        )
    
    def test_brightness_zero_produces_black(self):
        """Test brightness=0 produces black image."""
        img = torch.rand(2, 3, 64, 64, device='cuda', dtype=torch.float32)
        result = F.adjust_brightness(img, 0.0)
        torch.testing.assert_close(result, torch.zeros_like(result), rtol=1e-5, atol=1e-5)


class TestContrastCorrectness:
    """Test contrast adjustment correctness against torchvision."""
    
    @pytest.mark.parametrize("contrast_factor", [0.0, 0.5, 1.0, 1.5, 2.0])
    @pytest.mark.parametrize("shape", [(2, 3, 64, 64), (1, 3, 224, 224)])
    def test_contrast_functional_matches_torchvision(self, contrast_factor, shape):
        """Test that F.adjust_contrast (torchvision-exact) matches torchvision."""
        img = torch.rand(*shape, device='cuda', dtype=torch.float32)
        
        tv_result = tvF.adjust_contrast(img, contrast_factor)
        ta_result = F.adjust_contrast(img, contrast_factor)
        
        torch.testing.assert_close(
            ta_result,
            tv_result,
            msg=f"Contrast mismatch for factor={contrast_factor}, shape={shape}"
        )
    
    def test_contrast_identity(self):
        """Test that contrast=1.0 produces identical results."""
        img = torch.rand(2, 3, 64, 64, device='cuda', dtype=torch.float32)
        result = F.adjust_contrast(img, 1.0)
        torch.testing.assert_close(result, img, rtol=1e-4, atol=1e-4)


class TestContrastFastCorrectness:
    """Test fast contrast adjustment (does NOT match torchvision)."""
    
    @pytest.mark.parametrize("contrast_factor", [0.0, 0.5, 1.0, 1.5, 2.0])
    @pytest.mark.parametrize("shape", [(2, 3, 64, 64), (1, 3, 224, 224)])
    def test_contrast_fast_formula(self, contrast_factor, shape):
        """Test that fast contrast uses correct formula: (pixel - 0.5) * factor + 0.5"""
        img = torch.rand(*shape, device='cuda', dtype=torch.float32)
        ta_result = F.adjust_contrast_fast(img, contrast_factor)
        
        # Manual calculation
        expected = (img - 0.5) * contrast_factor + 0.5
        expected = torch.clamp(expected, 0.0, 1.0)
        
        torch.testing.assert_close(
            ta_result,
            expected,
            msg=f"Fast contrast mismatch for factor={contrast_factor}, shape={shape}"
        )
    
    def test_contrast_fast_differs_from_torchvision(self):
        """Verify that fast contrast produces different results from torchvision."""
        img = torch.linspace(0, 1, 2*3*128*128, device='cuda').reshape(2, 3, 128, 128)
        contrast_factor = 1.5
        
        tv_result = tvF.adjust_contrast(img, contrast_factor)
        ta_fast_result = F.adjust_contrast_fast(img, contrast_factor)
        
        max_diff = torch.abs(ta_fast_result - tv_result).max().item()
        assert max_diff > 0.01, f"Fast contrast too similar to torchvision (max_diff={max_diff})"


class TestSaturationCorrectness:
    """Test saturation adjustment correctness against torchvision."""
    
    @pytest.mark.parametrize("saturation_factor", [0.0, 0.5, 1.0, 1.5, 2.0])
    @pytest.mark.parametrize("shape", [(2, 3, 64, 64), (1, 3, 224, 224)])
    def test_saturation_functional_matches_torchvision(self, saturation_factor, shape):
        """Test that F.adjust_saturation matches torchvision exactly."""
        img = torch.rand(*shape, device='cuda', dtype=torch.float32)
        
        tv_result = tvF.adjust_saturation(img, saturation_factor)
        ta_result = F.adjust_saturation(img, saturation_factor)
        
        torch.testing.assert_close(
            ta_result,
            tv_result,
            msg=f"Saturation mismatch for factor={saturation_factor}, shape={shape}"
        )
    
    def test_saturation_zero_produces_grayscale(self):
        """Test saturation=0 produces grayscale."""
        img = torch.rand(2, 3, 64, 64, device='cuda', dtype=torch.float32)
        result = F.adjust_saturation(img, 0.0)
        
        # Check that all channels are equal (grayscale)
        r, g, b = result[:, 0], result[:, 1], result[:, 2]
        torch.testing.assert_close(r, g, rtol=1e-4, atol=1e-4)
        torch.testing.assert_close(g, b, rtol=1e-4, atol=1e-4)
    
    def test_saturation_identity(self):
        """Test that saturation=1.0 produces identical results."""
        img = torch.rand(2, 3, 64, 64, device='cuda', dtype=torch.float32)
        result = F.adjust_saturation(img, 1.0)
        torch.testing.assert_close(result, img, rtol=1e-4, atol=1e-4)


class TestNormalizeCorrectness:
    """Test normalization correctness against torchvision."""
    
    @pytest.mark.parametrize("shape", [(2, 3, 64, 64), (1, 3, 224, 224), (4, 3, 128, 128)])
    @pytest.mark.parametrize("mean,std", [
        ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),  # ImageNet
        ((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),  # Simple
        ((0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),  # Identity
    ])
    def test_normalize_functional_matches_torchvision(self, shape, mean, std):
        """Test that F.normalize matches torchvision exactly."""
        img = torch.rand(*shape, device='cuda', dtype=torch.float32)
        
        tv_result = tvF.normalize(img, mean=list(mean), std=list(std))
        ta_result = F.normalize(img, mean=mean, std=std)
        
        torch.testing.assert_close(
            ta_result,
            tv_result,
            msg=f"Normalize mismatch for mean={mean}, std={std}, shape={shape}"
        )
    
    def test_normalize_deterministic(self):
        """Test that normalize produces consistent results."""
        img = torch.rand(2, 3, 128, 128, device='cuda', dtype=torch.float32)
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
        
        result1 = F.normalize(img, mean=mean, std=std)
        result2 = F.normalize(img, mean=mean, std=std)
        
        torch.testing.assert_close(result1, result2, msg="Normalize should be deterministic")
    
    def test_normalize_transform_class(self):
        """Test TritonNormalize transform class."""
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
        transform = ta.TritonNormalize(mean=mean, std=std)
        
        img = torch.rand(2, 3, 128, 128, device='cuda', dtype=torch.float32)
        result = transform(img)
        
        # Should match functional API
        expected = F.normalize(img, mean=mean, std=std)
        torch.testing.assert_close(result, expected)


class TestGrayscaleCorrectness:
    """Test grayscale conversion correctness against torchvision."""
    
    @pytest.mark.parametrize("num_output_channels", [1, 3])
    @pytest.mark.parametrize("shape", [(2, 3, 64, 64), (1, 3, 224, 224)])
    def test_rgb_to_grayscale_functional_matches_torchvision(self, num_output_channels, shape):
        """Test that F.rgb_to_grayscale matches torchvision exactly."""
        img = torch.rand(*shape, device='cuda', dtype=torch.float32)
        
        tv_result = tvF.rgb_to_grayscale(img, num_output_channels=num_output_channels)
        ta_result = F.rgb_to_grayscale(img, num_output_channels=num_output_channels)
        
        torch.testing.assert_close(
            ta_result,
            tv_result,
            msg=f"Grayscale mismatch for num_output_channels={num_output_channels}, shape={shape}"
        )
        
        # Verify output shape
        expected_shape = (shape[0], num_output_channels, shape[2], shape[3])
        assert ta_result.shape == expected_shape
    
    def test_grayscale_deterministic(self):
        """Test that grayscale conversion is deterministic."""
        img = torch.rand(2, 3, 128, 128, device='cuda', dtype=torch.float32)
        
        result1 = F.rgb_to_grayscale(img, num_output_channels=3)
        result2 = F.rgb_to_grayscale(img, num_output_channels=3)
        
        torch.testing.assert_close(result1, result2)
    
    def test_rgb_to_grayscale_with_mask_all_false(self):
        """Test that rgb_to_grayscale with all-false mask returns original image."""
        img = torch.rand(2, 3, 128, 128, device='cuda', dtype=torch.float32)
        grayscale_mask = torch.zeros(2, device='cuda', dtype=torch.uint8)  # All false
        
        result = F.rgb_to_grayscale(img, num_output_channels=3, grayscale_mask=grayscale_mask)
        
        # Should be unchanged
        torch.testing.assert_close(result, img)
    
    def test_rgb_to_grayscale_with_mask_all_true(self):
        """Test that rgb_to_grayscale with all-true mask converts to grayscale."""
        img = torch.rand(2, 3, 128, 128, device='cuda', dtype=torch.float32)
        grayscale_mask = torch.ones(2, device='cuda', dtype=torch.uint8)  # All true
        
        result = F.rgb_to_grayscale(img, num_output_channels=3, grayscale_mask=grayscale_mask)
        
        # Should be grayscale (all channels identical)
        assert torch.allclose(result[:, 0], result[:, 1])
        assert torch.allclose(result[:, 1], result[:, 2])
    
    def test_triton_grayscale_transform_class(self):
        """Test TritonGrayscale transform class."""
        transform = ta.TritonGrayscale(num_output_channels=3)
        img = torch.rand(2, 3, 128, 128, device='cuda', dtype=torch.float32)
        
        result = transform(img)
        
        # Should be grayscale
        assert result.shape == (2, 3, 128, 128)
        assert torch.allclose(result[:, 0], result[:, 1])
        assert torch.allclose(result[:, 1], result[:, 2])
    
    def test_triton_random_grayscale_p_zero(self):
        """Test that TritonRandomGrayscale with p=0 returns original image."""
        transform = ta.TritonRandomGrayscale(p=0.0, num_output_channels=3)
        img = torch.rand(2, 3, 128, 128, device='cuda', dtype=torch.float32)
        result = transform(img)
        torch.testing.assert_close(result, img)
    
    def test_triton_random_grayscale_p_one(self):
        """Test that TritonRandomGrayscale with p=1 always converts to grayscale."""
        transform = ta.TritonRandomGrayscale(p=1.0, num_output_channels=3)
        img = torch.rand(2, 3, 128, 128, device='cuda', dtype=torch.float32)
        result = transform(img)
        assert torch.allclose(result[:, 0], result[:, 1])


class TestColorJitterTransformClass:
    """Test TritonColorJitter transform class."""
    
    def test_color_jitter_parameter_ranges(self):
        """Test that parameter ranges are computed correctly."""
        # Test symmetric range
        transform = ta.TritonColorJitter(brightness=0.2, contrast=0.3, saturation=0.1)
        assert transform.brightness == (0.8, 1.2), f"Expected (0.8, 1.2), got {transform.brightness}"
        assert transform.contrast == (0.7, 1.3), f"Expected (0.7, 1.3), got {transform.contrast}"
        assert transform.saturation == (0.9, 1.1), f"Expected (0.9, 1.1), got {transform.saturation}"
        
        # Test custom range
        transform = ta.TritonColorJitter(brightness=(0.5, 1.5), contrast=(0.8, 1.2))
        assert transform.brightness == (0.5, 1.5)
        assert transform.contrast == (0.8, 1.2)
    
    def test_color_jitter_produces_valid_output(self):
        """Test that color jitter produces valid output shapes."""
        transform = ta.TritonColorJitter(brightness=0.2, contrast=0.2, saturation=0.2)
        img = torch.rand(2, 3, 64, 64, device='cuda', dtype=torch.float32)
        
        result = transform(img)
        
        assert result.shape == img.shape
        assert result.device == img.device
        assert result.dtype == img.dtype


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

