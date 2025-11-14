"""
Tests for float16 (half precision) support.

Tests both correctness (float16 vs torchvision float16) and precision
(float16 vs float32 comparisons).
"""

import pytest
import torch
import triton_augment as ta
import triton_augment.functional as F

try:
    import torchvision.transforms.v2.functional as tvF
except ImportError:
    tvF = None


class TestFloat16Precision:
    """Test float16 vs float32 precision comparisons."""
    
    @pytest.mark.parametrize("brightness_factor", [0.5, 1.0, 1.5])
    def test_brightness_float16_vs_float32(self, brightness_factor):
        """Test that float16 brightness matches float32 within appropriate tolerance."""
        # Create test images in both dtypes
        img_fp32 = torch.rand(2, 3, 224, 224, device='cuda', dtype=torch.float32)
        img_fp16 = img_fp32.to(torch.float16)
        
        # Apply brightness in both dtypes
        result_fp32 = F.adjust_brightness(img_fp32, brightness_factor)
        result_fp16 = F.adjust_brightness(img_fp16, brightness_factor)
        
        # Compare (float16 has lower precision, so use relaxed tolerance)
        # float16 has ~3-4 decimal digits of precision
        torch.testing.assert_close(
            result_fp16.float(), 
            result_fp32, 
            rtol=1e-3,  # 0.1% relative tolerance
            atol=1e-3   # 0.001 absolute tolerance
        )
    
    @pytest.mark.parametrize("contrast_factor", [0.5, 1.0, 1.5])
    def test_contrast_float16_vs_float32(self, contrast_factor):
        """Test that float16 contrast matches float32 within appropriate tolerance."""
        img_fp32 = torch.rand(2, 3, 224, 224, device='cuda', dtype=torch.float32)
        img_fp16 = img_fp32.to(torch.float16)
        
        # Apply contrast
        result_fp32 = F.adjust_contrast(img_fp32, contrast_factor)
        result_fp16 = F.adjust_contrast(img_fp16, contrast_factor)
        
        torch.testing.assert_close(
            result_fp16.float(), 
            result_fp32, 
            rtol=1e-3,
            atol=1e-3
        )
    
    @pytest.mark.parametrize("saturation_factor", [0.0, 0.5, 1.0, 1.5])
    def test_saturation_float16_vs_float32(self, saturation_factor):
        """Test that float16 saturation matches float32 within appropriate tolerance."""
        img_fp32 = torch.rand(2, 3, 224, 224, device='cuda', dtype=torch.float32)
        img_fp16 = img_fp32.to(torch.float16)
        
        # Apply saturation
        result_fp32 = F.adjust_saturation(img_fp32, saturation_factor)
        result_fp16 = F.adjust_saturation(img_fp16, saturation_factor)
        
        torch.testing.assert_close(
            result_fp16.float(), 
            result_fp32, 
            rtol=1e-3,
            atol=1e-3
        )
    
    def test_normalize_float16_vs_float32(self):
        """Test that float16 normalization matches float32 within appropriate tolerance."""
        img_fp32 = torch.rand(2, 3, 224, 224, device='cuda', dtype=torch.float32)
        img_fp16 = img_fp32.to(torch.float16)
        
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
        
        # Apply normalization
        result_fp32 = F.normalize(img_fp32, mean, std)
        result_fp16 = F.normalize(img_fp16, mean, std)
        
        # Normalize involves division by small std values (~0.2), which amplifies
        # float16 precision errors. Relaxed tolerance for precision comparison.
        torch.testing.assert_close(
            result_fp16.float(), 
            result_fp32, 
            rtol=0.01,   # 1% relative tolerance (relaxed from 0.1%)
            atol=0.01    # 0.01 absolute tolerance (relaxed from 0.001)
        )
    
    def test_fused_float16_vs_float32(self):
        """Test that fused kernel produces similar results for float16 vs float32."""
        img_fp32 = torch.rand(4, 3, 224, 224, device='cuda', dtype=torch.float32)
        img_fp16 = img_fp32.to(torch.float16)
        
        # Apply fused transform (uses FAST contrast)
        _, _, h, w = img_fp32.shape
        result_fp32 = F.fused_augment(
            img_fp32,
            top=0, left=0, height=h, width=w, flip_horizontal=False,  # No-op geometric
            brightness_factor=1.2,
            contrast_factor=1.1,
            saturation_factor=0.9,
            grayscale=False,
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225)
        )
        
        result_fp16 = F.fused_augment(
            img_fp16,
            top=0, left=0, height=h, width=w, flip_horizontal=False,  # No-op geometric
            brightness_factor=1.2,
            contrast_factor=1.1,
            saturation_factor=0.9,
            grayscale=False,
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225)
        )
        
        # Fused operation with normalize has same float16 precision issues
        # due to division by small std values
        torch.testing.assert_close(
            result_fp16.float(), 
            result_fp32, 
            rtol=0.01,   # 1% relative tolerance (relaxed from 0.1%)
            atol=0.01    # 0.01 absolute tolerance (relaxed from 0.001)
        )


class TestFloat16Correctness:
    """Test float16 correctness against torchvision float16."""
    
    @pytest.mark.parametrize("brightness_factor", [0.0, 0.5, 1.0, 1.5, 2.0])
    def test_float16_matches_torchvision_brightness(self, brightness_factor):
        """Test that float16 brightness matches torchvision float16."""
        img = torch.rand(2, 3, 224, 224, device='cuda', dtype=torch.float16)
        
        # Apply both
        tv_result = tvF.adjust_brightness(img, brightness_factor)
        ta_result = F.adjust_brightness(img, brightness_factor)
        
        # Should match within float16 precision
        torch.testing.assert_close(ta_result, tv_result, rtol=1e-3, atol=1e-3)
    
    @pytest.mark.parametrize("saturation_factor", [0.0, 0.5, 1.0, 1.5, 2.0])
    def test_float16_matches_torchvision_saturation(self, saturation_factor):
        """Test that float16 saturation matches torchvision float16."""
        img = torch.rand(2, 3, 224, 224, device='cuda', dtype=torch.float16)
        
        # Apply both
        tv_result = tvF.adjust_saturation(img, saturation_factor)
        ta_result = F.adjust_saturation(img, saturation_factor)
        
        # Should match within float16 precision
        torch.testing.assert_close(ta_result, tv_result, rtol=1e-3, atol=1e-3)
    
    @pytest.mark.parametrize("contrast_factor", [0.0, 0.5, 1.0, 1.5, 2.0])
    def test_float16_matches_torchvision_contrast(self, contrast_factor):
        """Test that float16 contrast matches torchvision float16."""
        img = torch.rand(2, 3, 224, 224, device='cuda', dtype=torch.float16)
        
        # Apply both
        tv_result = tvF.adjust_contrast(img, contrast_factor)
        ta_result = F.adjust_contrast(img, contrast_factor)
        
        # Should match within float16 precision
        torch.testing.assert_close(ta_result, tv_result, rtol=1e-3, atol=1e-3)
    
    @pytest.mark.parametrize("mean,std", [
        ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),  # ImageNet
        ((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),              # Simple
        ((0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),              # Identity
    ])
    def test_float16_matches_torchvision_normalize(self, mean, std):
        """Test that float16 normalize matches torchvision float16."""
        img = torch.rand(2, 3, 224, 224, device='cuda', dtype=torch.float16)
        
        # Apply both
        tv_result = tvF.normalize(img, mean=list(mean), std=list(std))
        ta_result = F.normalize(img, mean=mean, std=std)
        
        # Should match within float16 precision
        torch.testing.assert_close(ta_result, tv_result, rtol=1e-3, atol=1e-3)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

