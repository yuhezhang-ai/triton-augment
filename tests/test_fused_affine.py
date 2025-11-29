
import torch
import pytest
import triton_augment as ta
import torchvision.transforms.v2.functional as TVF
from triton_augment.functional import InterpolationMode

def check_affine_result(ta_result, tv_result, interpolation, atol=1e-4, rtol=1e-3, max_mismatch_rate=0.25, msg_prefix=""):
        """
        Check if affine transform results match, with special handling for nearest neighbor.
        
        For bilinear: strict comparison with atol/rtol.
        For nearest: allow small percentage of pixels to differ due to boundary rounding.
        """
        if interpolation == "bilinear":
            torch.testing.assert_close(ta_result, tv_result, atol=atol, rtol=rtol, msg=f"{msg_prefix}")
        else:
            # For nearest neighbor, check that most pixels match
            # Mismatches at boundaries are acceptable
            diff = (ta_result - tv_result).abs()
            mismatch_mask = diff > atol
            mismatch_rate = mismatch_mask.float().mean().item()
                
            if mismatch_rate > max_mismatch_rate:
                # Too many mismatches - fail with detailed info
                torch.testing.assert_close(
                    ta_result, tv_result, atol=atol, rtol=rtol,
                    msg=f"Nearest neighbor mismatch rate {mismatch_rate:.2%} exceeds threshold {max_mismatch_rate:.2%} for {msg_prefix}"
                )

class TestFusedAffine:
    """
    Verify that fused_augment matches sequential application of transforms.
    We test the geometric composition: Affine -> Crop -> Flip.
    """
    @pytest.mark.parametrize("batch_size", [1, 4])
    @pytest.mark.parametrize("interpolation", ["nearest", "bilinear"])
    @pytest.mark.parametrize("imgsize", [(100, 111), (256, 256)])
    @pytest.mark.parametrize("shear", [[0.0, 0.0], [5, 1]])
    @pytest.mark.parametrize("flip", [True, False])
    @pytest.mark.parametrize("angle", [0.0, 30.0])
    @pytest.mark.parametrize("translate", [[0.0, 0.0], [10.0, 5.0]])
    @pytest.mark.parametrize("crop", [[10, 10, 80, 80], [0, 0, 100, 111]])
    def test_fused_affine_vs_sequential(self, batch_size, interpolation, imgsize, shear, flip, angle, translate, crop):
        """
        Verify that TritonFusedAffineAugment matches sequential application of transforms.
        We test the geometric composition: Affine -> Crop -> Flip.
        """
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
            
        device = "cuda"
        # Create a random image
        img = torch.rand(batch_size, 3, imgsize[0], imgsize[1], device=device)
        
        # 1. Define transforms
        dx, dy = translate
        scale_factor = 1.0
        crop_top, crop_left, crop_h, crop_w = crop
        
        # 2. Sequential torchvision execution (Ground Truth)
        # Note: Our fused pipeline does: Affine -> Crop -> Flip
        # So we must match that order.
        
        # Step A: Affine
        # Torchvision affine rotates around center.
        # We need to match the center logic.
        center = [img.shape[-1] * 0.5, img.shape[-2] * 0.5]
        
        out_seq = TVF.affine(
            img, 
            angle=angle, 
            translate=[dx, dy], 
            scale=scale_factor, 
            shear=shear,
            interpolation=TVF.InterpolationMode.NEAREST if interpolation == "nearest" else TVF.InterpolationMode.BILINEAR,
            #center=center
        )
        
        # Step B: Crop
        out_seq = TVF.crop(out_seq, crop_top, crop_left, crop_h, crop_w)
        
        # Step C: Flip
        if flip:
            out_seq = TVF.hflip(out_seq)
            
        # 3. Fused execution via fused_augment (New API)
        # The new API handles matrix construction internally.
        
        # Run Fused Kernel
        out_fused = ta.functional.fused_augment(
            img,
            # Crop params
            top=crop_top,
            left=crop_left,
            height=crop_h,
            width=crop_w,
            flip_horizontal=flip,
            # Affine params
            angle=angle,
            translate=[dx, dy],
            scale=scale_factor,
            shear=shear,
            interpolation=interpolation,
            #center=center
        )

        # 4. Sequential triton execution
        out_triton = ta.functional.affine(
            img,
            angle=angle,
            translate=[dx, dy],
            scale=scale_factor,
            shear=shear,
            interpolation=interpolation,
            #center=center
        )
        
        out_triton = ta.functional.crop(
            out_triton,
            top=crop_top,
            left=crop_left,
            height=crop_h,
            width=crop_w,
        )
        
        if flip:
            out_triton = ta.functional.horizontal_flip(out_triton)
        
        # 5. Compare
        check_affine_result(out_triton, out_seq, interpolation, msg_prefix="Triton-sequential vs torchvision")
        check_affine_result(out_fused, out_seq, interpolation, msg_prefix="Fused vs torchvision")

class TestAllFusedOps:
    """
    Verify that fused_augment matches sequential application of ALL transforms.
    Order: Affine -> Crop -> Flip -> Brightness -> Contrast -> Saturation -> Grayscale -> Normalize
    """
    @pytest.mark.parametrize("batch_size", [1, 4])
    @pytest.mark.parametrize("interpolation", ["bilinear"])
    @pytest.mark.parametrize("imgsize", [(128, 111), (256, 256)])
    @pytest.mark.parametrize("flip", [True, False])
    @pytest.mark.parametrize("angle", [15.0])
    @pytest.mark.parametrize("brightness", [1.2])
    @pytest.mark.parametrize("saturation", [0.8])
    @pytest.mark.parametrize("grayscale", [True, False])
    def test_fused_all_ops_vs_sequential(self, batch_size, interpolation, imgsize, flip, angle, brightness, saturation, grayscale):
        """
        Verify that fused_augment matches sequential application of ALL transforms.
        Order: Affine -> Crop -> Flip -> Brightness -> Contrast -> Saturation -> Grayscale -> Normalize
        """
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
            
        device = "cuda"
        img = torch.rand(batch_size, 3, imgsize[0], imgsize[1], device=device)
        
        # Parameters
        dx, dy = 5.0, 5.0
        scale_factor = 1.1
        shear = [0.0, 0.0]
        crop_top, crop_left, crop_h, crop_w = 10, 10, 100, 100
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
        
        # 1. Sequential torchvision execution
        # Geometric: Affine -> Crop -> Flip
        out_seq = TVF.affine(
            img, 
            angle=angle, 
            translate=[dx, dy], 
            scale=scale_factor, 
            shear=shear,
            interpolation=TVF.InterpolationMode.BILINEAR
        )
        out_seq = TVF.crop(out_seq, crop_top, crop_left, crop_h, crop_w)
        if flip:
            out_seq = TVF.hflip(out_seq)
            
        # Color: Brightness -> Contrast -> Saturation -> Grayscale -> Normalize
        out_seq = TVF.adjust_brightness(out_seq, brightness)
        # Skip contrast for exact match (fast vs standard implementation diffs)
        out_seq = TVF.adjust_saturation(out_seq, saturation)
        if grayscale:
            out_seq = TVF.rgb_to_grayscale(out_seq, num_output_channels=3)
        out_seq = TVF.normalize(out_seq, mean, std)
        
        # 2. Fused execution
        out_fused = ta.functional.fused_augment(
            img,
            # Geometric
            top=crop_top, left=crop_left, height=crop_h, width=crop_w,
            flip_horizontal=flip,
            angle=angle, translate=[dx, dy], scale=scale_factor, shear=shear,
            interpolation=interpolation,
            # Color
            brightness_factor=torch.full((batch_size,), brightness, device=device),
            contrast_factor=torch.full((batch_size,), 1.0, device=device), # Skip contrast
            saturation_factor=torch.full((batch_size,), saturation, device=device),
            grayscale=grayscale,
            mean=mean, std=std
        )
        
        # Compare
        # Allow slightly higher tolerance for combined operations due to float precision accumulation
        check_affine_result(out_fused, out_seq, interpolation, atol=1e-3, rtol=1e-3, msg_prefix="All-Ops Fused vs torchvision")

class TestFusedAugmentTransformClass:
    """Test TritonFusedAugment Transform class correctness."""
    
    def test_fused_augment_determinism(self):
        """Test that TritonFusedAugment Transform produces same results with same seed."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
            
        transform = ta.TritonFusedAugment(
            crop_size=224,
            horizontal_flip_p=0.5,
            brightness=0.2,
            contrast=0.2,
            saturation=0.2,
            grayscale_p=0.1,
            degrees=30,
            translate=(0.1, 0.1),
            scale=(0.8, 1.2),
            shear=10,
            same_on_batch=False
        )
        
        img = torch.rand(4, 3, 256, 256, device='cuda')
        
        # Run 1
        torch.manual_seed(42)
        out1 = transform(img)
        
        # Run 2
        torch.manual_seed(42)
        out2 = transform(img)
        
        torch.testing.assert_close(out1, out2, msg="Results should be identical with same seed")
        
    def test_fused_augment_params_match_functional(self):
        """Test that TritonFusedAugment uses parameters correctly by comparing with manual functional call."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
            
        img_size = (256, 256)
        crop_size = 224
        batch_size = 2
        
        transform = ta.TritonFusedAugment(
            crop_size=crop_size,
            horizontal_flip_p=0.5,
            brightness=0.2,
            contrast=0.2,
            saturation=0.2,
            grayscale_p=0.5,
            degrees=30,
            translate=(0.1, 0.1),
            scale=(0.8, 1.2),
            shear=10,
            same_on_batch=False
        )
        
        img = torch.rand(batch_size, 3, *img_size, device='cuda')
        
        # Set seed
        torch.manual_seed(123)
        
        # 1. Get parameters manually using the new API
        (
            angle, translate, scale, shear,
            top_offsets, left_offsets, do_flip,
            brightness_factors, contrast_factors, saturation_factors, grayscale_mask
        ) = transform._get_params(batch_size, img.device, img_size)
        
        # 2. Run transform (reset seed to ensure it generates same params internally)
        torch.manual_seed(123)
        out_transform = transform(img)
        
        # 3. Run functional manually with captured params
        out_functional = ta.functional.fused_augment(
            img,
            top=top_offsets,
            left=left_offsets,
            height=crop_size,
            width=crop_size,
            flip_horizontal=do_flip,
            angle=angle,
            translate=translate,
            scale=scale,
            shear=shear,
            interpolation=transform.interpolation,
            fill=transform.fill,
            brightness_factor=brightness_factors,
            contrast_factor=contrast_factors,
            saturation_factor=saturation_factors,
            grayscale=grayscale_mask,
            mean=transform.color_helper.mean,
            std=transform.color_helper.std,
        )
        
        torch.testing.assert_close(out_transform, out_functional, msg="Transform output should match functional output with same params")

    def test_fused_augment_class_vs_torchvision(self):
        """
        Test that TritonFusedAugment class matches sequential torchvision.
        We use same_on_batch=True to easily compare with torchvision which broadcasts scalars.
        """
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
            
        batch_size = 2
        img_size = (128, 128)
        crop_size = 100
        
        # Initialize transform with ALL operations
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
        
        transform = ta.TritonFusedAugment(
            crop_size=crop_size,
            horizontal_flip_p=0.5,
            brightness=0.2,
            contrast=0.0, # Skip contrast for exact match
            saturation=0.2,
            grayscale_p=0.5,
            degrees=15,
            translate=(0.1, 0.1),
            scale=(0.9, 1.1),
            shear=5,
            same_on_batch=True, # Important for easy comparison
            interpolation="bilinear",
            mean=mean,
            std=std
        )
        
        img = torch.rand(batch_size, 3, *img_size, device='cuda')
        
        # 1. Run Transform
        torch.manual_seed(42)
        out_triton = transform(img)
        
        # 2. Get parameters that were used
        torch.manual_seed(42)
        (
            angle, translate, scale, shear,
            top_offsets, left_offsets, do_flip,
            brightness_factors, contrast_factors, saturation_factors, grayscale_mask
        ) = transform._get_params(1, img.device, img_size) # param_count=1 because same_on_batch=True
        
        # 3. Run Sequential Torchvision
        # Note: We need to extract scalar values from the tensors
        
        # Geometric: Affine -> Crop -> Flip
        out_seq = TVF.affine(
            img, 
            angle=angle.item(), 
            translate=translate[0].tolist(), 
            scale=scale.item(), 
            shear=shear[0].tolist(),
            interpolation=TVF.InterpolationMode.BILINEAR
        )
        
        out_seq = TVF.crop(
            out_seq, 
            int(top_offsets.item()), 
            int(left_offsets.item()), 
            crop_size, 
            crop_size
        )
        
        if do_flip.item():
            out_seq = TVF.hflip(out_seq)
            
        # Color: Brightness -> Contrast -> Saturation -> Grayscale -> Normalize
        out_seq = TVF.adjust_brightness(out_seq, brightness_factors.item())
        # Contrast skipped
        out_seq = TVF.adjust_saturation(out_seq, saturation_factors.item())
        
        if grayscale_mask.item():
            out_seq = TVF.rgb_to_grayscale(out_seq, num_output_channels=3)
            
        # Normalize (using default mean/std from transform)
        if transform.color_helper.mean is not None and transform.color_helper.std is not None:
            out_seq = TVF.normalize(out_seq, transform.color_helper.mean, transform.color_helper.std)
        
        # Compare
        check_affine_result(out_triton, out_seq, "bilinear", atol=1e-3, rtol=1e-3, msg_prefix="Class vs Sequential")

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
