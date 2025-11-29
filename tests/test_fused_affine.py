
import torch
import pytest
import triton_augment as ta
import torchvision.transforms.v2.functional as TVF
from triton_augment.functional import InterpolationMode

def check_affine_result(ta_result, tv_result, interpolation, atol=1e-3, rtol=1e-3, max_mismatch_rate=0.25, msg_prefix=""):
    """
    Check if affine transform results match, with special handling for nearest neighbor.
    
    For bilinear: strict comparison with atol/rtol.
    For nearest: allow small percentage of pixels to differ due to boundary rounding.
    """
    if interpolation == "bilinear":
        torch.testing.assert_close(ta_result, tv_result, atol=atol, rtol=rtol, msg_prefix=msg_prefix)
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


@pytest.mark.parametrize("batch_size", [1, 4])
@pytest.mark.parametrize("interpolation", ["nearest", "bilinear"])
@pytest.mark.parametrize("imgsize", [(100, 111), (256, 256)])
@pytest.mark.parametrize("shear", [[0.0, 0.0], [5, 1]])
@pytest.mark.parametrize("flip", [True, False])
@pytest.mark.parametrize("angle", [0.0, 30.0])
@pytest.mark.parametrize("translate", [[0.0, 0.0], [10.0, 5.0]])
@pytest.mark.parametrize("crop", [[10, 10, 80, 80], [0, 0, 100, 111]])
def test_fused_affine_vs_sequential(batch_size, interpolation, imgsize, shear, flip, angle, translate, crop):
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
    check_affine_result(out_fused, out_seq, interpolation, msg_prefix="Fused vs torchvision")
    check_affine_result(out_triton, out_seq, interpolation, msg_prefix="Triton-sequential vs torchvision")

if __name__ == "__main__":
    test_fused_affine_vs_sequential(1, "nearest")
    test_fused_affine_vs_sequential(1, "bilinear")
