import torch
import torchvision.transforms.v2.functional as tvF

# Test with 90-degree rotation, 111x100 (non-square to catch dimension issues)
height, width = 111, 100
batch_size = 1

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Create a gradient tensor where each pixel value encodes its (x, y) position
# pixel[y, x] = y * 1000 + x
# This way we can decode which pixel was sampled by looking at the value
img = torch.zeros(batch_size, 1, height, width, device=device)
for y in range(height):
    for x in range(width):
        img[0, 0, y, x] = y * 1000 + x

print(f"=== Gradient Tensor Test ===")
print(f"Image size: {width}x{height}")
print(f"Each pixel value = y * 1000 + x")
print(f"Example: pixel[50, 30] = {img[0, 0, 50, 30].item()}")
print()

# Parameters for 90-degree rotation
angle = 90.0
translate = [0.0, 0.0]
scale = 1.0
shear = [0.0, 0.0]

# Get torchvision's result
tv_result = tvF.affine(img, angle=angle, translate=translate, scale=scale, shear=shear,
                        interpolation=tvF.InterpolationMode.NEAREST)

print(f"Rotation: {angle} degrees")
print()

def decode_pixel(value):
    """Decode pixel value to (x, y) coordinates."""
    if value == 0:
        return None  # Out of bounds (fill value)
    y = int(value) // 1000
    x = int(value) % 1000
    return (x, y)

if device == 'cuda':
    import triton_augment.functional as F
    
    ta_result = F.affine(img, angle=angle, translate=translate, scale=scale, shear=shear,
                          interpolation='nearest')
    
    # Find mismatches
    diff = (ta_result - tv_result).abs()
    mismatch_mask = diff > 0.5  # Use 0.5 since values are integers
    mismatch_count = mismatch_mask.sum().item()
    total_pixels = ta_result.numel()
    
    print(f"Mismatched pixels: {mismatch_count}/{total_pixels} ({100*mismatch_count/total_pixels:.2f}%)")
    print()
    
    if mismatch_count > 0:
        print("=== First 10 Mismatches ===")
        print(f"{'Output (x,y)':<15} {'TV sampled from':<20} {'Triton sampled from':<20}")
        print("-" * 55)
        
        indices = torch.where(mismatch_mask)
        for i in range(min(10, len(indices[0]))):
            n, c, y_out, x_out = indices[0][i].item(), indices[1][i].item(), indices[2][i].item(), indices[3][i].item()
            
            tv_val = tv_result[n, c, y_out, x_out].item()
            ta_val = ta_result[n, c, y_out, x_out].item()
            
            tv_src = decode_pixel(tv_val)
            ta_src = decode_pixel(ta_val)
            
            tv_str = f"({tv_src[0]}, {tv_src[1]})" if tv_src else "OUT OF BOUNDS"
            ta_str = f"({ta_src[0]}, {ta_src[1]})" if ta_src else "OUT OF BOUNDS"
            
            print(f"({x_out:3}, {y_out:3})      {tv_str:<20} {ta_str:<20}")
        
        print()
        
        # Get the actual matrix that triton uses
        from triton_augment.functional import _get_inverse_affine_matrix
        center_t = torch.tensor([[0.0, 0.0]], device=device)
        angle_t = torch.tensor([angle], device=device)
        translate_t = torch.tensor([translate], device=device)
        scale_t = torch.tensor([scale], device=device)
        shear_t = torch.tensor([shear], device=device)
        matrix = _get_inverse_affine_matrix(center_t, angle_t, translate_t, scale_t, shear_t)
        a, b, c_tx, d, e, f_ty = matrix[0].tolist()
        print(f"Triton matrix: [{a}, {b}, {c_tx}, {d}, {e}, {f_ty}]")
        
        print("=== Detailed Analysis of First 5 Mismatches ===")
        
        cx, cy = width / 2, height / 2  # 50, 55.5
        
        for idx in range(min(5, len(indices[0]))):
            n, c, y_out, x_out = indices[0][idx].item(), indices[1][idx].item(), indices[2][idx].item(), indices[3][idx].item()
            
            tv_val = tv_result[n, c, y_out, x_out].item()
            ta_val = ta_result[n, c, y_out, x_out].item()
            
            tv_src = decode_pixel(tv_val)
            ta_src = decode_pixel(ta_val)
            
            print(f"\n--- Mismatch #{idx+1}: Output pixel ({x_out}, {y_out}) ---")
            print(f"Torchvision sampled from: {tv_src}")
            print(f"Triton sampled from: {ta_src}")
            
            # Trace through EXACTLY what the kernel does:
            # Step 1: Center the output coordinate (kernel lines 403-406)
            half_ow = width * 0.5
            half_oh = height * 0.5
            x_centered = x_out - half_ow + 0.5
            y_centered = y_out - half_oh + 0.5
            
            # Step 2: Apply matrix with rescaling (kernel lines 414-417)
            half_iw = width * 0.5
            half_ih = height * 0.5
            x_norm = (a * x_centered + b * y_centered + c_tx) / half_iw
            y_norm = (d * x_centered + e * y_centered + f_ty) / half_ih
            
            # Step 3: Convert normalized to pixel coords (kernel lines 422-423)
            x_in = ((x_norm + 1.0) * width - 1.0) * 0.5
            y_in = ((y_norm + 1.0) * height - 1.0) * 0.5
            
            print(f"Input coords: ({x_in:.10f}, {y_in:.10f})")
            print(f"  y_in fraction: {y_in - int(y_in):.10f}")
            print(f"  floor(x+0.5)={int(x_in + 0.5 if x_in >= 0 else x_in - 0.5)}, floor(y+0.5)={int(y_in + 0.5 if y_in >= 0 else y_in - 0.5)}")
            print(f"  Python round: ({round(x_in)}, {round(y_in)})")
            print(f"  int(y_in): {int(y_in)}, y_in is exactly .5? {abs(y_in - int(y_in) - 0.5) < 1e-9}")
        
    else:
        print("All pixels match!")
        
else:
    print("CUDA not available, skipping Triton test")
