import torch
import torchvision.transforms.v2.functional as tvF

# Test with 90-degree rotation, 111x100
height, width = 111, 100
batch_size = 1

# Create test image - use CUDA if available
device = 'cuda' if torch.cuda.is_available() else 'cpu'
img = torch.rand(batch_size, 3, height, width, device=device)

# Parameters for 90-degree rotation
angle = 90.0
translate = [0.0, 0.0]
scale = 1.0
shear = [0.0, 0.0]

# Get torchvision's result
tv_result = tvF.affine(img, angle=angle, translate=translate, scale=scale, shear=shear,
                        interpolation=tvF.InterpolationMode.NEAREST)

print("=== Understanding Torchvision's Affine ===\n")
print(f"Image size: {width}x{height}")
print(f"Rotation angle: {angle} degrees")
print()

import math
half_w = width * 0.5
half_h = height * 0.5

# Matrix elements (for 90-degree rotation)
a, b, c = 0.0, 1.0, 0.0
d, e, f = -1.0, 0.0, 0.0

def trace_pixel(x_out, y_out):
    """Trace the coordinate transformation for a given output pixel."""
    print(f"\n=== Tracing output pixel ({x_out}, {y_out}) ===")
    
    # Step 1: Centered coordinates
    x_centered = (1 - width) * 0.5 + x_out
    y_centered = (1 - height) * 0.5 + y_out
    print(f"  Centered: ({x_centered}, {y_centered})")
    
    # Step 2: Apply rescaled theta
    x_norm = (a * x_centered + b * y_centered + c) / half_w
    y_norm = (d * x_centered + e * y_centered + f) / half_h
    print(f"  Normalized: ({x_norm:.6f}, {y_norm:.6f})")
    
    # Step 3: Convert to input pixel coords
    x_in = ((x_norm + 1.0) * width - 1.0) * 0.5
    y_in = ((y_norm + 1.0) * height - 1.0) * 0.5
    print(f"  Input coords: ({x_in:.6f}, {y_in:.6f})")
    
    # Step 4: Nearest neighbor
    x_nearest = round(x_in)
    y_nearest = round(y_in)
    print(f"  Nearest: ({x_nearest}, {y_nearest})")
    
    # Get torchvision result
    tv_pixel = tv_result[0, 0, y_out, x_out].item()
    
    # Expected value
    if 0 <= x_nearest < width and 0 <= y_nearest < height:
        expected = img[0, 0, y_nearest, x_nearest].item()
        in_bounds = True
    else:
        expected = 0.0
        in_bounds = False
    
    print(f"  TV result: {tv_pixel:.6f}, Expected: {expected:.6f}, In bounds: {in_bounds}")
    
    if abs(tv_pixel - expected) < 1e-5:
        print("  ✓ Match!")
        return True
    else:
        print(f"  ✗ Mismatch! Diff: {abs(tv_pixel - expected):.6f}")
        return False

# Test multiple pixels to find where the mismatch occurs
print("\n=== Testing multiple pixels ===")

# Center of image should be stable
center_x, center_y = width // 2, height // 2
trace_pixel(center_x, center_y)

# Try some other pixels
test_pixels = [
    (50, 55),   # Near center
    (50, 50),   # Slightly off center
    (60, 55),   # Right of center
    (40, 55),   # Left of center
    (50, 60),   # Below center
    (50, 45),   # Above center
    (30, 30),   # Upper left quadrant
    (70, 80),   # Lower right quadrant
]

mismatches = 0
for x, y in test_pixels:
    if not trace_pixel(x, y):
        mismatches += 1

print(f"\n\nTotal mismatches: {mismatches}/{len(test_pixels)}")

if device == 'cuda':
    print("\n" + "="*60)
    print("Comparing Triton vs Torchvision for all pixels...")
    print("="*60 + "\n")

    import triton_augment.functional as F

    ta_result = F.affine(img, angle=angle, translate=translate, scale=scale, shear=shear,
                          interpolation='nearest')

    diff = (ta_result - tv_result).abs()
    mismatch_mask = diff > 1e-3
    mismatch_count = mismatch_mask.sum().item()
    total_pixels = ta_result.numel()
    
    print(f"Mismatched pixels: {mismatch_count}/{total_pixels} ({100*mismatch_count/total_pixels:.2f}%)")
    
    if mismatch_count > 0:
        # Find first few mismatches
        indices = torch.where(mismatch_mask)
        print(f"\nFirst 5 mismatches:")
        for i in range(min(5, len(indices[0]))):
            n, c, y, x = indices[0][i].item(), indices[1][i].item(), indices[2][i].item(), indices[3][i].item()
            print(f"  (n={n}, c={c}, y={y}, x={x}): Triton={ta_result[n, c, y, x].item():.6f}, TV={tv_result[n, c, y, x].item():.6f}")
        
        # Trace first mismatch pixel in detail
        n, c, y, x = indices[0][0].item(), indices[1][0].item(), indices[2][0].item(), indices[3][0].item()
        print(f"\n=== Detailed trace of first mismatch (y={y}, x={x}) ===")
        trace_pixel(x, y)
        
        # Also trace the specific failing coordinate from test: (5, 0, 97, 98)
        print(f"\n=== Tracing the failing test coordinate (y=97, x=98) ===")
        trace_pixel(98, 97)
else:
    print("\n(Skipping Triton kernel test - CUDA not available)")
