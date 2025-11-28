import torch
import torch.nn.functional as F_torch
import torchvision.transforms.v2.functional as tvF
from torchvision.transforms.functional import _get_inverse_affine_matrix as tv_get_matrix

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

print("=== Comparing Matrix Computation ===\n")

# Torchvision's matrix (it uses center in pixel coords internally)
center_tv = [width * 0.5, height * 0.5]
tv_matrix = tv_get_matrix(center_tv, angle, translate, scale, shear)
print(f"Torchvision matrix (center={center_tv}):")
print(f"  {tv_matrix}")
print()

# Our matrix computation (we use translated coords where [0,0] = center)
if device == 'cuda':
    import triton_augment.functional as F
    from triton_augment.functional import _get_inverse_affine_matrix
    
    # Our center is in translated coords: [0, 0] for image center
    center_ours = torch.tensor([[0.0, 0.0]], device=device)
    angle_t = torch.tensor([angle], device=device)
    translate_t = torch.tensor([translate], device=device)
    scale_t = torch.tensor([scale], device=device)
    shear_t = torch.tensor([shear], device=device)
    
    our_matrix = _get_inverse_affine_matrix(center_ours, angle_t, translate_t, scale_t, shear_t)
    print(f"Our matrix (center=[0,0] in translated coords):")
    print(f"  {our_matrix[0].tolist()}")
    print()
    
    # Compare
    tv_tensor = torch.tensor(tv_matrix, device=device)
    diff = (our_matrix[0] - tv_tensor).abs()
    print(f"Matrix difference: {diff.tolist()}")
    print(f"Max diff: {diff.max().item()}")
    print()

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

# Matrix elements from torchvision (for 90-degree rotation around center)
# For 90 degrees: cos(90)=0, sin(90)=1
# The inverse rotation matrix is [[0, 1], [-1, 0]]
# With center at [50, 55.5], the matrix should be:
# a=0, b=1, c=55.5-50=5.5, d=-1, e=0, f=50+55.5=105.5
# Wait, let me check torchvision's exact formula...
a, b, c, d, e, f = tv_matrix
print(f"Matrix from torchvision: a={a}, b={b}, c={c}, d={d}, e={e}, f={f}")
print()

def trace_pixel(x_out, y_out, matrix=None):
    """Trace the coordinate transformation for a given output pixel."""
    if matrix is None:
        matrix = tv_matrix
    a, b, c, d, e, f = matrix
    
    print(f"\n=== Tracing output pixel ({x_out}, {y_out}) ===")
    
    # Step 1: Centered coordinates (as torchvision's base_grid does)
    x_centered = (1 - width) * 0.5 + x_out
    y_centered = (1 - height) * 0.5 + y_out
    print(f"  Centered: ({x_centered}, {y_centered})")
    
    # Step 2: Apply rescaled theta
    # torchvision: output_grid = base_grid @ (theta.T / [0.5*w, 0.5*h])
    x_norm = (a * x_centered + b * y_centered + c) / half_w
    y_norm = (d * x_centered + e * y_centered + f) / half_h
    print(f"  Normalized: ({x_norm:.6f}, {y_norm:.6f})")
    
    # Step 3: Convert to input pixel coords (grid_sample formula)
    x_in = ((x_norm + 1.0) * width - 1.0) * 0.5
    y_in = ((y_norm + 1.0) * height - 1.0) * 0.5
    print(f"  Input coords: ({x_in:.6f}, {y_in:.6f})")
    
    # Step 4: Nearest neighbor (using Python's round for comparison)
    x_nearest = round(x_in)
    y_nearest = round(y_in)
    print(f"  Nearest (Python round): ({x_nearest}, {y_nearest})")
    
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
        # Try to find which pixel TV actually sampled from
        for dy in range(-2, 3):
            for dx in range(-2, 3):
                nx, ny = x_nearest + dx, y_nearest + dy
                if 0 <= nx < width and 0 <= ny < height:
                    val = img[0, 0, ny, nx].item()
                    if abs(tv_pixel - val) < 1e-5:
                        print(f"  -> TV actually sampled from ({nx}, {ny})")
                        return False
        return False

# Test specific pixels
print("\n=== Testing pixels ===")
test_pixels = [
    (50, 55),   # Near center
    (2, 5),     # First mismatch from test
    (98, 97),   # Failing test coordinate
    (50, 60),   # Had mismatch earlier
]

for x, y in test_pixels:
    trace_pixel(x, y)

if device == 'cuda':
    print("\n" + "="*60)
    print("Comparing Triton vs Torchvision for all pixels...")
    print("="*60 + "\n")

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
else:
    print("\n(Skipping Triton kernel test - CUDA not available)")
