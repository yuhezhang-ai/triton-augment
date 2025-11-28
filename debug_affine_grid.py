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

# Now let's understand what torchvision does internally
# Step 1: Get the inverse affine matrix (in pixel coordinates)
import math
center = [width * 0.5, height * 0.5]  # Torchvision uses center of image

# Torchvision's _get_inverse_affine_matrix (simplified for our case)
angle_rad = math.radians(angle)
rot = [
    [math.cos(angle_rad), math.sin(angle_rad)],
    [-math.sin(angle_rad), math.cos(angle_rad)]
]
# For 90 degrees: cos(90)=0, sin(90)=1
# rot = [[0, 1], [-1, 0]]

# Translation to center, then rotate, then translate back
# The matrix maps OUTPUT coords to INPUT coords (inverse mapping)
print("=== Understanding Torchvision's Affine ===\n")
print(f"Image size: {width}x{height}")
print(f"Center: {center}")
print(f"Rotation angle: {angle} degrees")
print()

# Let's trace what happens to a specific output pixel
# For pixel (5, 2) in output, where does it sample from in input?
x_out, y_out = 5, 2

print(f"=== Tracing output pixel ({x_out}, {y_out}) ===\n")

# Torchvision's internal process:
# 1. Create base_grid with centered coordinates
#    x_grid: from (1-ow)*0.5 to (ow-1)*0.5 → from -49.5 to 49.5 for width=100
#    y_grid: from (1-oh)*0.5 to (oh-1)*0.5 → from -55 to 55 for height=111
x_centered = (1 - width) * 0.5 + x_out  # -49.5 + 5 = -44.5
y_centered = (1 - height) * 0.5 + y_out  # -55 + 2 = -53

print(f"Step 1: Centered coordinates")
print(f"  x_centered = (1 - {width}) * 0.5 + {x_out} = {x_centered}")
print(f"  y_centered = (1 - {height}) * 0.5 + {y_out} = {y_centered}")
print()

# 2. Apply rescaled theta
#    rescaled_theta = theta.T / [0.5*w, 0.5*h]
#    For 90-degree rotation: theta = [[0, 1, 0], [-1, 0, 0]]
#    theta.T = [[0, -1], [1, 0], [0, 0]]
#    rescaled = [[0, -1/(0.5*111)], [1/(0.5*100), 0], [0, 0]]
#             = [[0, -0.018], [0.02, 0], [0, 0]]

half_w = width * 0.5
half_h = height * 0.5

# Matrix elements (for 90-degree rotation)
a, b, c = 0.0, 1.0, 0.0
d, e, f = -1.0, 0.0, 0.0

# output_grid = [x_centered, y_centered, 1] @ rescaled_theta
# x_norm = (a*x + b*y + c) / half_w = (0*(-44.5) + 1*(-53) + 0) / 50 = -53/50 = -1.06
# y_norm = (d*x + e*y + f) / half_h = (-1*(-44.5) + 0*(-53) + 0) / 55.5 = 44.5/55.5 = 0.8018

x_norm = (a * x_centered + b * y_centered + c) / half_w
y_norm = (d * x_centered + e * y_centered + f) / half_h

print(f"Step 2: Apply rescaled theta")
print(f"  half_w = {half_w}, half_h = {half_h}")
print(f"  x_norm = ({a}*{x_centered} + {b}*{y_centered} + {c}) / {half_w} = {x_norm}")
print(f"  y_norm = ({d}*{x_centered} + {e}*{y_centered} + {f}) / {half_h} = {y_norm}")
print()

# 3. Convert normalized coords to input pixel coords
#    grid_sample with align_corners=False:
#    pixel = ((normalized + 1) * size - 1) / 2
x_in = ((x_norm + 1.0) * width - 1.0) * 0.5
y_in = ((y_norm + 1.0) * height - 1.0) * 0.5

print(f"Step 3: Convert to input pixel coords (for grid_sample)")
print(f"  x_in = (({x_norm} + 1) * {width} - 1) * 0.5 = {x_in}")
print(f"  y_in = (({y_norm} + 1) * {height} - 1) * 0.5 = {y_in}")
print()

# For nearest neighbor, round to nearest integer
x_nearest = round(x_in)
y_nearest = round(y_in)

print(f"Step 4: Nearest neighbor sampling")
print(f"  x_nearest = round({x_in}) = {x_nearest}")
print(f"  y_nearest = round({y_in}) = {y_nearest}")
print()

# Now let's verify by checking what torchvision produced
# Get the pixel value at output (5, 2) from torchvision's result
tv_pixel = tv_result[0, :, y_out, x_out]
print(f"=== Verification ===")
print(f"  Torchvision output at ({x_out}, {y_out}): {tv_pixel.tolist()}")

# If our calculation is correct, this should match input at (x_nearest, y_nearest)
if 0 <= x_nearest < width and 0 <= y_nearest < height:
    input_pixel = img[0, :, y_nearest, x_nearest]
    print(f"  Input at ({x_nearest}, {y_nearest}): {input_pixel.tolist()}")
    if torch.allclose(tv_pixel, input_pixel, atol=1e-5):
        print("  ✓ MATCH!")
    else:
        print("  ✗ MISMATCH!")
else:
    print(f"  ({x_nearest}, {y_nearest}) is out of bounds, should be fill value (0)")
    if torch.allclose(tv_pixel, torch.zeros_like(tv_pixel)):
        print("  ✓ Correctly filled with zeros!")
    else:
        print("  ✗ Expected zeros but got different values!")

if device == 'cuda':
    print("\n" + "="*60)
    print("Now testing our kernel...")
    print("="*60 + "\n")

    # Test our implementation
    import triton_augment.functional as F

    ta_result = F.affine(img, angle=angle, translate=translate, scale=scale, shear=shear,
                          interpolation='nearest')

    ta_pixel = ta_result[0, :, y_out, x_out]
    print(f"  Triton output at ({x_out}, {y_out}): {ta_pixel.tolist()}")
    print(f"  Torchvision at ({x_out}, {y_out}): {tv_pixel.tolist()}")

    if torch.allclose(ta_pixel, tv_pixel, atol=1e-5):
        print("  ✓ Triton matches Torchvision!")
    else:
        print("  ✗ Triton differs from Torchvision!")
        print(f"  Difference: {(ta_pixel - tv_pixel).abs().tolist()}")
else:
    print("\n(Skipping Triton kernel test - CUDA not available)")
