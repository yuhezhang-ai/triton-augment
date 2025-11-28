import torch
import torch.nn.functional as F

# 90-degree rotation parameters
height = 111
width = 100

# Test pixel that's mismatching
x_out, y_out = 5, 2

print("=== Comparing Our Kernel Logic vs PyTorch's Grid ===")
print(f"Image size: {height}x{width}")
print(f"Output pixel: ({x_out}, {y_out})")
print()

# === 1. PyTorch's actual grid ===
theta = torch.tensor([
    [0.0, 1.0, 0.0],
    [-1.0, 0.0, 0.0]
], dtype=torch.float32, device='cuda').unsqueeze(0)

grid = F.affine_grid(theta, [1, 1, height, width], align_corners=False)

grid_x = grid[0, y_out, x_out, 0].item()
grid_y = grid[0, y_out, x_out, 1].item()

# Convert to pixel coords
pixel_x_torch = ((grid_x + 1.0) * width - 1.0) * 0.5
pixel_y_torch = ((grid_y + 1.0) * height - 1.0) * 0.5

print("PyTorch's affine_grid:")
print(f"  Grid coords: ({grid_x:.10f}, {grid_y:.10f})")
print(f"  Pixel coords: ({pixel_x_torch:.10f}, {pixel_y_torch:.10f})")
print()

# === 2. Our kernel's logic (from geometric_kernel.py) ===
# Matrix for 90-degree rotation
a, b, c_tx = 0.0, 1.0, 0.0
d, e, f_ty = -1.0, 0.0, 0.0

half_w = width * 0.5
half_h = height * 0.5

# Step 1: Convert to centered coordinates
x_centered = x_out - half_w + 0.5
y_centered = y_out - half_h + 0.5

# Step 2: Apply matrix with rescaling
x_norm = (a * x_centered + b * y_centered + c_tx) / half_w
y_norm = (d * x_centered + e * y_centered + f_ty) / half_h

# Step 3: Convert to input pixel coords
x_in = ((x_norm + 1.0) * width - 1.0) * 0.5
y_in = ((y_norm + 1.0) * height - 1.0) * 0.5

print("Our kernel's calculation:")
print(f"  Centered: ({x_centered:.6f}, {y_centered:.6f})")
print(f"  Normalized: ({x_norm:.10f}, {y_norm:.10f})")
print(f"  Pixel coords: ({x_in:.10f}, {y_in:.10f})")
print()

# === 3. Compare ===
print("Comparison:")
print(f"  PyTorch pixel coords: ({pixel_x_torch:.6f}, {pixel_y_torch:.6f})")
print(f"  Our kernel pixel coords: ({x_in:.6f}, {y_in:.6f})")
print(f"  Difference: ({abs(pixel_x_torch - x_in):.6f}, {abs(pixel_y_torch - y_in):.6f})")

if abs(pixel_x_torch - x_in) > 0.01 or abs(pixel_y_torch - y_in) > 0.01:
    print("  ✗ MISMATCH! Our coordinate calculation is wrong!")
else:
    print("  ✓ Match! Coordinates are correct.")
print()

# === 4. What does grid_sample actually sample? ===
img = torch.arange(height * width, dtype=torch.float32).reshape(1, 1, height, width).cuda()
result = F.grid_sample(img, grid, mode='nearest', align_corners=False, padding_mode='zeros')
sampled_value = result[0, 0, y_out, x_out].item()

if sampled_value > 0:
    input_y = int(sampled_value) // width
    input_x = int(sampled_value) % width
    print(f"PyTorch grid_sample sampled:")
    print(f"  Value: {sampled_value:.1f}")
    print(f"  Input pixel: ({input_x}, {input_y})")
    print()
    
    # What would our RNE give?
    import math
    def round_rne(val, epsilon=1e-4):
        val_floor = math.floor(val)
        val_frac = val - val_floor
        is_half = abs(val_frac - 0.5) < epsilon
        
        if is_half:
            val_floor_int = int(val_floor)
            is_odd = (val_floor_int % 2) != 0
            return val_floor_int + (1 if is_odd else 0)
        else:
            return int(math.floor(val + 0.5))
    
    x_rne = round_rne(x_in)
    y_rne = round_rne(y_in)
    
    print(f"Our RNE would give: ({x_rne}, {y_rne})")
    
    if x_rne == input_x and y_rne == input_y:
        print("  ✓ Our RNE matches PyTorch!")
    else:
        print(f"  ✗ Our RNE doesn't match! Expected ({input_x}, {input_y}), got ({x_rne}, {y_rne})")
