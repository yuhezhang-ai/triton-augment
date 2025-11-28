import torch
import torch.nn.functional as F

# 90-degree rotation parameters
height = 111
width = 100

# Create a simple test image with known values
# Use a gradient so we can see which pixel was sampled
img = torch.arange(height * width, dtype=torch.float32).reshape(1, 1, height, width).cuda()

# Matrix for 90-degree rotation
# [a, b, c, d, e, f] = [0, 1, 0, -1, 0, 0]
theta = torch.tensor([
    [0.0, 1.0, 0.0],
    [-1.0, 0.0, 0.0]
], dtype=torch.float32, device='cuda').unsqueeze(0)

# Generate grid using PyTorch's affine_grid
grid = F.affine_grid(theta, [1, 1, height, width], align_corners=False)

# Sample using grid_sample with nearest neighbor
result = F.grid_sample(img, grid, mode='nearest', align_corners=False, padding_mode='zeros')

print("=== PyTorch grid_sample behavior ===")
print(f"Image size: {height}x{width}")
print()

# Check specific pixels that were mismatching
test_pixels = [
    (5, 2),
    (5, 4),
    (50, 55),
]

for x_out, y_out in test_pixels:
    # What did PyTorch sample?
    sampled_value = result[0, 0, y_out, x_out].item()
    
    # What's the grid coordinate?
    grid_x = grid[0, y_out, x_out, 0].item()
    grid_y = grid[0, y_out, x_out, 1].item()
    
    # Convert normalized grid coords to pixel coords
    # grid_sample formula: pixel = ((grid + 1) * size - 1) / 2
    pixel_x = ((grid_x + 1.0) * width - 1.0) * 0.5
    pixel_y = ((grid_y + 1.0) * height - 1.0) * 0.5
    
    print(f"Output pixel ({x_out}, {y_out}):")
    print(f"  Grid coords: ({grid_x:.10f}, {grid_y:.10f})")
    print(f"  Pixel coords: ({pixel_x:.10f}, {pixel_y:.10f})")
    print(f"  Sampled value: {sampled_value:.1f}")
    
    # Which input pixel does this correspond to?
    # The sampled value is the linear index in the original image
    if sampled_value > 0:
        input_y = int(sampled_value) // width
        input_x = int(sampled_value) % width
        print(f"  Input pixel: ({input_x}, {input_y})")
    else:
        print(f"  Input pixel: OUT OF BOUNDS")
    
    # What would round() give us?
    round_x = round(pixel_x)
    round_y = round(pixel_y)
    print(f"  round(): ({round_x}, {round_y})")
    
    # Check if they match
    if sampled_value > 0:
        if round_x == input_x and round_y == input_y:
            print(f"  ✓ round() matches PyTorch")
        else:
            print(f"  ✗ round() MISMATCH! Expected ({input_x}, {input_y}), got ({round_x}, {round_y})")
    
    print()

# Also check the bounds - what happens at negative coordinates?
print("=== Checking negative coordinate handling ===")
for x_out, y_out in [(5, 2), (0, 0)]:
    grid_x = grid[0, y_out, x_out, 0].item()
    grid_y = grid[0, y_out, x_out, 1].item()
    pixel_x = ((grid_x + 1.0) * width - 1.0) * 0.5
    pixel_y = ((grid_y + 1.0) * height - 1.0) * 0.5
    
    sampled_value = result[0, 0, y_out, x_out].item()
    
    print(f"Output ({x_out}, {y_out}): pixel_coords=({pixel_x:.2f}, {pixel_y:.2f}), sampled={sampled_value:.1f}")
    
    # Is it out of bounds?
    if pixel_x < 0 or pixel_x >= width or pixel_y < 0 or pixel_y >= height:
        print(f"  Coordinate is OUT OF BOUNDS")
        if sampled_value == 0:
            print(f"  ✓ Correctly returns 0 (padding)")
        else:
            print(f"  ✗ Should be 0 but got {sampled_value}")
    print()
