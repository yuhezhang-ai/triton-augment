import torch
import torch.nn.functional as F

print("=== Testing grid_sample's nearest neighbor rounding ===\n")

# Create a simple 5x5 image with known values
img = torch.arange(25, dtype=torch.float32).reshape(1, 1, 5, 5).cuda()
print("Image (5x5):")
print(img[0, 0])
print()

# Test specific normalized coordinates that map to half-integers
# For a 5x5 image with align_corners=False:
# pixel = ((norm + 1) * 5 - 1) / 2
# So: norm = (2*pixel + 1) / 5 - 1

test_cases = [
    # (pixel_x, pixel_y, description)
    (0.5, 0.5, "exact half between pixels 0 and 1"),
    (1.5, 1.5, "exact half between pixels 1 and 2"),
    (2.5, 2.5, "exact half between pixels 2 and 3"),
    (0.0, 0.0, "exact pixel 0"),
    (1.0, 1.0, "exact pixel 1"),
    (0.4, 0.4, "slightly below half"),
    (0.6, 0.6, "slightly above half"),
]

for pixel_x, pixel_y, desc in test_cases:
    # Convert pixel coords to normalized coords
    norm_x = (2 * pixel_x + 1) / 5 - 1
    norm_y = (2 * pixel_y + 1) / 5 - 1
    
    # Create a grid with just this one point
    grid = torch.tensor([[[[norm_x, norm_y]]]], dtype=torch.float32, device='cuda')
    
    # Sample
    result = F.grid_sample(img, grid, mode='nearest', align_corners=False, padding_mode='zeros')
    sampled_value = result[0, 0, 0, 0].item()
    
    # Which pixel was sampled?
    if sampled_value >= 0:
        sampled_y = int(sampled_value) // 5
        sampled_x = int(sampled_value) % 5
    else:
        sampled_x, sampled_y = -1, -1
    
    # What would round() give?
    round_x = round(pixel_x)
    round_y = round(pixel_y)
    
    # What would floor(x + 0.5) give?
    import math
    floor_half_x = math.floor(pixel_x + 0.5)
    floor_half_y = math.floor(pixel_y + 0.5)
    
    print(f"Pixel coords: ({pixel_x:.1f}, {pixel_y:.1f}) - {desc}")
    print(f"  Normalized: ({norm_x:.6f}, {norm_y:.6f})")
    print(f"  grid_sample sampled: pixel ({sampled_x}, {sampled_y}), value={sampled_value:.0f}")
    print(f"  round() would give: ({round_x}, {round_y})")
    print(f"  floor(x+0.5) would give: ({floor_half_x}, {floor_half_y})")
    
    if sampled_x == round_x and sampled_y == round_y:
        print(f"  ✓ Matches round()")
    elif sampled_x == floor_half_x and sampled_y == floor_half_y:
        print(f"  ✓ Matches floor(x+0.5)")
    else:
        print(f"  ✗ Doesn't match either!")
    print()

# Now test with the actual 90-degree rotation case
print("\n=== 90-degree rotation case (111x100) ===\n")

height, width = 111, 100
theta = torch.tensor([
    [0.0, 1.0, 0.0],
    [-1.0, 0.0, 0.0]
], dtype=torch.float32, device='cuda').unsqueeze(0)

grid = F.affine_grid(theta, [1, 1, height, width], align_corners=False)

# Check a few pixels
test_pixels = [(5, 2), (5, 4), (50, 55)]

for x_out, y_out in test_pixels:
    grid_x = grid[0, y_out, x_out, 0].item()
    grid_y = grid[0, y_out, x_out, 1].item()
    
    # Convert to pixel coords
    pixel_x = ((grid_x + 1.0) * width - 1.0) * 0.5
    pixel_y = ((grid_y + 1.0) * height - 1.0) * 0.5
    
    print(f"Output ({x_out}, {y_out}):")
    print(f"  Pixel coords: ({pixel_x:.10f}, {pixel_y:.10f})")
    print(f"  Fraction: ({pixel_x - math.floor(pixel_x):.10f}, {pixel_y - math.floor(pixel_y):.10f})")
    print(f"  Diff from 0.5: ({abs((pixel_x - math.floor(pixel_x)) - 0.5):.10e}, {abs((pixel_y - math.floor(pixel_y)) - 0.5):.10e})")
    print()
