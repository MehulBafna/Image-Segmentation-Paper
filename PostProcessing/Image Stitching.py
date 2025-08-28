import os
from PIL import Image, ImageFile
import numpy as np
import openslide

# Allow loading of truncated TIFF images
ImageFile.LOAD_TRUNCATED_IMAGES = True

def get_largest_factor_in_range(value, min_val, max_val):
    """
    Finds the largest factor of `value` within the range [min_val, max_val]
    This ensures patch_width is a factor of slide_width AND within the specified range
    """
    factors = [i for i in range(min_val, max_val + 1) if value % i == 0]
    if factors:
        return max(factors)
    else:
        print(f"⚠️ Warning: No factors of {value} found in range [{min_val}, {max_val}]. Using fallback: {min_val}")
        return min_val  # Fallback to min_val if no valid factor found

# Removed make_multiple_of_32 function - not needed

# Constants
input_dir = "/work/xi47luy/batchoverlay_output4"
slide_path = "/path/to/your/slide.ndpi"  # Update this path
output_file = "/work/xi47luy/image4.jpg"

# Get slide dimensions using openslide
slide = openslide.OpenSlide(slide_path)
slide_width, slide_height = slide.dimensions
slide.close()

print(f"Slide dimensions: {slide_width} x {slide_height}")

# Calculate patch dimensions - use the largest factor in range as final image size
# patch_width will be a factor of slide_width AND within range [900, 1220]
patch_width = get_largest_factor_in_range(slide_width, 900, 1220)
# patch_height will be a factor of slide_height AND within range [900, 1200]  
patch_height = get_largest_factor_in_range(slide_height, 900, 1200)

# Use patch dimensions directly as image dimensions (no padding to multiple of 32)
image_width = patch_width
image_height = patch_height

print(f"Final patch/image size: {image_width} x {image_height}")

# Verify the factor requirement
print(f"Verification: {slide_width} ÷ {patch_width} = {slide_width // patch_width} (exact division: {slide_width % patch_width == 0})")
print(f"Verification: {slide_height} ÷ {patch_height} = {slide_height // patch_height} (exact division: {slide_height % patch_height == 0})")

# Calculate grid dimensions
patches_x = int(np.ceil(slide_width / patch_width))
patches_y = int(np.ceil(slide_height / patch_height))
print(f"Number of patches: {patches_x} x {patches_y}")

# Set parameters based on calculations
rows_per_column = patches_y  # slide_height / patch_height
columns = patches_x          # slide_width / patch_width
total_images = patches_x * patches_y

print(f"Grid layout: {rows_per_column} rows × {columns} columns = {total_images} total images")
print(f"Canvas size will be: {columns * image_width} x {rows_per_column * image_height}")

# Create empty canvas
canvas = np.zeros((rows_per_column * image_height, columns * image_width, 3), dtype=np.uint8)

# Optional: keep track of problematic images
skipped_images = []

# Process each image
for i in range(1, total_images + 1):
    img_path = os.path.join(input_dir, f"overlay_patch_{i}.tiff")
    try:
        img = Image.open(img_path).convert("RGB")
        
        # Crop to the actual patch size (remove padding if any)
        img = img.crop((0, 0, image_width, image_height))
        img_np = np.array(img)

        # Calculate position in the canvas
        col = (i - 1) // rows_per_column
        row = (i - 1) % rows_per_column

        y_start = row * image_height
        y_end = y_start + image_height
        x_start = col * image_width
        x_end = x_start + image_width

        canvas[y_start:y_end, x_start:x_end, :] = img_np
        
        if i % 100 == 0 or i == total_images:  # Print progress every 100 images
            print(f"Processed image {i}/{total_images}")
    
    except Exception as e:
        print(f"⚠️ Error loading image {img_path}: {e}")
        skipped_images.append((i, img_path))

# Convert canvas to PIL Image
img = Image.fromarray(canvas)
print(f"Final canvas size: {img.width} x {img.height}")

# Resize to half size with high-quality downsampling
half_size = (img.width // 2, img.height // 2)
img_resized = img.resize(half_size, Image.LANCZOS)
print(f"Resized to: {img_resized.width} x {img_resized.height}")

# Save as JPEG
img_resized.save(output_file, quality=90)
print(f"\n✅ Stitched image saved to: {output_file}")

# Optional: Report skipped images
if skipped_images:
    skipped_log = "/work/xi47luy/skipped_images.txt"
    with open(skipped_log, "w") as f:
        for idx, path in skipped_images:
            f.write(f"{idx}: {path}\n")
    print(f"\n⚠️ {len(skipped_images)} images were skipped due to errors. Details saved to: {skipped_log}")
else:
    print("\n✅ All images processed successfully.")

print(f"\nSummary:")
print(f"- Slide dimensions: {slide_width} x {slide_height}")
print(f"- Patch size: {patch_width} x {patch_height}")
print(f"- Padded patch size: {image_width} x {image_height}")
print(f"- Grid: {patches_x} x {patches_y} = {total_images} patches")
print(f"- Final image size: {img_resized.width} x {img_resized.height}")