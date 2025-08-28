import os
import numpy as np
import openslide
from PIL import Image

class PatchExtractor:
    def __init__(self, slide_path, output_folder):
        self.slide_path = os.path.abspath(slide_path)
        self.output_folder = os.path.abspath(output_folder)
        os.makedirs(self.output_folder, exist_ok=True)

        self.slide = openslide.OpenSlide(self.slide_path)
        self.width, self.height = self.slide.dimensions
        print(f"Slide dimensions: {self.width} x {self.height}")

        # Automatically determine patch width and height
        self.patch_width = self._get_largest_factor_in_range(self.width, 900, 1200)
        self.patch_height = self._get_largest_factor_in_range(self.height, 900, 1200)
        print(f"Chosen patch size (before 32 alignment): {self.patch_width} x {self.patch_height}")

        # Calculate patch width/height factors BEFORE 32 alignment
        self.patch_width_factor = self.width // self.patch_width
        self.patch_height_factor = self.height // self.patch_height

        # Make patch size a multiple of 32
        self.target_patch_width = self._make_multiple_of_32(self.patch_width)
        self.target_patch_height = self._make_multiple_of_32(self.patch_height)
        print(f"Final padded patch size (multiple of 32): {self.target_patch_width} x {self.target_patch_height}")
        print(f"Patch width factor: {self.patch_width_factor}, Patch height factor: {self.patch_height_factor}")

        self.num_patches_x = int(np.ceil(self.width / self.patch_width))
        self.num_patches_y = int(np.ceil(self.height / self.patch_height))
        self.patch_count = 1

    def _get_largest_factor_in_range(self, value, min_val, max_val):
        # Finds the largest factor of `value` within the range [min_val, max_val]
        factors = [i for i in range(min_val, max_val + 1) if value % i == 0]
        return max(factors) if factors else min_val  # Fallback to min_val if no valid factor found

    def _make_multiple_of_32(self, value):
        remainder = value % 32
        return value if remainder == 0 else value + (32 - remainder)

    def extract_patches(self):
        for i in range(self.num_patches_x):
            for j in range(self.num_patches_y):
                x = i * self.patch_width
                y = j * self.patch_height

                region_width = min(self.patch_width, self.width - x)
                region_height = min(self.patch_height, self.height - y)
                print(f"Reading patch at coordinates: ({x}, {y}), width: {region_width}, height: {region_height}")

                region = self.slide.read_region((x, y), 0, (region_width, region_height)).convert("RGB")

                # Pad the region to match the final target size
                padded_region = Image.new("RGB", (self.target_patch_width, self.target_patch_height), color=(0, 0, 0))
                padded_region.paste(region, (0, 0))

                filename_uncompressed = os.path.join(self.output_folder, f'patch_{self.patch_count}.tiff')
                padded_region.save(filename_uncompressed, format='TIFF', compression='none')

                self.patch_count += 1

    def close(self):
        self.slide.close()


# Usage as per the path
slide_path1 = r"Path of slide"
output_folder = r"output path"

patch_extractor = PatchExtractor(slide_path1, output_folder)
patch_extractor.extract_patches()
patch_extractor.close()
