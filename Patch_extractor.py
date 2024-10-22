import os
import numpy as np
import openslide
from PIL import Image

OPENSLIDE_PATH = r'C:\Program Files\Openslide\openslide-bin-4.0.0.6-windows-x64\openslide-bin-4.0.0.6-windows-x64\bin'

import os
if hasattr(os, 'add_dll_directory'):
    # Windows
    with os.add_dll_directory(OPENSLIDE_PATH):
        import openslide
else:
    import openslide

class PatchExtractor:
    def __init__(self, slide_path, output_folder, num_columns=16, num_rows=16):
        self.slide_path = slide_path
        self.slide = openslide.OpenSlide(slide_path)
        self.width, self.height = self.slide.dimensions
        self.patch_width = int(self.width / num_columns)
        self.patch_height = int(self.height / num_rows)
        self.num_patches_x = int(np.ceil(self.width / self.patch_width))
        self.num_patches_y = int(np.ceil(self.height / self.patch_height))
        self.output_folder = output_folder
        os.makedirs(self.output_folder, exist_ok=True)
        self.patch_count = 1

    def extract_patch(self, i, j):
        x = i * self.patch_width
        y = j * self.patch_height
        region_width = min(self.patch_width, self.width - x)
        region_height = min(self.patch_height, self.height - y)
        region = self.slide.read_region((x, y), 0, (region_width, region_height))
        region = region.convert("RGB")
        return region

    def save_patch(self, patch, patch_count):
        filename = os.path.join(self.output_folder, f'patch_{patch_count}.tiff')
        patch.save(filename, format='TIFF', compression='none')

    def generate_patches(self):
        for i in range(self.num_patches_x):
            for j in range(self.num_patches_y - 1, -1, -1):
                patch = self.extract_patch(i, j)
                self.save_patch(patch, self.patch_count)
                self.patch_count += 1

    def close(self):
        self.slide.close()


# Example
slide_path = r"C:\Users\Mehul\Documents\MATLAB\SSES-1 10_J-21-152_100_Pig_HE_RUN23__liver_MAX.ndpi"
output_folder = r'C:\Users\Mehul\Documents\MATLAB\Patched_Images_uncompressed'

patch_extractor = PatchExtractor(slide_path, output_folder)
patch_extractor.generate_patches()
patch_extractor.close()
