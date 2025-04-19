import os
import numpy as np
import openslide
from PIL import Image

class PatchExtractor:
    def __init__(self, slide_path, output_folder, patch_width_factor=112, patch_height_factor=64, target_patch_width=1056, target_patch_height=1056):
        
        self.slide_path = os.path.abspath(slide_path)
        self.output_folder = os.path.abspath(output_folder)
        
        os.makedirs(self.output_folder, exist_ok=True)

        self.patch_width_factor = patch_width_factor
        self.patch_height_factor = patch_height_factor
        self.target_patch_width = target_patch_width
        self.target_patch_height = target_patch_height
        
        self.slide = openslide.OpenSlide(self.slide_path)
        self.width, self.height = self.slide.dimensions
        self.patch_width = self.width // self.patch_width_factor
        self.patch_height = self.height // self.patch_height_factor
        
        self.num_patches_x = int(np.ceil(self.width / self.patch_width))
        self.num_patches_y = int(np.ceil(self.height / self.patch_height))
        
        self.patch_count = 1
    
    def extract_patches(self):
        for i in range(self.num_patches_x):
            for j in range(self.num_patches_y):
                x = i * self.patch_width
                y = j * self.patch_height
                region_width = min(self.patch_width, self.width - x)
                region_height = min(self.patch_height, self.height - y)
                region = self.slide.read_region((x, y), 0, (region_width, region_height))
                region = region.convert("RGB")

                padded_region = Image.new("RGB", (self.target_patch_width, self.target_patch_height), color=(0, 0, 0))

                padded_region.paste(region, (0, 0))
                filename_uncompressed = os.path.join(self.output_folder, f'patch_{self.patch_count}.tiff')
                padded_region.save(filename_uncompressed, format='TIFF', compression='none')

                self.patch_count += 1

    def close(self):
        self.slide.close()

#slide_path to be replaced with actual path where the .ndpi file is stored
slide_path = r"SSES2021 14 LIVER LL 14 7 21_J-21-157_4_Pig_GS 1-10000 + PSR_RUN08_Part II_Liver LL 0h_VAB.ndpi"
output_folder = "images"
patch_extractor = PatchExtractor(slide_path, output_folder)
patch_extractor.extract_patches()
patch_extractor.close()
