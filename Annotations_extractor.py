import os
import numpy as np

# Define the OpenSlide DLL directory and configuration
OPENSLIDE_PATH = r'C:\Program Files\Openslide\openslide-bin-4.0.0.6-windows-x64\openslide-bin-4.0.0.6-windows-x64\bin'

if hasattr(os, 'add_dll_directory'):
    with os.add_dll_directory(OPENSLIDE_PATH):
        import openslide
else:
    import openslide

class NDPIImagePatcher:
    def __init__(self, slide_path, output_folder, patch_divisor=16):
        self.slide_path = slide_path
        self.output_folder = output_folder
        self.patch_divisor = patch_divisor
        self.slide = None
        self.width = None
        self.height = None
        self.patch_width = None
        self.patch_height = None
        self.num_patches_x = None
        self.num_patches_y = None
        self.patch_count = 1

        os.makedirs(self.output_folder, exist_ok=True)

    def load_slide(self):
        """Load the NDPI file using OpenSlide."""
        self.slide = openslide.OpenSlide(self.slide_path)
        self.width, self.height = self.slide.dimensions
        self.patch_width = int(self.width / self.patch_divisor)
        self.patch_height = int(self.height / self.patch_divisor)
        self.num_patches_x = int(np.ceil(self.width / self.patch_width))
        self.num_patches_y = int(np.ceil(self.height / self.patch_height))
        print(f"Slide loaded: {self.width}x{self.height}, Patch size: {self.patch_width}x{self.patch_height}")

    def extract_patches(self):
        """Extract patches from the slide and save them in different TIFF formats."""
        for i in range(self.num_patches_x):
            for j in range(self.num_patches_y):
                x = i * self.patch_width
                y = j * self.patch_height

                region_width = min(self.patch_width, self.width - x)
                region_height = min(self.patch_height, self.height - y)

                # Read the patch
                region = self.slide.read_region((x, y), 0, (region_width, region_height)).convert("RGB")

                # Save patches in different TIFF formats as per requirements
                self.save_patch(region)

    def save_patch(self, region):
        """Save the extracted patch in various TIFF formats."""
        # Uncompressed TIFF
        filename_uncompressed = os.path.join(self.output_folder, f'patch_{self.patch_count}.tiff')
        region.save(filename_uncompressed, format='TIFF', compression='none')

        # LZW Compressed TIFF
        #filename_lzw = os.path.join(self.output_folder, f'patch_{self.patch_count}_2.tiff')
        #region.save(filename_lzw, format='TIFF', compression='tiff_lzw')

        # Deflate Compressed TIFF
        #filename_deflate = os.path.join(self.output_folder, f'patch_{self.patch_count}_3.tiff')
        #region.save(filename_deflate, format='TIFF', compression='tiff_adobe_deflate')

        # Increment patch counter
        self.patch_count += 1

    def cleanup(self):
        if self.slide:
            self.slide.close()

    def process(self):
        try:
            self.load_slide()
            self.extract_patches()
        finally:
            self.cleanup()

if __name__ == "__main__":
    slide_path = input(r'Enter path of slide : ')#r"C:\Users\Mehul\Documents\MATLAB\SSES-1 10_J-21-152_100_Pig_HE_RUN23__liver_MAX.ndpi"
    output_folder = input(r'Enter patch where patches are required to be saved : ')#r'C:\Users\Mehul\Documents\MATLAB\Patched_Images'
    patcher = NDPIImagePatcher(slide_path, output_folder)
    patcher.process()
