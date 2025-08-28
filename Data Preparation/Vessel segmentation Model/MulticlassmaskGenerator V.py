import os
import numpy as np
import pandas as pd
import ast
import torch
from PIL import Image, ImageDraw
import openslide

def save_image(tensor, filepath, normalize=False):
    if normalize:
        tensor = tensor.clone()  
        if tensor.min() < 0:
            tensor = tensor - tensor.min()
        if tensor.max() > 0:
            tensor = tensor / tensor.max()
    
    if len(tensor.shape) == 3:  
        ndarr = tensor.mul(255).clamp(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
        img = Image.fromarray(ndarr)
        img.save(filepath)
    else:
        raise ValueError(f"Unsupported tensor shape: {tensor.shape}")

def to_tensor(pic):
    """Convert a PIL Image to a tensor"""
    if not isinstance(pic, Image.Image):
        raise TypeError(f'pic should be PIL Image. Got {type(pic)}')
    
    img = np.array(pic)
    
    if img.ndim == 2:
        img = img[:, :, None]
    
    img = img.transpose((2, 0, 1))
    img = torch.from_numpy(img.copy())
    if isinstance(img, torch.ByteTensor):
        return img.float().div(255)
    else:
        return img

class PatchMaskGenerator:
    def __init__(self, annotations_csv, image_dir, mask_dir, slide_path):
        self.annotations_csv = annotations_csv
        self.image_directory = image_dir
        self.mask_directory = mask_dir
        self.slide_path = slide_path

        os.makedirs(self.mask_directory, exist_ok=True)

        self.slide = openslide.OpenSlide(self.slide_path)
        self.slide_width, self.slide_height = self.slide.dimensions
        print(f"Slide dimensions: {self.slide_width} x {self.slide_height}")

        # Use automated patch sizing like in PatchExtractor
        self.patch_width = self._get_largest_factor_in_range(self.slide_width, 900, 1200)
        self.patch_height = self._get_largest_factor_in_range(self.slide_height, 900, 1200)
        print(f"Chosen patch size (before 32 alignment): {self.patch_width} x {self.patch_height}")

        # Calculate patch width/height factors BEFORE 32 alignment
        self.patch_width_factor = self.slide_width // self.patch_width
        self.patch_height_factor = self.slide_height // self.patch_height

        # Make patch size a multiple of 32 (target size for padding)
        self.target_width = self._make_multiple_of_32(self.patch_width)
        self.target_height = self._make_multiple_of_32(self.patch_height)
        print(f"Final padded patch size (multiple of 32): {self.target_width} x {self.target_height}")
        print(f"Patch width factor: {self.patch_width_factor}, Patch height factor: {self.patch_height_factor}")

        self.num_patches_x = int(np.ceil(self.slide_width / self.patch_width))
        self.num_patches_y = int(np.ceil(self.slide_height / self.patch_height))
        print(f"Number of patches: {self.num_patches_x} x {self.num_patches_y} = {self.num_patches_x * self.num_patches_y}")

        self.color_mapping = {
            'portal vein': (0, 0, 1),
            'central vein': (0, 1, 0),
            'bile duct': (0.5, 0, 0.5),
            'artery': (1, 0, 0)
        }

    def _get_largest_factor_in_range(self, value, min_val, max_val):
        """Finds the largest factor of `value` within the range [min_val, max_val]"""
        factors = [i for i in range(min_val, max_val + 1) if value % i == 0]
        return max(factors) if factors else min_val  # Fallback to min_val if no valid factor found

    def _make_multiple_of_32(self, value):
        """Make value a multiple of 32 by padding up"""
        remainder = value % 32
        return value if remainder == 0 else value + (32 - remainder)

    def load_data(self):
        self.annotations_df = pd.read_csv(self.annotations_csv)
        self.annotations_df['Coordinates'] = self.annotations_df['Coordinates'].apply(ast.literal_eval)

    def generate_masks(self):
        self.final_image_paths = []
        self.final_mask_paths = []

        for patch_index_x in range(self.num_patches_x):
            for patch_index_y in range(self.num_patches_y):
                patch_top_left_x = patch_index_x * self.patch_width
                patch_top_left_y = patch_index_y * self.patch_height

                combined_mask = torch.zeros((3, self.patch_height, self.patch_width), dtype=torch.float32)

                patch_index = patch_index_x * self.num_patches_y + patch_index_y
                image_path = os.path.join(self.image_directory, f'patch_{patch_index + 1}.tiff')
                self.final_image_paths.append(image_path)

                # Note: You may need to adjust this mask_index calculation based on your specific indexing scheme
                mask_index = self.num_patches_y - 1 - patch_index_y if patch_index_y < self.num_patches_y else (2 * self.num_patches_y) - 1 - patch_index_y

                for _, row in self.annotations_df.iterrows():
                    color_name = row['Color'].lower()
                    coordinates = row['Coordinates']

                    if color_name not in self.color_mapping:
                        continue

                    color = torch.tensor(self.color_mapping[color_name]).view(3, 1, 1)

                    if len(coordinates) > 3 and isinstance(coordinates[0], tuple):
                        coordinates = np.array(coordinates, dtype=np.float32)
                        within_patch = np.logical_and(
                            np.logical_and(coordinates[:, 0] >= patch_top_left_x, coordinates[:, 0] < patch_top_left_x + self.patch_width),
                            np.logical_and(coordinates[:, 1] >= patch_top_left_y, coordinates[:, 1] < patch_top_left_y + self.patch_height)
                        )
                        if np.any(within_patch):
                            shifted_coordinates = coordinates - np.array([patch_top_left_x, patch_top_left_y])
                            mirrored_coordinates = shifted_coordinates.copy()
                            mirrored_coordinates[:, 1] = self.patch_height - shifted_coordinates[:, 1]

                            shifted_coordinates = mirrored_coordinates.astype(int)

                            pil_mask = Image.new("RGB", (self.patch_width, self.patch_height), (0, 0, 0))
                            draw = ImageDraw.Draw(pil_mask)
                            draw.polygon(shifted_coordinates.flatten().tolist(), outline=(255, 255, 255), fill=(255, 255, 255))

                            mask_tensor = to_tensor(pil_mask)
                            combined_mask = torch.where(mask_tensor > 0, color, combined_mask)

                    elif len(coordinates[0]) == 3:
                        center_x, center_y, radius = coordinates[0]

                        if (patch_top_left_x <= center_x < patch_top_left_x + self.patch_width and
                                patch_top_left_y <= center_y < patch_top_left_y + self.patch_height):

                            local_center_x = int(center_x - patch_top_left_x)
                            local_center_y = int(center_y - patch_top_left_y)

                            mirrored_center_y = self.patch_height - local_center_y

                            pil_mask = Image.new("RGB", (self.patch_width, self.patch_height), (0, 0, 0))
                            draw = ImageDraw.Draw(pil_mask)
                            draw.ellipse(
                                (local_center_x - radius, mirrored_center_y - radius,
                                 local_center_x + radius, mirrored_center_y + radius),
                                outline=(255, 255, 255), fill=(255, 255, 255)
                            )

                            mask_tensor = to_tensor(pil_mask)
                            combined_mask = torch.where(mask_tensor > 0, color, combined_mask)

                # Pad the mask to match the target size (multiple of 32)
                padded_mask = torch.zeros((3, self.target_height, self.target_width), dtype=torch.float32)
                padded_mask[:, :self.patch_height, :self.patch_width] = combined_mask

                mask_name = f'mask_{mask_index + self.num_patches_y * patch_index_x+1}.tiff'
                mask_path = os.path.join(self.mask_directory, mask_name)
                save_image(padded_mask, mask_path)
                self.final_mask_paths.append(mask_path)

# Paths required to be changed alternatively
if __name__ == "__main__":
    generator = PatchMaskGenerator(
        annotations_csv=r"C:\Users\mehul\Image-Segmentation-Paper\Dataset\Vascular Structure Detection Model\Annotationsv.csv",
        image_dir=r"C:\Users\mehul\P2-P10-Project\data\images\images",
        mask_dir=r"C:\Users\mehul\P2-P10-Project\data\images\masks",
        slide_path=r"C:\Users\mehul\Downloads\WholeSlideImage\WSI.ndpi"
    )
    generator.load_data()
    generator.generate_masks()