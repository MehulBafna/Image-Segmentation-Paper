import os
import numpy as np
import pandas as pd
import cv2
import ast

OPENSLIDE_PATH = r'C:\Program Files\Openslide\openslide-bin-4.0.0.6-windows-x64\openslide-bin-4.0.0.6-windows-x64\bin'

if hasattr(os, 'add_dll_directory'):
    # Windows
    with os.add_dll_directory(OPENSLIDE_PATH):
        import openslide
else:
    import openslide
    
class PatchMaskGenerator:
    def __init__(self, slide_path, csv_file, image_directory, mask_directory, num_patches_x=16, num_patches_y=16):
        self.slide = openslide.OpenSlide(slide_path)
        self.annotations_df = pd.read_csv(csv_file)
        self.annotations_df['Coordinates'] = self.annotations_df['Coordinates'].apply(ast.literal_eval)
        self.slide_width, self.slide_height = self.slide.dimensions
        self.patch_width = self.slide_width // num_patches_x
        self.patch_height = self.slide_height // num_patches_y
        self.image_directory = image_directory
        self.mask_directory = mask_directory
        os.makedirs(self.mask_directory, exist_ok=True)
        self.final_image_paths = []
        self.final_mask_paths = {color: [] for color in self.annotations_df['Color'].unique()}

    def process_patches(self):
        for patch_index_x in range(self.patch_width):
            for patch_index_y in range(self.patch_height):
                patch_top_left_x = patch_index_x * self.patch_width
                patch_top_left_y = patch_index_y * self.patch_height
                masks = {color: np.zeros((self.patch_height, self.patch_width), dtype=np.uint8) for color in self.final_mask_paths.keys()}
                patch_index = patch_index_x * self.patch_height + patch_index_y
                image_path = os.path.join(self.image_directory, f'patch_{patch_index + 1}.tiff')
                self.final_image_paths.append(image_path)

                for index, row in self.annotations_df.iterrows():
                    color_name = row['Color']
                    coordinates = np.array(row['Coordinates'], dtype=np.float32)
                    within_patch = np.logical_and(
                        np.logical_and(coordinates[:, 0] >= patch_top_left_x, coordinates[:, 0] < patch_top_left_x + self.patch_width),
                        np.logical_and(coordinates[:, 1] >= patch_top_left_y, coordinates[:, 1] < patch_top_left_y + self.patch_height)
                    )

                    if np.any(within_patch):
                        patch_coordinates = coordinates[within_patch]
                        shifted_coordinates = patch_coordinates - np.array([patch_top_left_x, patch_top_left_y])
                        shifted_coordinates = np.array(shifted_coordinates, dtype=np.int32)
                        mirrored_coordinates = shifted_coordinates.copy()
                        mirrored_coordinates[:, 1] = self.patch_height - mirrored_coordinates[:, 1]
                        cv2.fillPoly(masks[color_name], [mirrored_coordinates], color=1)

                for color_name, mask_image in masks.items():
                    mask_path = os.path.join(self.mask_directory, f'{color_name.replace(" ", "_")}_mask_{patch_index + 1}.tiff')
                    cv2.imwrite(mask_path, mask_image * 255, [cv2.IMWRITE_TIFF_COMPRESSION, 1])
                    self.final_mask_paths[color_name].append(mask_path)

    def create_final_dataframe(self):
        final_df = pd.DataFrame({'Image Path': self.final_image_paths})
        for color_name in self.final_mask_paths.keys():
            final_df[color_name.capitalize() + ' Mask Path'] = self.final_mask_paths[color_name]
        return final_df

    def save_final_dataframe(self, output_csv):
        final_df = self.create_final_dataframe()
        final_df.to_csv(output_csv, index=False)

    def close(self):
        self.slide.close()


# Example
slide_path = r"C:\Users\Mehul\Documents\MATLAB\SSES-1 10_J-21-152_100_Pig_HE_RUN23__liver_MAX.ndpi"
csv_file = r'C:\Users\Mehul\Documents\MATLAB\annotations2.csv'
image_directory = r'C:\Users\Mehul\Documents\MATLAB\Patched_Images'
mask_directory = r'C:\Users\Mehul\Documents\MATLAB\Masksupdated1'
output_csv = r'C:\Users\Mehul\Documents\MATLAB\final_dataset.csv'

patch_mask_generator = PatchMaskGenerator(slide_path, csv_file, image_directory, mask_directory)
patch_mask_generator.process_patches()
patch_mask_generator.save_final_dataframe(output_csv)
patch_mask_generator.close()
