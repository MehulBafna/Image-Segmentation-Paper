import cv2
import os

# Set your paths
base_dir = r"C:\Users\mehul\P2-P10-Project\data\batchpatch4"
mask_dir = r"C:\Users\mehul\P2-P10-Project\data\batchblended_output4"
output_dir = r"C:\Users\mehul\P2-P10-Project\data\batchoverlay_output4"
os.makedirs(output_dir, exist_ok=True)

# Loop over the base image files
for file in os.listdir(base_dir):
    if file.endswith(".tiff") or file.endswith(".tiff"):
        base_path = os.path.join(base_dir, file)
        mask_path = os.path.join(mask_dir, "mask_" + file)

        if not os.path.exists(mask_path):
            print(f"Mask not found for {file}, skipping.")
            continue

        # Read the base and mask images
        base_img = cv2.imread(base_path)
        mask_img = cv2.imread(mask_path)

        if base_img.shape != mask_img.shape:
            print(f"Size mismatch for {file}, skipping.")
            continue

        # Blend images with transparency (alpha)
        alpha = 0.4  # Adjust transparency here
        overlay = cv2.addWeighted(base_img, 1 - alpha, mask_img, alpha, 0)

        # Save result
        out_path = os.path.join(output_dir, f"overlay_{file}")
        cv2.imwrite(out_path, overlay)
