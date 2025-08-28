import os
import logging
from PIL import Image
import numpy as np

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

path1 = r"C:\Users\mehul\P2-P10-Project\data\batchpred4"
path2 = r"C:\Users\mehul\P2-P10-Project\data\batchlobulepred4"
output_path = r"C:\Users\mehul\P2-P10-Project\data\batchblended_output4"

os.makedirs(output_path, exist_ok=True)

processed_count = 0
skipped_count = 0

for filename in os.listdir(path1):
    if filename.lower().endswith(".tiff") and os.path.exists(os.path.join(path2, filename)):
        file1 = os.path.join(path1, filename)
        file2 = os.path.join(path2, filename)
        
        try:
            # Open and convert to RGB
            img1 = Image.open(file1).convert("RGB")
            img2 = Image.open(file2).convert("RGB")
            if img1.size != img2.size:
                logger.warning(f"Skipping {filename}: Image sizes do not match")
                skipped_count += 1
                continue

            arr1 = np.array(img1)
            arr2 = np.array(img2)
            
            mask = ~np.all(arr1 == [255, 255, 255], axis=-1)
            
            blended = np.where(mask[..., None], arr1, arr2)
            
            result_img = Image.fromarray(blended.astype('uint8'))
            result_img.save(os.path.join(output_path, filename))
            
            processed_count += 1
            
        except Exception as e:
            logger.error(f"Error processing {filename}: {e}")
            skipped_count += 1

logger.info(f"Blending complete. Processed: {processed_count}, Skipped: {skipped_count}")