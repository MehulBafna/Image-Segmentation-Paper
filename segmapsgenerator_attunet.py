import os
import torch
from Unet_seb_att import UNet 
from PIL import Image
import numpy as np
from torchvision import transforms
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class_to_rgb = {
    0: (0, 0, 0),        # Background -> Class 0
    1: (0, 0, 255),      # Portal vein (blue) -> Class 1
    2: (0, 255, 0),      # Central vein (green) -> Class 2
    3: (255, 0, 0),      # Artery (red) -> Class 3
    4: (128, 0, 128),    # Bile duct (purple) -> Class 4
}

def save_colored_masks(predicted_mask, original_size, output_dir, file_name):
    
    os.makedirs(output_dir, exist_ok=True)

    h, w = predicted_mask.shape
    rgb_mask = np.zeros((h, w, 3), dtype=np.uint8)

    for class_idx, rgb in class_to_rgb.items():
        rgb_mask[predicted_mask == class_idx] = rgb

    mask_image = Image.fromarray(rgb_mask).resize(original_size, resample=Image.NEAREST)
    mask_image.save(os.path.join(output_dir, file_name), format="TIFF")

# Adjust the paths as per the directory 
def main():
    images_test_path = "images"
    output_masks_path = "testsetmaps"
    model_path = "unet_seb_att.pth"

    if not os.path.exists(model_path):
        return
        
    if not os.path.exists(images_test_path):
        return

    os.makedirs(output_masks_path, exist_ok=True)

    model = UNet(n_channels=3, n_classes=5)
    
    try:
        checkpoint = torch.load(model_path, map_location=device)
        
        if "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        else:
            model.load_state_dict(checkpoint)
            
    except Exception as e:
        return
    
    model.to(device)
    model.eval()
    
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])

    try:
        image_files = [f for f in os.listdir(images_test_path) 
                    if f.endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff'))]
        image_paths = [os.path.join(images_test_path, f) for f in image_files]
        
        if not image_paths:
            return
            
    except Exception as e:
        return

    with torch.no_grad():
        for i, image_path in enumerate(image_paths):
            try:
                original_image = Image.open(image_path)
                original_size = original_image.size 

                image = original_image.convert("RGB")
                image_tensor = test_transform(image).unsqueeze(0).to(device) 

                output = model(image_tensor)
                pred_mask = torch.argmax(output, dim=1).squeeze().cpu().numpy()

                base_name = os.path.basename(image_path)
                file_name = f"mask_{base_name}"
                if not file_name.endswith('.tiff'):
                    file_name = file_name.rsplit('.', 1)[0] + '.tiff'
                save_colored_masks(pred_mask, original_size, output_masks_path, file_name)
                
            except Exception as e:
                continue

if __name__ == "__main__":
    main()
