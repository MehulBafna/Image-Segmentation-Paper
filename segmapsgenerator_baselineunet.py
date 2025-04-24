import os
import torch
from Baselineunet import SimpleUNet  
from PIL import Image
import numpy as np
from torchvision import transforms

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class_to_rgb = {
    0: (0, 0, 0),        # Background
    1: (0, 0, 255),      # Portal vein
    2: (0, 255, 0),      # Central vein
    3: (255, 0, 0),      # Artery
    4: (128, 0, 128),    # Bile duct
}

def save_colored_masks(predicted_mask, original_size, output_dir, file_name):
    os.makedirs(output_dir, exist_ok=True)

    h, w = predicted_mask.shape
    rgb_mask = np.zeros((h, w, 3), dtype=np.uint8)

    for class_idx, rgb in class_to_rgb.items():
        rgb_mask[predicted_mask == class_idx] = rgb

    mask_image = Image.fromarray(rgb_mask).resize(original_size, resample=Image.NEAREST)
    mask_image.save(os.path.join(output_dir, file_name), format="TIFF")

# Adjust paths as per directory
def main():
    images_test_path = "images"
    output_masks_path = "testset"
    model_path = "Baseline.pth" 

    if not os.path.exists(model_path) or not os.path.exists(images_test_path):
        return

    os.makedirs(output_masks_path, exist_ok=True)

    model = SimpleUNet(n_channels=3, n_classes=5)
    
    try:
        checkpoint = torch.load(model_path, map_location=device)
        checkpoint = {k.replace('module.', ''): v for k, v in checkpoint.items()}
        
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
                       if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff'))]
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
                file_name = f"mask_{os.path.splitext(base_name)[0]}.tiff"
                save_colored_masks(pred_mask, original_size, output_masks_path, file_name)

            except Exception as e:
                continue

if __name__ == "__main__":
    main()
