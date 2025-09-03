import os
import torch
from modellobule import BinaryUNet
from PIL import Image
import numpy as np
from torchvision import transforms
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class_to_rgb = {
    0: (0,0,0),
    1: (255,255,0)   # Bile duct (purple) -> Class 4
}

def save_colored_masks(predicted_mask, original_size, output_dir, file_name):
    
    os.makedirs(output_dir, exist_ok=True)

    h, w = predicted_mask.shape
    rgb_mask = np.zeros((h, w, 3), dtype=np.uint8)

    for class_idx, rgb in class_to_rgb.items():
        rgb_mask[predicted_mask == class_idx] = rgb

    mask_image = Image.fromarray(rgb_mask).resize(original_size, resample=Image.NEAREST)
    mask_image.save(os.path.join(output_dir, file_name), format="TIFF")

def convert_black_to_white(mask_dir):
    print(f"Converting black pixels to white in masks from: {mask_dir}")
    
    mask_files = [f for f in os.listdir(mask_dir) 
                 if f.startswith('mask_') and f.endswith(('.tif', '.tiff'))]
    
    processed_count = 0
    for mask_file in mask_files:
        try:
            mask_path = os.path.join(mask_dir, mask_file)
            mask = Image.open(mask_path)
            mask_array = np.array(mask)
            
            # Find black pixels (0,0,0) and convert to white (255,255,255)
            black_pixels = np.all(mask_array == [0,0,0], axis=-1)
            mask_array[black_pixels] = [255, 255, 255]
            
            # Save the modified mask
            processed_mask = Image.fromarray(mask_array)
            processed_mask.save(mask_path)
            processed_count += 1
            
            if processed_count % 10 == 0:
                print(f"Processed {processed_count} masks")
                
        except Exception as e:
            print(f"Error processing {mask_file}: {str(e)}")
            continue
    
    print(f"Completed! Converted black pixels to white in {processed_count} mask images.")

def main():
    images_test_path = r"C:\Users\mehul\P2-P10-Project\data\patch4"
    output_masks_path = r"C:\Users\mehul\P2-P10-Project\data\lobulepred4"
    model_path = r"C:\Users\mehul\P2-P10-Project\data\LobuleModel.pth"

    if not os.path.exists(model_path):
        print(f"Model path not found: {model_path}")
        return
        
    if not os.path.exists(images_test_path):
        print(f"Test images path not found: {images_test_path}")
        return

    os.makedirs(output_masks_path, exist_ok=True)

    model = BinaryUNet(n_channels=3)
    
    try:
        checkpoint = torch.load(model_path, map_location=device)
        
        if "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        else:
            model.load_state_dict(checkpoint)
        print("Model loaded successfully")
            
    except Exception as e:
        print(f"Error loading model: {str(e)}")
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
            print(f"No image files found in: {images_test_path}")
            return
        
        print(f"Found {len(image_paths)} images to process")
            
    except Exception as e:
        print(f"Error listing image files: {str(e)}")
        return

    with torch.no_grad():
        for i, image_path in enumerate(image_paths):
            try:
                original_image = Image.open(image_path)
                original_size = original_image.size 

                image = original_image.convert("RGB")
                image_tensor = test_transform(image).unsqueeze(0).to(device) 

                output = model(image_tensor)
                pred_mask = (output.squeeze(0).squeeze(0) > 0.5).cpu().numpy().astype(np.uint8)

                # Debug: print unique values
                if i % 10 == 0:
                    print(f"Processing image {i+1}/{len(image_paths)}: {os.path.basename(image_path)}")
                    print(f"Unique values in prediction: {np.unique(pred_mask)}")

                base_name = os.path.basename(image_path)
                file_name = f"mask_{base_name}"
                if not file_name.endswith('.tiff'):
                    file_name = file_name.rsplit('.', 1)[0] + '.tiff'
                save_colored_masks(pred_mask, original_size, output_masks_path, file_name)
                
            except Exception as e:
                print(f"Error processing {image_path}: {str(e)}")
                continue
    
    print("\nStarting black to white pixel conversion...")
    convert_black_to_white(output_masks_path)

if __name__ == "__main__":
    main()
