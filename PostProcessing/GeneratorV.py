import os
import torch
from Baselineunet import SimpleUNet 
from PIL import Image
import numpy as np
from torchvision import transforms

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class_to_rgb = {
    0: (0, 0, 0),        # Background
    1: (0, 0, 255),      # Portal vein (blue)
    2: (0, 255, 0),      # Central vein (green)
    3: (255, 0, 0),      # Artery (red)
    4: (128, 0, 128),    # Bile duct (purple)
}

def save_colored_masks(predicted_mask, original_size, output_dir, file_name):
    os.makedirs(output_dir, exist_ok=True)
    h, w = predicted_mask.shape
    rgb_mask = np.zeros((h, w, 3), dtype=np.uint8)

    for class_idx, rgb in class_to_rgb.items():
        rgb_mask[predicted_mask == class_idx] = rgb

    mask_image = Image.fromarray(rgb_mask).resize(original_size, resample=Image.NEAREST)
    mask_image.save(os.path.join(output_dir, file_name), format="TIFF")

def is_mostly_white(image, threshold=0.8):
    img_np = np.array(image.convert("RGB"))
    white_pixels = np.sum((img_np[:,:,0] > 210) & (img_np[:,:,1] > 210) & (img_np[:,:,2] > 210))
    total_pixels = img_np.shape[0] * img_np.shape[1]
    white_ratio = white_pixels / total_pixels
    print(white_ratio)
    return white_ratio > threshold

def modify_mask(pred_mask, convert_blue_green_to_white=False):
    h, w = pred_mask.shape
    rgb_mask = np.zeros((h, w, 3), dtype=np.uint8)

    for class_idx, rgb in class_to_rgb.items():
        rgb_mask[pred_mask == class_idx] = rgb

    if convert_blue_green_to_white:
        blue_pixels = np.all(rgb_mask == [0, 0, 255], axis=-1)
        green_pixels = np.all(rgb_mask == [0, 255, 0], axis=-1)
        rgb_mask[blue_pixels] = [255, 255, 255]
        rgb_mask[green_pixels] = [255, 255, 255]
    else:
        has_red = np.any(np.all(rgb_mask == [255, 0, 0], axis=-1))
        has_purple = np.any(np.all(rgb_mask == [128, 0, 128], axis=-1))

        if has_red or has_purple:
            green_pixels = np.all(rgb_mask == [0, 255, 0], axis=-1)
            rgb_mask[green_pixels] = [0, 0, 255]
        else:
            mask_colors = np.unique(rgb_mask.reshape(-1, rgb_mask.shape[2]), axis=0)
            mask_colors = [tuple(color) for color in mask_colors if tuple(color) != (0, 0, 0)]

            if set(mask_colors).issubset({(0, 0, 255), (0, 255, 0)}):
                blue_count = np.sum(np.all(rgb_mask == [0, 0, 255], axis=-1))
                green_count = np.sum(np.all(rgb_mask == [0, 255, 0], axis=-1))

                if blue_count > green_count:
                    green_pixels = np.all(rgb_mask == [0, 255, 0], axis=-1)
                    rgb_mask[green_pixels] = [0, 0, 255]
                else:
                    blue_pixels = np.all(rgb_mask == [0, 0, 255], axis=-1)
                    rgb_mask[blue_pixels] = [0, 255, 0]

    modified_mask = np.zeros((h, w), dtype=np.uint8)
    for class_idx, rgb in class_to_rgb.items():
        mask = np.all(rgb_mask == rgb, axis=-1)
        modified_mask[mask] = class_idx

    return modified_mask





def main():
    images_test_path = r"C:\Users\mehul\P2-P10-Project\data\batchpatch4"
    output_masks_path = r"C:\Users\mehul\P2-P10-Project\data\batchpred4"
    model_path = r"C:\Users\mehul\P2-P10-Project\VascularStructure.pth"

    if not os.path.exists(model_path) or not os.path.exists(images_test_path):
        return

    os.makedirs(output_masks_path, exist_ok=True)

    model = SimpleUNet(n_channels=3, n_classes=5)

    try:
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint.get("model_state_dict", checkpoint))
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

    images_to_convert = set()
    
    def is_mostly_white_file(path):
        try:
            img = Image.open(path)
            img_np = np.array(img.convert("RGB"))
            white_pixels = np.sum((img_np[:,:,0] > 210) & (img_np[:,:,1] > 210) & (img_np[:,:,2] > 210))
            total_pixels = img_np.shape[0] * img_np.shape[1]
            white_ratio = white_pixels / total_pixels
            return white_ratio > 0.8
        except:
            return False
    
    try:
        image_indices = {}
        for f in image_files:
            try:
                patch_num = int(f.split("_")[-1].split(".")[0])
                image_indices[patch_num] = f
            except:
                continue
                
        if image_indices:
            sorted_indices = sorted(image_indices.keys())
            
            for i, patch_num in enumerate(sorted_indices):
                current_path = os.path.join(images_test_path, image_indices[patch_num])
                
                if is_mostly_white_file(current_path):
                    continue

                preceding_white = True
                if i >= 3:
                    for j in range(i-3, i):
                        prev_patch_num = sorted_indices[j]
                        prev_path = os.path.join(images_test_path, image_indices[prev_patch_num])
                        if not is_mostly_white_file(prev_path):
                            preceding_white = False
                            break
                else:
                    preceding_white = False
                    
                succeeding_white = True
                if i + 3 < len(sorted_indices):
                    for j in range(i+1, i+4):
                        next_patch_num = sorted_indices[j]
                        next_path = os.path.join(images_test_path, image_indices[next_patch_num])
                        if not is_mostly_white_file(next_path):
                            succeeding_white = False
                            break
                else:
                    succeeding_white = False
                    
                if preceding_white or succeeding_white:
                    images_to_convert.add(current_path)
                    
    except Exception as e:
        print(f"Error in pre-computing conversion list: {e}")

    with torch.no_grad():
        for image_path in image_paths:
            try:
                print(image_path)
                original_image = Image.open(image_path)
                original_size = original_image.size

                base_name = os.path.basename(image_path)
                file_name = f"mask_{base_name}"
                if not file_name.endswith('.tiff'):
                    file_name = file_name.rsplit('.', 1)[0] + '.tiff'
                output_path = os.path.join(output_masks_path, file_name)

                if is_mostly_white(original_image):
                    white_image = Image.new("RGB", original_size, color=(255, 255, 255))
                    white_image.save(output_path, format="TIFF")
                else:
                    convert_blue_green = image_path in images_to_convert
                    
                    image = original_image.convert("RGB")
                    image_tensor = test_transform(image).unsqueeze(0).to(device)
                    output = model(image_tensor)
                    pred_mask = torch.argmax(output, dim=1).squeeze().cpu().numpy()

                    pred_mask = modify_mask(pred_mask, convert_blue_green_to_white=convert_blue_green)
                    save_colored_masks(pred_mask, original_size, output_masks_path, file_name)

            except Exception as e:
                print(f"Error processing {image_path}: {e}")
                continue

    mask_files = [f for f in os.listdir(output_masks_path) if f.endswith(".tiff")]
    for f in mask_files:
        path = os.path.join(output_masks_path, f)
        try:
            img = Image.open(path).convert("RGB")
            arr = np.array(img)
            arr[(arr == [0, 0, 0]).all(axis=-1)] = [255, 255, 255]
            Image.fromarray(arr).save(path, format="TIFF")
        except Exception as e:
            print(f"Error converting black to white in {f}: {e}")

    print("Post-processing complete.")

if __name__ == "__main__":
    main()
