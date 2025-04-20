import sys
import os
import traceback

sys.stdout.reconfigure(line_buffering=True)

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.optim as optim
    from torch.utils.data import DataLoader, Dataset, random_split
    from torchvision import transforms
    import torchvision.transforms.functional as TF
    from Unet_seb_att import UNet
    from PIL import Image
    import numpy as np
    import random
    from sklearn.model_selection import KFold
    import matplotlib.pyplot as plt
    from tqdm import tqdm
    import warnings
    import seaborn as sns
    from typing import Dict, List, Tuple, Optional
    
except Exception as e:
    traceback.print_exc()
    sys.exit(1)

class LiverDataset(Dataset):
    def __init__(self, image_paths, mask_paths, transform=None, augment=True):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.transform = transform
        self.augment = augment
        self.rgb_to_class = {
            (0, 0, 0): 0,        
            (0, 0, 255): 1,      
            (0, 255, 0): 2,      
            (255, 0, 0): 3,      
            (128, 0, 128): 4     
        }

    def __len__(self):
        return len(self.image_paths)

    def apply_augmentation(self, image, mask):
        if random.random() > 0.5:
            image = TF.hflip(image)
            mask = TF.hflip(mask)
        
        if random.random() > 0.5:
            image = TF.vflip(image)
            mask = TF.vflip(mask)
        
        if random.random() > 0.5:
            angle = random.randint(-30, 30)
            image = TF.rotate(image, angle, fill=0)
            mask = TF.rotate(mask, angle, fill=0)
        
        return image, mask

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert("RGB")
        mask = Image.open(self.mask_paths[idx]).convert("RGB")
        
        if self.augment:
            image, mask = self.apply_augmentation(image, mask)
        
        mask = np.array(mask)
        mask_class = np.zeros((mask.shape[0], mask.shape[1]), dtype=np.uint8)
        for rgb, class_idx in self.rgb_to_class.items():
            mask_class[(mask == rgb).all(axis=-1)] = class_idx
        
        image = transforms.ToTensor()(image)
        mask = torch.tensor(mask_class, dtype=torch.long)
        
        if self.transform:
            image = self.transform(image)
        
        return image, mask

def load_data(data_dir, batch_size, transform=None, val_split=0.2):
    image_paths = sorted([os.path.join(data_dir, "PSR_image", f) for f in os.listdir(os.path.join(data_dir, "PSR_image"))])
    mask_paths = sorted([os.path.join(data_dir, "PSR_mask", f) for f in os.listdir(os.path.join(data_dir, "PSR_mask"))])
    dataset = LiverDataset(image_paths, mask_paths, transform=transform)
    val_size = int(len(dataset) * val_split)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=16)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=16)
    return train_loader, val_loader

def dice_loss(pred, target, class_weights=None, smooth=1e-5):
    pred = F.softmax(pred, dim=1)
    target_one_hot = F.one_hot(target, num_classes=pred.shape[1]).permute(0, 3, 1, 2)
    
    intersection = torch.sum(pred * target_one_hot, dim=(2, 3)) + smooth
    union = torch.sum(pred, dim=(2, 3)) + torch.sum(target_one_hot, dim=(2, 3)) + smooth
    
    device = pred.device
    if class_weights is None:
        class_weights = torch.tensor([0.1, 1.5, 2.5, 4.5, 4.5], device=device).view(1, -1)
    elif not class_weights.dim() == 2:
        class_weights = class_weights.view(1, -1)
    
    dice = torch.clamp(2.0 * intersection / union, min=0.0, max=1.0)
    weighted_dice = dice * class_weights
    return 1.0 - torch.mean(weighted_dice)

def focal_loss(pred, target, class_weights=None, alpha=0.25, gamma=2.0, smooth=1e-5):
    pred = F.softmax(pred, dim=1)
    target_one_hot = F.one_hot(target, num_classes=pred.shape[1]).permute(0, 3, 1, 2)
    
    pred = torch.clamp(pred, min=smooth, max=1.0 - smooth)
    
    device = pred.device
    
    if class_weights is None:
        gamma_values = torch.tensor([1.0, 2.0, 2.5, 3.5, 3.5], device=device).view(1, -1, 1, 1)
        alpha_values = torch.tensor([0.1, 0.6, 0.7, 0.9, 0.9], device=device).view(1, -1, 1, 1)
    else:
        base_gamma = torch.tensor([1.0, 2.0, 2.5, 3.0, 3.0], device=device)
        base_alpha = torch.tensor([0.1, 0.6, 0.7, 0.8, 0.8], device=device)
        
        norm_weights = class_weights.clone()
        if norm_weights.dim() == 1:
            norm_weights = norm_weights / torch.max(norm_weights)
            gamma_values = (base_gamma * (1 + 0.5 * norm_weights)).view(1, -1, 1, 1)
            alpha_values = (base_alpha * (1 + 0.2 * norm_weights)).view(1, -1, 1, 1)
        else:
            norm_weights = norm_weights / torch.max(norm_weights)
            gamma_values = (base_gamma * (1 + 0.5 * norm_weights.view(-1))).view(1, -1, 1, 1)
            alpha_values = (base_alpha * (1 + 0.2 * norm_weights.view(-1))).view(1, -1, 1, 1)
    
    focal = -alpha_values * ((1 - pred) ** gamma_values) * target_one_hot * torch.log(pred)
    return torch.mean(focal)

def ce_loss(pred, target, class_weights=None):
    device = pred.device
    if class_weights is None:
        class_weights = torch.tensor([0.1, 1.5, 2.5, 4.5, 4.5], device=device)
    
    return F.cross_entropy(pred, target, weight=class_weights)

def combined_loss(pred, target, class_weights=None):
    device = pred.device
    if class_weights is None:
        class_weights = torch.tensor([0.1, 1.5, 2.5, 4.5, 4.5], device=device)
    
    ce = ce_loss(pred, target, class_weights)
    dice = dice_loss(pred, target, class_weights)
    focal = focal_loss(pred, target, class_weights)
    
    return 0.2*ce + 0.6*dice + 0.2*focal

def dice_coefficient_per_class(pred, target, n_classes):
    if pred.dim() == 4:
        pred = torch.argmax(torch.softmax(pred, dim=1), dim=1)
    
    dice_scores = {}
    for class_idx in range(n_classes):
        pred_mask = (pred == class_idx)
        target_mask = (target == class_idx)
        
        intersection = torch.sum(pred_mask * target_mask)
        union = torch.sum(pred_mask) + torch.sum(target_mask)
        
        if union == 0:
            dice_scores[class_idx] = torch.tensor(1.0, device=pred.device)
        else:
            dice_scores[class_idx] = (2.0 * intersection) / (union + 1e-8)
    
    return dice_scores

def post_process_predictions(prediction, ground_truth):
    processed_pred = prediction.clone()
    
    if torch.any(prediction == 3) or torch.any(prediction == 4):
        processed_pred[prediction == 2] = 1
    
    return processed_pred

def calculate_metrics_per_class(pred, target, n_classes):
    pred = torch.argmax(torch.softmax(pred, dim=1), dim=1)
    
    dice_scores = {}
    tpr_scores = {}
    iou_scores = {}
    
    for c in range(n_classes):
        pred_c = (pred == c).float()
        target_c = (target == c).float()
        
        intersection = (pred_c * target_c).sum()
        union = pred_c.sum() + target_c.sum() - intersection
        
        if union > 0:
            dice_scores[c] = (2 * intersection / (pred_c.sum() + target_c.sum())).item()
            iou_scores[c] = (intersection / union).item()
        else:
            dice_scores[c] = 1.0
            iou_scores[c] = 1.0
        
        true_positives = ((pred_c == 1) & (target_c == 1)).sum().item()
        total_positives = (target_c == 1).sum().item()
        if total_positives > 0:
            tpr_scores[c] = true_positives / total_positives
        else:
            tpr_scores[c] = 1.0
    
    return dice_scores, tpr_scores, iou_scores

def train(model, train_loader, val_loader, device, epochs=250, lr=5e-4, n_classes=5, fold=None):
    optimizer = optim.AdamW(model.parameters(), lr=lr, eps=1e-7, weight_decay=2e-4, betas=(0.9, 0.999))
    
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=lr,            
        total_steps=epochs,
        pct_start=0.3,        
        div_factor=10.0,      
        final_div_factor=50,  
        anneal_strategy='cos',
    )
    
    model.to(device)
    best_mean_vessel_dice = 0.0
    best_model = None
    best_epoch = 0
    scaler = torch.amp.GradScaler('cuda')
    
    base_weights = torch.tensor([0.1, 1.5, 2.5, 4.5, 4.5], device=device)
    current_class_weights = base_weights.clone()
    
    max_weight_caps = torch.tensor([0.1, 8.0, 10.0, 15.0, 15.0], device=device)
    
    class_performances = {i: [] for i in range(1, n_classes)}
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        batch_count = 0
        
        if epoch % 10 == 0 and epoch > 0:
            avg_performances = {}
            for class_idx in range(1, n_classes):
                if class_performances[class_idx]:
                    avg_performance = sum(class_performances[class_idx]) / len(class_performances[class_idx])
                    avg_performances[class_idx] = avg_performance
                else:
                    avg_performances[class_idx] = 0.0
            
            for class_idx in range(1, n_classes):
                class_score = avg_performances.get(class_idx, 0)
                
                if class_score < 0.2:
                    boost_factor = 1.3
                elif class_score < 0.4:
                    boost_factor = 1.2
                elif class_score < 0.6:
                    boost_factor = 1.1
                else:
                    boost_factor = 1.0
                
                if boost_factor > 1.0:
                    old_weight = current_class_weights[class_idx].item()
                    current_class_weights[class_idx] *= boost_factor
                    
                    current_class_weights[class_idx] = torch.min(
                        current_class_weights[class_idx], 
                        max_weight_caps[class_idx]
                    )
    
        for images, masks in train_loader:
            images = images.to(device)
            masks = masks.to(device)
            
            optimizer.zero_grad()
            
            with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                outputs = model(images)
                loss = combined_loss(outputs, masks, current_class_weights)
            
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            
            running_loss += loss.item()
            batch_count += 1
        
        avg_train_loss = running_loss / batch_count
        
        model.eval()
        val_loss = 0.0
        val_batch_count = 0
        
        val_dice_scores = {i: [] for i in range(n_classes)}
        val_tpr_scores = {i: [] for i in range(n_classes)}
        val_iou_scores = {i: [] for i in range(n_classes)}
        
        with torch.no_grad():
            for val_images, val_masks in val_loader:
                val_images = val_images.to(device)
                val_masks = val_masks.to(device)
                
                val_outputs = model(val_images)
                val_loss += combined_loss(val_outputs, val_masks, current_class_weights).item()
                val_batch_count += 1
                
                dice_scores, tpr_scores, iou_scores = calculate_metrics_per_class(val_outputs, val_masks, n_classes)
                
                for c in range(n_classes):
                    val_dice_scores[c].append(dice_scores[c])
                    val_tpr_scores[c].append(tpr_scores[c])
                    val_iou_scores[c].append(iou_scores[c])
        
        avg_val_loss = val_loss / val_batch_count
        
        mean_dice_scores = {c: np.mean(val_dice_scores[c]) for c in range(n_classes)}
        mean_tpr_scores = {c: np.mean(val_tpr_scores[c]) for c in range(n_classes)}
        mean_iou_scores = {c: np.mean(val_iou_scores[c]) for c in range(n_classes)}
        
        mean_vessel_dice = np.mean([mean_dice_scores[c] for c in range(1, n_classes)])
        
        for c in range(1, n_classes):
            class_performances[c].append(mean_dice_scores[c])
        
        scheduler.step()
        
        if mean_vessel_dice > best_mean_vessel_dice:
            best_mean_vessel_dice = mean_vessel_dice
            best_model = model.state_dict().copy()
            best_epoch = epoch + 1
            
    return best_model, best_epoch

def split_test_data(image_paths, mask_paths, test_ratio=0.1, random_seed=42):
    total_samples = len(image_paths)
    test_size = int(total_samples * test_ratio)
    
    indices = list(range(total_samples))
    random.seed(random_seed)
    random.shuffle(indices)
    
    test_indices = indices[:test_size]
    train_indices = indices[test_size:]
    
    train_images = [image_paths[i] for i in train_indices]
    train_masks = [mask_paths[i] for i in train_indices]
    test_images = [image_paths[i] for i in test_indices]
    test_masks = [mask_paths[i] for i in test_indices]
    
    return train_images, train_masks, test_images, test_masks

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    data_dir = '/work/xi47luy'
    
    image_paths = sorted([os.path.join(data_dir, "PSR_image", f) for f in os.listdir(os.path.join(data_dir, "PSR_image"))])
    mask_paths = sorted([os.path.join(data_dir, "PSR_mask", f) for f in os.listdir(os.path.join(data_dir, "PSR_mask"))])
    
    train_images, train_masks, test_images, test_masks = split_test_data(image_paths, mask_paths, test_ratio=0.1)
    
    # Save test set filenames to testset.txt
    with open('testset.txt', 'w') as f:
        f.write(f"# Total training samples: {len(train_images)}\n")
        f.write(f"# Total test samples: {len(test_images)}\n")
        f.write("# Test images:\n")
        for img_path in test_images:
            f.write(f"{os.path.basename(img_path)}\n")
    
    # Run k-fold CV
    k_fold_cross_validation(
        train_images=train_images,
        train_masks=train_masks,
        device=device,
        n_splits=5,
        batch_size=8,
        epochs=250,
        lr=5e-4
    )

def evaluate_model_with_metrics(model, data_loader, device):
    model.eval()
    dice_scores = {i: [] for i in range(5)}
    tpr_scores = {i: [] for i in range(5)}
    iou_scores = {i: [] for i in range(5)}
    
    with torch.no_grad():
        for images, masks in data_loader:
            images = images.to(device)
            masks = masks.to(device)
            
            outputs = model(images)
            pred = torch.argmax(torch.softmax(outputs, dim=1), dim=1)
            
            for i in range(pred.size(0)):
                pred[i] = post_process_predictions(pred[i], masks[i])
            
            for c in range(5):
                pred_c = (pred == c).float()
                target_c = (masks == c).float()
                
                intersection = (pred_c * target_c).sum()
                union = pred_c.sum() + target_c.sum() - intersection
                
                if union > 0:
                    dice_scores[c].append((2 * intersection / (pred_c.sum() + target_c.sum())).item())
                    iou_scores[c].append((intersection / union).item())
                else:
                    dice_scores[c].append(1.0)
                    iou_scores[c].append(1.0)
                
                true_positives = ((pred_c == 1) & (target_c == 1)).sum().item()
                total_positives = (target_c == 1).sum().item()
                if total_positives > 0:
                    tpr_scores[c].append(true_positives / total_positives)
                else:
                    tpr_scores[c].append(1.0)
    
    final_dice = {c: np.mean(scores) for c, scores in dice_scores.items()}
    final_tpr = {c: np.mean(scores) for c, scores in tpr_scores.items()}
    final_iou = {c: np.mean(scores) for c, scores in iou_scores.items()}
    
    return final_dice, final_tpr, final_iou

def k_fold_cross_validation(train_images, train_masks, device, n_splits=5, batch_size=4, epochs=250, lr=5e-4):
    dataset = LiverDataset(train_images, train_masks)
    
    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    fold_results = []
    
    for fold, (train_idx, val_idx) in enumerate(kfold.split(dataset)):
        print(f"\nTraining Fold {fold + 1}/{n_splits}")
        
        train_subsampler = torch.utils.data.SubsetRandomSampler(train_idx)
        val_subsampler = torch.utils.data.SubsetRandomSampler(val_idx)
        
        num_gpus = torch.cuda.device_count()
        effective_batch_size = batch_size * num_gpus if num_gpus > 0 else batch_size
        
        train_loader = DataLoader(dataset, batch_size=effective_batch_size, sampler=train_subsampler, num_workers=16)
        val_loader = DataLoader(dataset, batch_size=effective_batch_size, sampler=val_subsampler, num_workers=16)
        
        model = UNet(n_classes=5)
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
        model.to(device)
        
        best_model, best_epoch = train(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            epochs=epochs,
            lr=lr,
            n_classes=5,
            fold=fold
        )
        
        # Save the best model for this fold
        torch.save(best_model, f'best_model_{fold+1}.pth')
        
        # Load the best model and evaluate on validation set
        model.load_state_dict(best_model)
        model.eval()
        
        # Evaluate best model on validation set
        val_dice, val_tpr, val_iou = evaluate_model_with_metrics(model, val_loader, device)
        
        # Print metrics for the best model on the validation set
        print(f"\nFold {fold + 1} - Best Model Results:")
        print("\nClass          Dice     TPR      IoU")
        print("-------------------------------------------")
        mean_vessel_metrics = {"dice": 0, "tpr": 0, "iou": 0, "count": 0}
        
        for c in range(5):
            class_name = ["Background", "Portal Vein", "Central Vein", "Artery", "Bile Duct"][c]
            print(f"{class_name:<13} {val_dice[c]:.4f}    {val_tpr[c]:.4f}    {val_iou[c]:.4f}")
            
            if c > 0:  # If not background
                mean_vessel_metrics["dice"] += val_dice[c]
                mean_vessel_metrics["tpr"] += val_tpr[c]
                mean_vessel_metrics["iou"] += val_iou[c]
                mean_vessel_metrics["count"] += 1
        
        # Print mean vessel metrics
        print("\nMean Vessel Metrics:")
        print(f"Dice: {mean_vessel_metrics['dice'] / mean_vessel_metrics['count']:.4f}")
        print(f"TPR: {mean_vessel_metrics['tpr'] / mean_vessel_metrics['count']:.4f}")
        print(f"IoU: {mean_vessel_metrics['iou'] / mean_vessel_metrics['count']:.4f}")
        
        fold_results.append({
            'fold': fold + 1,
            'best_epoch': best_epoch,
            'dice': val_dice,
            'tpr': val_tpr,
            'iou': val_iou
        })
    
    return fold_results

if __name__ == '__main__':
    main()