import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split, SubsetRandomSampler
from torchvision import transforms
import torchvision.transforms.functional as TF
from modellobule import BinaryUNet
from PIL import Image
import numpy as np
import random
import torch.multiprocessing as mp
from sklearn.model_selection import KFold
import time
import datetime

class LobuleDataset(Dataset):
    def __init__(self, image_paths, mask_paths, transform=None, augment=True):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.transform = transform
        self.augment = augment
        self.yellow_rgb = (255, 255, 0)

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
        binary_mask = np.zeros((mask.shape[0], mask.shape[1]), dtype=np.float32)

        yellow_pixels = np.logical_and(
            mask[:,:,0] == 255,  # Red channel = 255
            np.logical_and(
                mask[:,:,1] == 255,  # Green channel = 255
                mask[:,:,2] == 0     # Blue channel = 0
            )
        )
        binary_mask[yellow_pixels] = 1.0
        
        image = transforms.ToTensor()(image)
        mask = torch.tensor(binary_mask, dtype=torch.float32).unsqueeze(0)  # Add channel dimension
        
        if self.transform:
            image = self.transform(image)
        
        return image, mask

def create_kfold_loaders(dataset, k=5, batch_size=1, num_workers=16):
    kfold = KFold(n_splits=k, shuffle=True, random_state=42)
    
    fold_loaders = []
    
    for fold, (train_ids, val_ids) in enumerate(kfold.split(range(len(dataset)))):
        train_sampler = SubsetRandomSampler(train_ids)
        val_sampler = SubsetRandomSampler(val_ids)
        
        train_loader = DataLoader(
            dataset, 
            batch_size=batch_size,
            sampler=train_sampler,
            num_workers=num_workers,
            pin_memory=True
        )
        
        val_loader = DataLoader(
            dataset, 
            batch_size=batch_size,
            sampler=val_sampler,
            num_workers=num_workers,
            pin_memory=True
        )
        
        fold_loaders.append((train_loader, val_loader))
    
    return fold_loaders

def binary_dice_loss(pred, target, smooth=1e-5):
    pred = torch.sigmoid(pred)
    
    pred_flat = pred.view(-1)
    target_flat = target.view(-1)
    
    intersection = (pred_flat * target_flat).sum()
    union = pred_flat.sum() + target_flat.sum()
    
    dice = (2.0 * intersection + smooth) / (union + smooth)
    
    return 1.0 - dice

def binary_ce_loss(pred, target):
    return F.binary_cross_entropy_with_logits(pred, target)

def combined_binary_loss(pred, target):
    bce = binary_ce_loss(pred, target)
    dice = binary_dice_loss(pred, target)
    return 0.5 * bce + 0.5 * dice

def iou_score(pred, target, threshold=0.5, smooth=1e-5):
    pred = (torch.sigmoid(pred) > threshold).float()
    
    pred_flat = pred.view(-1)
    target_flat = target.view(-1)
    
    intersection = (pred_flat * target_flat).sum()
    union = pred_flat.sum() + target_flat.sum() - intersection
    
    iou = (intersection + smooth) / (union + smooth)
    
    return iou

def train_fold(model, train_loader, val_loader, fold, device, epochs=100, lr=1e-4):
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='max',          # Monitor validation IoU (higher is better)
        factor=0.5,          # Reduce LR by half when plateau
        patience=10,         # Wait 10 epochs before reducing LR
        verbose=True,
        min_lr=1e-6
    )
    
    model.to(device)
    best_val_iou = 0.0
    
    patience = 20
    patience_counter = 0
    
    fold_results = {
        'train_losses': [],
        'val_losses': [],
        'val_ious': [],
        'best_val_iou': 0.0,
        'best_epoch': 0
    }
    
    print(f"\n{'='*50}")
    print(f"STARTING FOLD {fold+1} TRAINING")
    print(f"{'='*50}\n")
    
    for epoch in range(epochs):
        epoch_start_time = time.time()
        
        model.train()
        running_loss = 0.0
        batch_count = 0
        
        for images, masks in train_loader:
            images, masks = images.to(device), masks.to(device)
            
            optimizer.zero_grad()
            
            outputs = model(images)
            
            loss = combined_binary_loss(outputs, masks)
            
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            running_loss += loss.item()
            batch_count += 1
        
        train_loss = running_loss / batch_count
        fold_results['train_losses'].append(train_loss)
        
        model.eval()
        val_running_loss = 0.0
        val_batch_count = 0
        val_iou_scores = []
        
        with torch.no_grad():
            for images, masks in val_loader:
                images, masks = images.to(device), masks.to(device)
                outputs = model(images)
                val_loss = combined_binary_loss(outputs, masks)
                val_running_loss += val_loss.item()
                val_batch_count += 1
                
                batch_iou = iou_score(outputs, masks)
                val_iou_scores.append(batch_iou.item())
        
        val_loss = val_running_loss / val_batch_count
        val_iou = sum(val_iou_scores) / len(val_iou_scores)

        fold_results['val_losses'].append(val_loss)
        fold_results['val_ious'].append(val_iou)

        scheduler.step(val_iou)

        epoch_duration = time.time() - epoch_start_time
        epoch_mins = int(epoch_duration // 60)
        epoch_secs = int(epoch_duration % 60)

        print(f"Fold {fold+1}, Epoch {epoch+1}/{epochs} - Time: {epoch_mins}m {epoch_secs}s")
        print(f"Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}, Val IoU={val_iou:.4f}")
        print(f"Current LR: {optimizer.param_groups[0]['lr']:.2e}")

        if val_iou > best_val_iou:
            best_val_iou = val_iou
            fold_results['best_val_iou'] = val_iou
            fold_results['best_epoch'] = epoch + 1

            if isinstance(model, nn.DataParallel):
                model_state_dict = model.module.state_dict()
            else:
                model_state_dict = model.state_dict()
                
            torch.save(model_state_dict, f'best_binary_model_fold_{fold+1}.pth')
            print(f"New best model saved! IoU: {val_iou:.4f}")
            patience_counter = 0
        else:
            patience_counter += 1

        if (epoch + 1) % 10 == 0:
            if isinstance(model, nn.DataParallel):
                model_state_dict = model.module.state_dict()
            else:
                model_state_dict = model.state_dict()
                
            torch.save({
                'epoch': epoch,
                'model_state_dict': model_state_dict,
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'val_iou': val_iou
            }, f'binary_checkpoint_fold_{fold+1}_epoch_{epoch+1}.pth')

        if patience_counter >= patience:
            print(f"Early stopping triggered after {epoch+1} epochs")
            break
    
    return fold_results

def main():
    # Data directory - update this to your lobule data path
    data_dir = r'/work/xi47luy'
    batch_size = 1  # Adjust based on your GPU memory
    epochs = 100
    lr = 1e-4
    num_folds = 5
    
    if not torch.cuda.is_available():
        print("No GPU available, using CPU")
        device = torch.device("cpu")
    else:
        device = torch.device("cuda")
        print(f"Using device: {device}")
        print(f"Number of GPUs available: {torch.cuda.device_count()}")
    
    torch.backends.cudnn.benchmark = True

    transform = transforms.Compose([
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])

    output_dir = "./k_fold_results"
    os.makedirs(output_dir, exist_ok=True)

    image_paths = sorted([os.path.join(data_dir, "lobulepatches", f) for f in os.listdir(os.path.join(data_dir, "lobulepatches"))])
    mask_paths = sorted([os.path.join(data_dir, "lobulemasks", f) for f in os.listdir(os.path.join(data_dir, "lobulemasks"))])
    
    dataset = LobuleDataset(image_paths, mask_paths, transform=transform)

    fold_loaders = create_kfold_loaders(dataset, k=num_folds, batch_size=batch_size)

    all_fold_results = []

    start_time = time.time()

    for fold, (train_loader, val_loader) in enumerate(fold_loaders):
        print(f"\n{'='*50}")
        print(f"FOLD {fold+1}/{num_folds}")
        print(f"{'='*50}")
        print(f"Training samples: {len(train_loader.sampler)}")
        print(f"Validation samples: {len(val_loader.sampler)}")

        model = BinaryUNet(n_channels=3, base_channels=64)
        
        def init_weights(m):
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        
        model.apply(init_weights)

        model.to(device)

        if torch.cuda.device_count() > 1:
            print(f"Using {torch.cuda.device_count()} GPUs for training")
            model = nn.DataParallel(model)

        fold_result = train_fold(model, train_loader, val_loader, fold, device, epochs, lr)
        all_fold_results.append(fold_result)

        torch.save(fold_result, os.path.join(output_dir, f'fold_{fold+1}_results.pth'))

    all_best_ious = [result['best_val_iou'] for result in all_fold_results]
    avg_best_iou = sum(all_best_ious) / len(all_best_ious)
    std_best_iou = np.std(all_best_ious)

    total_time = time.time() - start_time
    hours = int(total_time // 3600)
    minutes = int((total_time % 3600) // 60)
    seconds = int(total_time % 60)

    print("\n" + "="*50)
    print("CROSS-VALIDATION RESULTS")
    print("="*50)
    print(f"Total training time: {hours}h {minutes}m {seconds}s")
    print(f"Average best IoU: {avg_best_iou:.4f} ± {std_best_iou:.4f}")
    print("Individual fold best IoUs:")
    
    for fold, result in enumerate(all_fold_results):
        print(f"Fold {fold+1}: {result['best_val_iou']:.4f} (epoch {result['best_epoch']})")

    summary = {
        'fold_best_ious': all_best_ious,
        'avg_best_iou': avg_best_iou,
        'std_best_iou': std_best_iou,
        'total_training_time': total_time,
        'timestamp': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    
    torch.save(summary, os.path.join(output_dir, 'cross_validation_summary.pth'))

    print("\n" + "="*50)
    print("TRAINING FINAL MODEL ON ALL DATA")
    print("="*50)

    full_train_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=16,
        pin_memory=True
    )

    final_model = BinaryUNet(n_channels=3, base_channels=64)
    final_model.apply(init_weights)

    final_model.to(device)
    if torch.cuda.device_count() > 1:
        final_model = nn.DataParallel(final_model)

    optimizer = optim.Adam(final_model.parameters(), lr=lr, weight_decay=1e-5)
    
    final_model.train()
    for epoch in range(30):  # Train for 30 epochs (adjust as needed)
        running_loss = 0.0
        batch_count = 0
        
        for images, masks in full_train_loader:
            images, masks = images.to(device), masks.to(device)
            
            optimizer.zero_grad()
            outputs = final_model(images)
            loss = combined_binary_loss(outputs, masks)
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(final_model.parameters(), max_norm=1.0)
            optimizer.step()
            
            running_loss += loss.item()
            batch_count += 1
        
        print(f"Final Model - Epoch {epoch+1}/30, Loss: {running_loss/batch_count:.4f}")

    if isinstance(final_model, nn.DataParallel):
        model_state_dict = final_model.module.state_dict()
    else:
        model_state_dict = final_model.state_dict()
        
    torch.save({
        'model_state_dict': model_state_dict,
        'architecture': 'BinaryUNet_v1',
        'cross_validation_results': summary
    }, os.path.join(output_dir, "binary_unet_lobule_segmentation_final.pth"))
    
    print("\nK-fold cross-validation and final model training completed!")

if __name__ == "__main__":
    main()