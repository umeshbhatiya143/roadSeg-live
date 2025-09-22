# road_segmentation_unet.py

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms
from PIL import Image
import os
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import random


# ---------------------------
# 1. U-Net Model
# ---------------------------
class UNet(nn.Module):
    def __init__(self, n_classes=1):
        super(UNet, self).__init__()

        def CBR(in_channels, out_channels):
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, 3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            )

        self.enc1 = CBR(3, 64)
        self.enc2 = CBR(64, 128)
        self.enc3 = CBR(128, 256)
        self.enc4 = CBR(256, 512)

        self.pool = nn.MaxPool2d(2)
        self.bottleneck = CBR(512, 1024)

        self.up4 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.dec4 = CBR(1024, 512)
        self.up3 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec3 = CBR(512, 256)
        self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec2 = CBR(256, 128)
        self.up1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec1 = CBR(128, 64)

        self.out_conv = nn.Conv2d(64, n_classes, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))

        b = self.bottleneck(self.pool(e4))

        d4 = self.dec4(torch.cat([self.up4(b), e4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))

        return torch.sigmoid(self.out_conv(d1))


# ---------------------------
# 2. Dataset
# ---------------------------
class RoadDataset(Dataset):
    def __init__(self, img_dir, mask_dir, img_transform=None, mask_transform=None):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.img_transform = img_transform
        self.mask_transform = mask_transform

        # Ensure only matching image–mask pairs
        images = sorted([f for f in os.listdir(img_dir) if f.endswith(".jpg")])
        masks = sorted([f for f in os.listdir(mask_dir) if f.endswith(".png")])

        image_basenames = set(os.path.splitext(f)[0] for f in images)
        mask_basenames = set(os.path.splitext(f)[0] for f in masks)

        valid_basenames = sorted(list(image_basenames & mask_basenames))
        self.pairs = [(bn + ".jpg", bn + ".png") for bn in valid_basenames]

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        img_name, mask_name = self.pairs[idx]
        img_path = os.path.join(self.img_dir, img_name)
        mask_path = os.path.join(self.mask_dir, mask_name)

        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        if self.img_transform:
            image = self.img_transform(image)
        if self.mask_transform:
            mask = self.mask_transform(mask)
        else:
            mask = transforms.ToTensor()(mask)

        return image, mask


# ---------------------------
# 3. Transforms
# ---------------------------
img_transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

mask_transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
    transforms.Lambda(lambda x: (x > 0).float())
])


# ---------------------------
# 4. Main Function
# ---------------------------
def main():
    # Dataset
    dataset = RoadDataset(
        img_dir="/home/umesh/Desktop/Road_segmentation/dataset/d_images",
        mask_dir="/home/umesh/Desktop/Road_segmentation/dataset/d_mask",
        img_transform=img_transform,
        mask_transform=mask_transform
    )

    # Train/Test Split
    total_size = len(dataset)
    train_size = int(0.8 * total_size)
    test_size = total_size - train_size

    indices = list(range(total_size))
    random.seed(42)
    random.shuffle(indices)

    train_indices = indices[:train_size]
    test_indices = indices[train_size:]

    train_dataset = Subset(dataset, train_indices)
    test_dataset = Subset(dataset, test_indices)

    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)

    # Device, model, loss, optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet().to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    # Training loop
    for epoch in range(50):
        model.train()
        epoch_loss = 0
        for imgs, masks in train_loader:
            imgs, masks = imgs.to(device), masks.to(device)
            preds = model(imgs)
            loss = criterion(preds, masks)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        print(f"Epoch {epoch+1}, Train Loss: {epoch_loss/len(train_loader):.4f}")

    # Evaluation
    model.eval()
    all_preds, all_masks = [], []
    with torch.no_grad():
        for imgs, masks in test_loader:
            imgs, masks = imgs.to(device), masks.to(device)
            preds = model(imgs)
            preds = (preds > 0.5).float()
            all_preds.append(preds.cpu().numpy())
            all_masks.append(masks.cpu().numpy())

    all_preds = np.concatenate(all_preds).ravel()
    all_masks = np.concatenate(all_masks).ravel()

    accuracy = accuracy_score(all_masks, all_preds)
    precision = precision_score(all_masks, all_preds, zero_division=0)
    recall = recall_score(all_masks, all_preds, zero_division=0)
    f1 = f1_score(all_masks, all_preds, zero_division=0)

    print("\nTest Metrics:")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")

    # Visualization
    imgs, masks = next(iter(test_loader))
    imgs, masks = imgs.to(device), masks.to(device)
    with torch.no_grad():
        preds = model(imgs)
        preds = (preds > 0.5).float()

    plt.figure(figsize=(12,4))
    plt.subplot(1,3,1)
    plt.title("Original Image")
    plt.imshow((imgs[0].permute(1,2,0).cpu().numpy()*0.5 + 0.5))
    plt.axis("off")
    plt.subplot(1,3,2)
    plt.title("Ground Truth")
    plt.imshow(masks[0,0].cpu(), cmap="gray")
    plt.axis("off")
    plt.subplot(1,3,3)
    plt.title("Prediction")
    plt.imshow(preds[0,0].cpu(), cmap="gray")
    plt.axis("off")
    plt.show()

    # Save model
    torch.save(model.state_dict(), "unet_road_seg.pth")


if __name__ == "__main__":
    main()
