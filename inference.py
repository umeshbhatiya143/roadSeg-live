import os
import cv2
import torch
import torch.nn as nn
from torchvision import transforms
import numpy as np
from PIL import Image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "unet_road_seg2.pth")

img_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

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

# Load model
model = UNet().to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

def stream_inference(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # segmentation preprocessing
        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        img_tensor = img_transform(img).unsqueeze(0).to(device)

        with torch.no_grad():
            pred = model(img_tensor)
            mask = (pred[0, 0].cpu().numpy() > 0.5).astype("uint8") * 255

        mask_resized = cv2.resize(mask, (frame.shape[1], frame.shape[0]))
        overlay = frame.copy()
        overlay[mask_resized == 255] = (0, 255, 0)
        blended = cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)

        # encode JPEG
        ret, buffer = cv2.imencode(".jpg", blended)
        if not ret:
            continue

        frame_bytes = buffer.tobytes()
        yield (b"--frame\r\n"
               b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n")

    cap.release()



# import cv2, torch, os, base64
# from PIL import Image
# from io import BytesIO
# import numpy as np
# from torchvision import transforms

# class UNet(nn.Module):
#     def __init__(self, n_classes=1):
#         super(UNet, self).__init__()
#         def CBR(in_channels, out_channels):
#             return nn.Sequential(
#                 nn.Conv2d(in_channels, out_channels, 3, padding=1),
#                 nn.BatchNorm2d(out_channels),
#                 nn.ReLU(inplace=True),
#                 nn.Conv2d(out_channels, out_channels, 3, padding=1),
#                 nn.BatchNorm2d(out_channels),
#                 nn.ReLU(inplace=True),
#             )
#         self.enc1 = CBR(3, 64)
#         self.enc2 = CBR(64, 128)
#         self.enc3 = CBR(128, 256)
#         self.enc4 = CBR(256, 512)
#         self.pool = nn.MaxPool2d(2)
#         self.bottleneck = CBR(512, 1024)
#         self.up4 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
#         self.dec4 = CBR(1024, 512)
#         self.up3 = nn.ConvTranspose2d(512, 256, 2, stride=2)
#         self.dec3 = CBR(512, 256)
#         self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
#         self.dec2 = CBR(256, 128)
#         self.up1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
#         self.dec1 = CBR(128, 64)
#         self.out_conv = nn.Conv2d(64, n_classes, 1)

#     def forward(self, x):
#         e1 = self.enc1(x)
#         e2 = self.enc2(self.pool(e1))
#         e3 = self.enc3(self.pool(e2))
#         e4 = self.enc4(self.pool(e3))
#         b = self.bottleneck(self.pool(e4))
#         d4 = self.dec4(torch.cat([self.up4(b), e4], dim=1))
#         d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))
#         d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
#         d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
#         return torch.sigmoid(self.out_conv(d1))


# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# MODEL_PATH = "models/unet_road_seg.pth"
# img_transform = transforms.Compose([
#     transforms.Resize((256, 256)),
#     transforms.ToTensor(),
#     transforms.Normalize([0.5]*3, [0.5]*3)
# ])

# model = UNet().to(device)
# model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
# model.eval()

# def generate_frames(video_path):
#     cap = cv2.VideoCapture(video_path)
#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             break
#         img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
#         img_tensor = img_transform(img).unsqueeze(0).to(device)
#         with torch.no_grad():
#             pred = model(img_tensor)
#             mask = (pred[0,0].cpu().numpy() > 0.5).astype(np.uint8) * 255
#         mask_resized = cv2.resize(mask, (frame.shape[1], frame.shape[0]))
#         overlay = frame.copy()
#         overlay[mask_resized==255] = (0,255,0)
#         blended = cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)
#         _, buffer = cv2.imencode(".jpg", blended)
#         frame_base64 = base64.b64encode(buffer).decode("utf-8")
#         yield frame_base64
#     cap.release()

