import torch
import torch.nn as nn
import torchvision.transforms as T
from torchvision.models import resnet18
import cv2
import numpy as np

class AppearanceEncoder:
    def __init__(self, device="cpu"):
        self.device = device

        model = resnet18(pretrained=True)
        model.fc = nn.Identity()  # remove classifier
        self.model = model.to(device).eval()

        self.transform = T.Compose([
            T.ToPILImage(),
            T.Resize((128, 64)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
        ])

    def extract(self, frame, boxes):
        features = []

        for box in boxes:
            x1, y1, x2, y2 = map(int, box)
            crop = frame[y1:y2, x1:x2]

            if crop.size == 0:
                features.append(np.zeros(512))
                continue

            crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            tensor = self.transform(crop).unsqueeze(0).to(self.device)

            with torch.no_grad():
                feat = self.model(tensor).cpu().numpy().flatten()

            feat = feat / (np.linalg.norm(feat) + 1e-6)
            features.append(feat)

        return np.array(features)