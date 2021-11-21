import os

import numpy as np
from cv2 import cv2
import pandas as pd
import pydicom
import torch
import torchvision
from PIL import Image
from torch import nn
from torchvision import transforms
import torch.nn.functional as F

# Define transforms
test_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
          ])

class efficientnetB0(nn.Module):
    def __init__(self, num_classes=3, num_channels=3):
        super().__init__()
        self.efficientnet = torchvision.models.efficientnet_b0(pretrained=True)
        self.classifier = nn.Sequential(
            nn.Linear(1000, 512),
            nn.BatchNorm1d(512),
            nn.Dropout(0.5),
            nn.Linear(512 , 256),
            nn.Linear(256 , 3)
        )
        
    def forward(self, x):
        out = self.efficientnet(x)
        out = self.classifier(out)
        return out

model = efficientnetB0()
model.load_state_dict(torch.load('best_model.pth'))
model = model.to('cuda')

# Generate submission
submission = pd.read_csv('sample_submission.csv', index_col='FileID')
root = 'data/data/valid'
prediction_dict = {0:'Negative', 1:'Typical', 2:'Atypical'}
idx = 0
model.eval()
for folder in os.listdir(root):
    for subfolder in os.listdir(os.path.join(root, folder)):
        fn = os.listdir(os.path.join(root, folder, subfolder))[0]
        img_path = os.path.join(root, folder, subfolder, fn)
        # print(img_path)
        ds = pydicom.dcmread(img_path)
        img = ds.pixel_array
        img = img / np.max(img)
        img = (255*img).clip(0, 255).astype(np.uint8)
        eq_img = cv2.equalizeHist(img)
        img = np.stack([eq_img] * 3, axis=2)
        # img = np.stack([img] * 3, axis=2)
        img = test_transform(img)
        img = torch.unsqueeze(img, 0)
        img = img.to('cuda')
        output = model(img)
        _, predictions = torch.max(output.data, 1)
        submission.loc[fn.split('.')[0], 'Type'] = prediction_dict[predictions.item()]
        idx += 1
        # if idx > 10: 
        #     break
print(idx)
print(submission.head)
submission.to_csv('effnetB0_eqdrop_predictions.csv')


