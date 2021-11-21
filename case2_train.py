import pydicom
import pandas as pd
import numpy as np
import random
import os
from tqdm import tqdm
from cv2 import cv2

import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from torch import nn, optim
import torch.nn.functional as F

from sklearn.metrics import accuracy_score

def same_seeds(seed):
    # Python built-in random module
    random.seed(seed)
    # Numpy
    np.random.seed(seed)
    # Torch
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

# Set random seed for reproducibility
same_seeds(123)

image_filenames = []
fn_path = '/media/caig/新增磁碟區/dataset/data/data/train'
def get_all_imgFilenames():
    for fn in os.listdir(fn_path):
      sub_path = os.path.join(fn_path, fn)
      sub_fn = os.path.join(sub_path, os.listdir(sub_path)[0])
      dcmFile = os.listdir(sub_fn)[0]
      image_filenames.append(os.path.join(sub_fn, dcmFile))

class pneumoniaDataset(Dataset):
    def __init__(self, files, transform=None):
        super(pneumoniaDataset, self).__init__()
        self.df = pd.read_csv('/media/caig/新增磁碟區/dataset/data/_info.csv', index_col='FileID')
        self.transform = transform
        self.imgFiles = files

    def __getitem__(self, index):
        # Load dicom image
        img_path = self.imgFiles[index]
        ds = pydicom.dcmread(img_path)
        img = ds.pixel_array
        img = img / np.max(img)
        img = (255*img).clip(0, 255).astype(np.uint8)
        eq_img = cv2.equalizeHist(img)
        img = np.stack([eq_img] * 3, axis=2)
        # img = np.stack([img] * 3, axis=2)
        # Load dicom image label
        filename = img_path.split('/')[-1].split('.')[0]
        if self.df.at[filename, 'Negative'] == 1:
          label = np.array([0])
        elif self.df.at[filename, 'Typical'] == 1:
          label = np.array([1])
        elif self.df.at[filename, 'Atypical'] == 1:
          label = np.array([2])
        label = torch.from_numpy(label).squeeze(-1)
        if self.transform is not None:
            img = self.transform(img)
     
        return img, label

    def __len__(self):
        return len(self.imgFiles)

get_all_imgFilenames()
train_set, val_set = torch.utils.data.random_split(image_filenames, [int(1200*0.8), int(1200*0.2)])


# Data agumentation
train_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ColorJitter(brightness=0.05, contrast=0.8, saturation=0.3),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            # transforms.RandomRotation(10),
            # transforms.RandomCrop((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
          ])
test_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
          ])

train_dataset = pneumoniaDataset(files=list(train_set), transform=train_transform)
val_dataset = pneumoniaDataset(files=list(val_set), transform=test_transform)
train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)
val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=0)


device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


class efficientnetB0(nn.Module):
    def __init__(self, num_classes=3, num_channels=3):
        super().__init__()
        self.efficientnet = torchvision.models.efficientnet_b0(pretrained=True)
        self.classifier = nn.Sequential(
            nn.Linear(1000 , 512),
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
model.to(device)

criterion = nn.CrossEntropyLoss()  #with softmax
# criterion = nn.NLLLoss()                  
optimizer = optim.Adam(model.parameters(), lr=0.001)
# optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
schedular = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=6, threshold=0.1)
# schedular = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.9, patience=3)

epochs = 30
val_acc_max = -np.Inf

for epoch in range(epochs):
    train_loss = 0.0
    val_loss = 0.0
    train_acc = 0.0
    val_acc = 0.0

    model.train()
    count = 0.0
    for idx, (images, labels) in enumerate(tqdm(train_dataloader)):
        optimizer.zero_grad()
        images = images.to(device)
        labels = labels.to(device)

        preds = model(images)
        _, predictions = torch.max(preds.data, 1)
        loss = criterion(preds, labels)
        train_loss += loss.item()
        train_acc += (labels == predictions).sum().item()
        count += predictions.size(0)

        loss.backward()
        optimizer.step()
    
    avg_train_loss = train_loss / count
    avg_train_acc = train_acc / count
 
    model.eval()
    count = 0
    with torch.no_grad():
        for val_idx, (val_images, val_labels) in enumerate(tqdm(val_dataloader)):
            val_images = val_images.to(device)
            val_labels = val_labels.to(device)

            val_preds = model(val_images)
            _, predictions = torch.max(val_preds.data, 1)
            loss = criterion(val_preds, val_labels)
            val_loss += loss.item()
            val_acc += (val_labels == predictions).sum().item()
            count += predictions.size(0)

        avg_val_loss = val_loss / count
        avg_val_acc = val_acc / count

    schedular.step(avg_val_acc)
    print("Epoch : {} \ntrain_loss : {:.4f}, \tTrain_acc : {:.4f}, \nVal_loss : {:.4f}, \tVal_acc : {:.4f}".format(epoch+1, avg_train_loss, avg_train_acc, avg_val_loss, avg_val_acc))

    if avg_val_acc >= val_acc_max:
        print('Validation acc increased from ({:.4f} --> {:.4f}).\nSaving model ...'.format(val_acc_max, avg_val_acc))
        torch.save(model.state_dict(), 'best_model.pth')
        val_acc_max = avg_val_acc

