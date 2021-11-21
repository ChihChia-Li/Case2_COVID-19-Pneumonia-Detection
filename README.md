# Case2_COVID-19-Pneumonia-Detection
## Digital Medicine 2021 

### Task Overview
In this case, we are required to detect COVID-19 pneumonia via chest x-ray images, which contain 400 negative, 400 typical pneumonia and 400 atypical pneumonia. It may be time-consuming and challenging for general radiologists in the community hospitals to read a high volume of chest X-ray images to detect subtle COVID-19 infected pneumonia and distinguish it from other community-acquired non-COVID-19 infected pneumonia. To address this challenge, developing a robust and accurate deep learning classifier model has been important. Thus, our task is to train a classfier that can detect the chest x-ray images with and without pneumonia, and whether the pneumonia is typical or atypical.

### Basic Requirements
* Python==3.8
* Pytorch==1.10.0
* torchvision==0.11.1
* cudatoolkit==10.2 
* pydicom==2.2.2
* pandas==1.3.4
* numpy==1.21.4
* tqdm==4.62.3
* opencv-python==4.5.4.58

### Reproducing Submission
To reproduce my result, do the following steps:
1. Dataset Preparation
2. Train from scratch or Download Pretrained models
3. Inference

### Dataset Preparation
* Traning and validation chest x-ray dicom images need to be placed in two different folders. You also need a ```data_info.csv``` file to get the labels of training images.
```
data
    +- data
    |  +- train
    |  |  +- 0a97737c6d00
    |  |  |  +- a578e647b295
    |  |  |     379473fc7c08.dcm
    |  |  +- 0b931420e93c
    |  |  |  +- 80a35c617e59
    |  |  |     a4694b0c94da.dcm
    |  |  +- ...
    |  +- valid
    |  |  +- 0a570ee06220
    |  |  |  +- 873d4c401124
    |  |  |     c82936b7514a.dcm
    |  |  +- 0b7bc6020f74
    |  |  |  +- 8456b947a678
    |  |  |     6fe5734d24c6.dcm
    |  |  +- ...
    -- data_info.csv
```

### Train from scratch
1. Preprocessing dicom images:
   * Convert a range of pixel values to 0-255.
   * Change dtype of image data to unit8 for subsequent image transformations (Since the dicom image dtype is float32).
   * Apply the histogram equalization on the grayscale image to enhance image constrast.
   * Stack the grayscale image after histogram equalization to three channel (Grayscale --> RGB).
   * Use same data augmentation (Since the size of training dataset is not large, data augmentation can reduce overfitting).
     ```
     # For training dataset 
     train_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ColorJitter(brightness=0.05, contrast=0.8, saturation=0.3),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
          ])
     ```
2. Using imgenet pre-trained Efficient-b0 and add some linear layers and dropout as training model 
   ```
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
   ```
3. Setting hyparameteres and starting training
   * Cross Entropy Loss
   * Adam optimizer with 0.001 learning rate
   * ReduceLROnPlateau learning rate schedular
   * epochs = 30

### Download Pretrained models
* Download pretrained models from this url: https://drive.google.com/file/d/1Stil1km9MSVUME94ouW7jroNrOMfoqqZ/view?usp=sharing
* Run ```python case2_train.py```

### Experiment results
![image](https://github.com/ChihChia-Li/Case2_COVID-19-Pneumonia-Detection/blob/main/Result/result.jpg)

### Inference
To inference the trained model, load the trained model and run ```python case2_test.py```
