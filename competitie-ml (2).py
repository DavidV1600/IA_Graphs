#!/usr/bin/env python
# coding: utf-8

# In[70]:


import warnings
warnings.filterwarnings('ignore')
import os
import numpy as np
import pandas as pd
import tqdm as tqdm
import cv2
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report
import joblib
import copy
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
import torchvision.models as models
import torch.nn.functional as F
from sklearn.ensemble import RandomForestClassifier
from PIL import Image 


# In[79]:


def citeste_imagini_din_folder(folder):
    imagini = []
    nume_imagini = [] # retine numele lor adica [...].png 
    for nume_imagine in os.listdir(folder):
        path_imagine = os.path.join(folder, nume_imagine)
        imagine = cv2.imread(path_imagine)
        if imagine is not None:
            imagini.append(imagine)
            nume_imagini.append(nume_imagine)
    return imagini, nume_imagini


# In[80]:


# Dataframe-urile
train_df = pd.read_csv('/kaggle/input/ml-comp/train.csv')
validation_df = pd.read_csv('/kaggle/input/ml-comp/validation.csv')
test_df = pd.read_csv('/kaggle/input/ml-comp/test.csv')


# In[81]:


# Dictionare ca sa retin pentru fiecare id al imaginii label-ul corect
train_dict = train_df.set_index('image_id')['label'].to_dict()
validation_dict = validation_df.set_index('image_id')['label'].to_dict()
# test_dict = test_df.set_index('image_id')['label'].to_dict()


# In[ ]:


train_images, train_images_names = citeste_imagini_din_folder('/kaggle/input/ml-competition2024/train')
validation_images, validation_images_names = citeste_imagini_din_folder('/kaggle/input/ml-competition2024/validation')
test_images, test_images_names = citeste_imagini_din_folder('/kaggle/input/ml-competition2024/test')


# In[83]:


train_images_names = [x.replace('.png','') for x in train_images_names]
validation_images_names = [x.replace('.png','') for x in validation_images_names]
test_images_names = [x.replace('.png','') for x in test_images_names]


# In[84]:


# retin in X - imaginile si in Y - label-urile
X_train = train_images
y_train = [train_dict[name] for name in train_images_names]
X_val = validation_images
y_val = [validation_dict[name] for name in validation_images_names]
X_test = test_images
print(y_train[:10])
#concatenez datele de train si validare pentru o predictie finala
X_train_val = np.concatenate((X_train, X_val))
y_train_val = np.concatenate((y_train, y_val))


# 

# In[85]:


#  transform imaginile in tensor astfel,
#  intai le fac np.array si le schimb tipul
#  apoi le fac tensor (astfel e mult mai rapid decat
#  sa transform direct in tensor) si dupa schimb axele din 
#  (NR_IMAGINI,H,W,RGB) -> (NR_IMAGINI,RBG,H,W) pentru a putea
#  sa le rotesc mai apoi (tre sa fie PIL images si asa e formatul lor)

X_train = torch.as_tensor(np.array(X_train), dtype=torch.float32).permute(0, 3, 1, 2)
X_val = torch.as_tensor(np.array(X_val), dtype=torch.float32).permute(0, 3, 1, 2)
X_test = torch.as_tensor(np.array(X_test), dtype=torch.float32).permute(0, 3, 1, 2)
X_train_val = torch.as_tensor(np.array(X_train_val), dtype=torch.float32).permute(0, 3, 1, 2)


# In[91]:


# clasa ca retin imaginile cu labe-urile intr-o singura variabila
# ca mai apoi sa le pun in dataloader
# dupa ce le-am aplicat transformarea
class Dataset_Imagini(torch.utils.data.Dataset):
    def __init__(self, imagini, labels_imagini, transforma=None):
        self.imagini = imagini
        self.labels_imagini = labels_imagini
        self.transforma = transforma

    def __len__(self):
        return len(self.imagini)

    def __getitem__(self, index):
        imagine = self.imagini[index]
        label_imagine = self.labels_imagini[index]
        if self.transforma:
            imagine = self.transforma(imagine)
        return imagine, label_imagine


# In[87]:


# Normalizez valorile pixelilor
X_train = X_train / 255
X_val = X_val / 255
X_test = X_test / 255
X_train_val = X_train_val / 255


# In[88]:


def calcul_medie_deviatie(imagini):
    suma_canale, suma_canale_squared = 0, 0
    for imagine in imagini:
        suma_canale += torch.mean(imagine, axis=[1,2])
        suma_canale_squared += torch.mean(imagine**2, axis=[1,2])
        
    medie = suma_canale / len(imagini)
    deviatie  = (suma_canale_squared / len(imagini) - medie**2) ** 0.5
    return medie, deviatie


# In[89]:


medie, deviatie = calcul_medie_deviatie(X_train)
print(medie, deviatie)


# In[92]:


# definec o transformare pentru datele de train
# le dau flip cu sansa de 0.5 pe orizontala si le rotesc cu +- 10 grade
# le normalizez cu media si deviatia per canale
transforma_train = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.Normalize(mean=[0.4986, 0.4727, 0.4257], std=[0.2609, 0.2547, 0.2739])
])


# transformare 
transforma_val = transforms.Compose([
    transforms.Normalize(mean=[0.4986, 0.4727, 0.4257], std=[0.2609, 0.2547, 0.2739])
])

# creez obiectele de tip dataset
train_dataset = Dataset_Imagini(X_train, y_train, transforma_train)
val_dataset = Dataset_Imagini(X_val, y_val, transforma_val)
test_dataset = Dataset_Imagini(X_test, test_images_names, transforma_val)
train_val_dataset = Dataset_Imagini(X_train_val, y_train_val, transforma_train)

# folosesc DataLoader-ul din pytorch ca mai apoi sa parcurg imaginile in train
# in batch-uri de 64 si shuffle-uite random
train_val_dataloader = DataLoader(train_val_dataset, batch_size=64, shuffle=True)
train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=64, shuffle=False)
test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=False)


# In[93]:


class BlocSE(nn.Module):
    def __init__(self, canale_input, reductie=16):
        super(BlocSE, self).__init__()
        self.fc1 = nn.Conv2d(canale_input, canale_input // reductie, kernel_size=1)
        self.fc2 = nn.Conv2d(canale_input // reductie, canale_input, kernel_size=1)
    
    def forward(self, x):
        batch_size, nr_canale, _, _ = x.size()
        squeeze = F.adaptive_avg_pool2d(x, 1)
        excitation = F.relu(self.fc1(squeeze))
        excitation = torch.sigmoid(self.fc2(excitation))
        excitation = excitation.view(batch_size, nr_canale, 1, 1)
        return x * excitation


# In[94]:


class BlocRezidual(nn.Module):
    def __init__(self, canale_input, canale_output, reductie=16):
        super(BlocRezidual, self).__init__()
        self.conv1 = nn.Conv2d(canale_input, canale_output, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(canale_output)
        self.conv2 = nn.Conv2d(canale_output, canale_output, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(canale_output)
        self.se = BlocSE(canale_output, reductie)
        
        if canale_input != canale_output:
            self.shortcut = nn.Sequential(
                nn.Conv2d(canale_input, canale_output, kernel_size=1, stride=1),
                nn.BatchNorm2d(canale_output)
            )
        else:
            self.direct = nn.Identity()

    def forward(self, x):
        residual = self.direct(x)
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x = self.se(x)
        x += residual
        return F.relu(x)


# In[95]:


class ModulInception(nn.Module):
    def __init__(self, canale_input, out1x1, red3x3, out3x3, red5x5, out5x5, out_pool, reductie=16):
        super(ModulInception, self).__init__()
        self.branch1 = nn.Sequential(
            nn.Conv2d(canale_input, out1x1, kernel_size=1),
            nn.BatchNorm2d(out1x1),
            nn.ReLU(True),
        )
        self.branch2 = nn.Sequential(
            nn.Conv2d(canale_input, red3x3, kernel_size=1),
            nn.BatchNorm2d(red3x3),
            nn.ReLU(True),
            nn.Conv2d(red3x3, out3x3, kernel_size=3, padding=1),
            nn.BatchNorm2d(out3x3),
            nn.ReLU(True),
        )
        self.branch3 = nn.Sequential(
            nn.Conv2d(canale_input, red5x5, kernel_size=1),
            nn.BatchNorm2d(red5x5),
            nn.ReLU(True),
            nn.Conv2d(red5x5, out5x5, kernel_size=5, padding=2),
            nn.BatchNorm2d(out5x5),
            nn.ReLU(True),
        )
        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(canale_input, out_pool, kernel_size=1),
            nn.BatchNorm2d(out_pool),
            nn.ReLU(True),
        )
        self.se = BlocSE(out1x1 + out3x3 + out5x5 + out_pool, reductie)
    
    def forward(self, x):
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)
        branch4 = self.branch4(x)
        outputs = [branch1, branch2, branch3, branch4]
        x = torch.cat(outputs, 1)
        x = self.se(x)
        return x


# In[98]:


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        
        self.inception1 = ModulInception(canale_input=32, out1x1=32, red3x3=32, out3x3=32, red5x5=32, out5x5=32, out_pool=32)
        self.inception2 = ModulInception(canale_input=128, out1x1=64, red3x3=64, out3x3=64, red5x5=64, out5x5=64, out_pool=64)
        
        self.residual1 = BlocRezidual(canale_input=256, canale_output=256) #canale_input sunt puse sa se potriveasca cu output de la inception2
        self.residual2 = BlocRezidual(canale_input=256, canale_output=256)
        
        self.conv2 = nn.Conv2d(in_channels=256, out_channels=64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.dropout = nn.Dropout(0.5)
        
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        
        self.fc1 = nn.Linear(64, 128)
        self.fc2 = nn.Linear(128, 256)
        self.output_layer = nn.Linear(256, 3)
        
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)
        x = self.inception1(x)
        x = self.pool(x)
        x = self.inception2(x)
        
        x = self.residual1(x)
        x = self.pool(x)
        x = self.residual2(x)
        
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        
        x = self.global_avg_pool(x)
        x = x.view(x.size(0), -1)
        
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.output_layer(x)
        
        return x


# In[99]:


model2 = CNN()
print(model2)


# In[108]:


from torch.optim.lr_scheduler import StepLR

optimizator = optim.Adam(model2.parameters(), lr=0.001)
lr_changer = StepLR(optimizator, step_size=7, gamma=0.1)
criteriu_loss = nn.CrossEntropyLoss()

NR_EPOCI = 25

for nr_epoca in range(NR_EPOCI):
    print(f"Epoca: {nr_epoca}:")
    model2.train()
    
    loss_epoca = 0.0
    predictii_corecte_epoca = 0
    nr_predictii_epoca = 0
    
    for batch, (batch_imagini, batch_labels) in enumerate(train_dataloader):
        optimizator.zero_grad()
        predictie_batch = model2(batch_imagini)
        loss_batch = criteriu_loss(predictie_batch, batch_labels)
        
        loss_batch.backward()
        optimizator.step()
        
        loss_epoca += loss_batch.item()
        
        _, predictie = torch.max(predictie_batch, 1)
        predictii_corecte_epoca += (predictie == batch_labels).sum().item()
        nr_predictii_epoca += batch_labels.size(0)
        
        if batch % 30 == 0:
            print(f"Batch {batch} loss: {loss_batch.item():>6f}")

    acurate_epoca = (predictii_corecte_epoca / nr_predictii_epoca) * 100
    loss_epoca = loss_epoca / len(train_dataloader)
    print(f"Epoca {nr_epoca}, Train Loss: {loss_epoca:.3f}, Train acuratete: {acurate_epoca:.2f}%")
    
    model2.eval()
    loss_validare = 0.0
    nr_predictii_corecte_validare = 0
    nr_predictii_totale_validare = 0
    
    with torch.no_grad():
        for image_batch, labels_batch in val_dataloader:            
            predictie = model2(image_batch)
            loss = criteriu_loss(predictie, labels_batch)
            loss_validare += loss.item()
            
            _, predictie = torch.max(predictie, 1)
            nr_predictii_corecte_validare += (predictie == labels_batch).sum().item()
            nr_predictii_totale_validare += labels_batch.size(0)
    
    loss_validare = loss_validare / len(val_dataloader)
    acuratete_validare = 100 * (nr_predictii_corecte_validare / nr_predictii_totale_validare)
    print(f"Validare loss: {loss_validare:.3f}, Validare acuratete: {acuratete_validare:.2f}%")
    
    lr_changer.step()


# In[ ]:


predictii = []
model2.eval()
with torch.no_grad():
    for imagini, nume_imagini in test_dataloader:
        predictii = model2(imagini)
        _, preds = torch.max(predictii, 1)
        for filename, pred in zip(nume_imagini, preds.cpu().numpy()):
            predictii.append((filename, pred))


# In[ ]:


predictii_df = pd.DataFrame(predictii, columns=['image_id', 'label'])
predictii_df.to_csv('test_predictions_FULL_CNN_INCEPTION_SE.csv', index=False)

