#!/usr/bin/env python
# coding: utf-8

# In[2]:


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


# In[3]:


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


# In[5]:


train_df = pd.read_csv('train.csv')
validation_df = pd.read_csv('validation.csv')
test_df = pd.read_csv('test.csv')


# In[6]:


# Dictionare ca sa retin pentru fiecare id al imaginii label-ul corect
train_dict = train_df.set_index('image_id')['label'].to_dict()
validation_dict = validation_df.set_index('image_id')['label'].to_dict()
# test_dict = test_df.set_index('image_id')['label'].to_dict()


# In[7]:


train_images, train_images_names = citeste_imagini_din_folder('train')
validation_images, validation_images_names = citeste_imagini_din_folder('validation')
test_images, test_images_names = citeste_imagini_din_folder('test')


# In[8]:


#scot .png ca sa fie acelasi format ca in submission.csv
train_images_names = [x.replace('.png','') for x in train_images_names]
validation_images_names = [x.replace('.png','') for x in validation_images_names]
test_images_names = [x.replace('.png','') for x in test_images_names]


# In[9]:


# retin in X - imaginile si in Y - label-urile
X_train = train_images
y_train = [train_dict[name] for name in train_images_names]
X_val = validation_images
y_val = [validation_dict[name] for name in validation_images_names]
X_test = test_images
print(y_train[:10])


# In[10]:


#le dau rezise sa devina vector (1d) ca sa le pot pune ca input in NN
X_train = torch.as_tensor(np.array(X_train), dtype=torch.float32).reshape(-1, 19200)
X_val = torch.as_tensor(np.array(X_val), dtype=torch.float32).reshape(-1, 19200)
X_test = torch.as_tensor(np.array(X_test), dtype=torch.float32).reshape(-1, 19200)


# In[11]:


print(X_train.dtype)


# In[12]:


#fac variabilele in care retin perechi de tipul (imagine-label) ca sa le pot pune in DataLoader
train_data = [(img, label) for img,label in zip(X_train,y_train)]
val_data = [(img, label) for img,label in zip(X_val,y_val)]


# In[13]:


plt.hist(y_train, bins=5)
#plt.hist(y_val, bins=5)


# In[ ]:





# In[14]:


#fac dataloader din ele
train_dataloader = DataLoader(train_data, batch_size=64)
val_dataloader = DataLoader(val_data, batch_size=64)


# In[16]:


class NN(nn.Module):
    def __init__(self):
        super(NN, self).__init__()
        self.f1 = nn.Linear(80*80*3, 1920)
        self.f2 = nn.Linear(1920, 1280)
        self.f3 = nn.Linear(1280, 640)
        self.f34 = nn.Linear(640, 320)
        self.f4 = nn.Linear(320, 160)
        self.output = nn.Linear(160, 3)
        self.dropout = nn.Dropout(0.5)
        self.norm1 = nn.BatchNorm1d(1920)
        self.norm2 = nn.BatchNorm1d(1280)
        self.norm3 = nn.BatchNorm1d(640)
        self.norm34 = nn.BatchNorm1d(320)
        self.norm4 = nn.BatchNorm1d(160)
        
    def forward(self, x):
        x = F.relu(self.norm1(self.f1(x)))
        x = self.dropout(x)
        x = F.relu(self.norm2(self.f2(x)))
        x = self.dropout(x)
        x = F.relu(self.norm3(self.f3(x)))
        x = self.dropout(x)
        x = F.relu(self.norm34(self.f34(x)))
        x = self.dropout(x)
        x = F.relu(self.norm4(self.f4(x)))
        x = self.dropout(x)
        x = self.output(x)
        return x


# In[17]:


model = NN()


# In[23]:


optimizator = optim.Adam(model.parameters(), lr=0.001)
criteriu_loss = nn.CrossEntropyLoss()
NR_EPOCI = 50
for nr_epoca in range(NR_EPOCI):
    print(f"Epoca: {nr_epoca}:")
    
    model.train()
    predictii_corecte_epoca = 0
    nr_predictii_epoca = 0
    loss_epoca = 0.0
    for batch, (batch_imagini, batch_labels) in enumerate(train_dataloader):
        optimizator.zero_grad()
        predictie_batch = model(batch_imagini)
        loss_batch = criteriu_loss(predictie_batch, batch_labels)
        loss_batch.backward()
        
        optimizator.step()
        loss_epoca += loss_batch.item()
        
        _, predictie = torch.max(predictie_batch, 1)
        predictii_corecte_epoca += (predictie == batch_labels).sum().item()
        nr_predictii_epoca += batch_labels.size(0)
        
        if batch % 30 == 0:
            print(f"Batch {batch} loss: {loss_batch.item():>6f}")
    loss_epoca = loss_epoca / len(train_dataloader)
    acurate_epoca = (predictii_corecte_epoca / nr_predictii_epoca) * 100
    print(f"Epoca {nr_epoca}, Train Loss: {loss_epoca:.3f}, Train acuratete: {acurate_epoca:.2f}%")
    
    model.eval()
    nr_predictii_corecte_validare = 0
    nr_predictii_totale_validare = 0
    loss_validare = 0.0
    with torch.no_grad():
        for batch_imagini, batch_labels in val_dataloader:            
            predictie = model(batch_imagini)
            loss = criteriu_loss(predictie, batch_labels)
            loss_validare += loss.item()
            
            _, predictie = torch.max(predictie, 1)
            nr_predictii_corecte_validare += (predictie == batch_labels).sum().item()
            nr_predictii_totale_validare += batch_labels.size(0)
            
    acuratete_validare = 100 * (nr_predictii_corecte_validare / nr_predictii_totale_validare)
    loss_validare = loss_validare / len(val_dataloader)

    print(f"Validare loss: {loss_validare:.3f}, Validare acuratete: {acuratete_validare:.2f}%")


# In[ ]:


predictii = []
model.eval()
with torch.no_grad():
    for imagini, nume_imagini in test_dataloader:
        predictii = model(imagini)
        _, preds = torch.max(predictii, 1)
        for filename, pred in zip(nume_imagini, preds.cpu().numpy()):
            predictii.append((filename, pred))


# In[23]:


predictii_df = pd.DataFrame(predictii, columns=['image_id', 'label'])
predictii_df.to_csv('test_predictions_FULL_CNN_INCEPTION_SE.csv', index=False)


# In[24]:





# In[ ]:




