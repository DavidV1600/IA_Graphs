#!/usr/bin/env python
# coding: utf-8

# In[69]:


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
from torch.optim.lr_scheduler import StepLR


# In[70]:


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


# In[71]:


# Dataframe-urile
train_df = pd.read_csv('/kaggle/input/ml-comp/train.csv')
validation_df = pd.read_csv('/kaggle/input/ml-comp/validation.csv')
test_df = pd.read_csv('/kaggle/input/ml-comp/test.csv')


# In[72]:


# Dictionare ca sa retin pentru fiecare id al imaginii label-ul corect
train_dict = train_df.set_index('image_id')['label'].to_dict()
validation_dict = validation_df.set_index('image_id')['label'].to_dict()
# test_dict = test_df.set_index('image_id')['label'].to_dict()


# In[ ]:


train_images, train_images_names = citeste_imagini_din_folder('/kaggle/input/ml-competition2024/train')
validation_images, validation_images_names = citeste_imagini_din_folder('/kaggle/input/ml-competition2024/validation')
test_images, test_images_names = citeste_imagini_din_folder('/kaggle/input/ml-competition2024/test')


# In[74]:


train_images_names = [x.replace('.png','') for x in train_images_names]
validation_images_names = [x.replace('.png','') for x in validation_images_names]
test_images_names = [x.replace('.png','') for x in test_images_names]


# In[75]:


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

# In[76]:


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


# In[77]:


# clasa ca retin imaginile cu labe-urile intr-o singura variabila
# ca mai apoi sa le pun in dataloader
# dupa ce le-am aplicat transformarea
class Dataset_Imagini(torch.utils.data.Dataset):
    def __init__(self, imagini, labels_imagini, transformare):
        self.imagini = imagini
        self.labels_imagini = labels_imagini
        self.transformare = transformare

    def __len__(self):
        return len(self.imagini)

    def __getitem__(self, index):
        imagine = self.imagini[index]
        imagine = self.transformare(imagine)
        label_imagine = self.labels_imagini[index]
        return imagine, label_imagine


# In[78]:


# Normalizez valorile pixelilor
X_train = X_train / 255
X_val = X_val / 255
X_test = X_test / 255
X_train_val = X_train_val / 255


# In[79]:


def calcul_medie_deviatie(imagini):
    suma_canale, suma_canale_squared = 0, 0
    for imagine in imagini:
        suma_canale += torch.mean(imagine, axis=[1,2])
        suma_canale_squared += torch.mean(imagine**2, axis=[1,2])
        
    medie = suma_canale / len(imagini)
    deviatie  = (suma_canale_squared / len(imagini) - medie**2) ** 0.5
    return medie, deviatie


# In[80]:


medie, deviatie = calcul_medie_deviatie(X_train)
print(medie, deviatie)


# In[81]:


# definec o transformare pentru datele de train
# le dau flip cu sansa de 0.5 pe orizontala si le rotesc cu +- 10 grade
# le normalizez cu media si deviatia per canale
transformare_train = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.Normalize(mean=[0.4986, 0.4727, 0.4257], std=[0.2609, 0.2547, 0.2739])
])


# transformare 
transformare_validare = transforms.Compose([
    transforms.Normalize(mean=[0.4986, 0.4727, 0.4257], std=[0.2609, 0.2547, 0.2739])
])

# creez obiectele de tip dataset
dataset_train = Dataset_Imagini(X_train, y_train, transformare_train)
dataset_validare = Dataset_Imagini(X_val, y_val, transformare_validare)
dataset_test = Dataset_Imagini(X_test, test_images_names, transformare_validare)
dataset_train_validare = Dataset_Imagini(X_train_val, y_train_val, transformare_train)

# folosesc DataLoader-ul din pytorch ca mai apoi sa parcurg imaginile in train
# in batch-uri de 64 si shuffle-uite random
train_val_dataloader = DataLoader(dataset_train_validare, batch_size=64, shuffle=True)
dataloader_train = DataLoader(dataset_train, batch_size=64, shuffle=True)
dataloader_validare = DataLoader(dataset_validare, batch_size=64, shuffle=False)
dataloader_test = DataLoader(dataset_test, batch_size=64, shuffle=False)


# In[82]:


class BlocSE(nn.Module):
    def __init__(self, canale_input, reductie=16):
        super(BlocSE, self).__init__()
        self.f1 = nn.Conv2d(canale_input, canale_input//reductie, 1) # reductia este de fapt cat de puternica sa 
        self.f2 = nn.Conv2d(canale_input//reductie, canale_input, 1) # fie calibrarea ponderilor mapelor
    
    def forward(self, x):
        marime_batch, nr_canale, a, b = x.size()
        squeeze = F.adaptive_avg_pool2d(x, 1) # reduce fiecare mapa la avg de pixeli
        excitation = F.relu(self.f1(squeeze)) # reduce numarul de canale
        excitation = torch.sigmoid(self.f2(excitation)) # aduce inapoi numarul de canale, cu ponderi mai accentuate
        excitation = excitation.view(marime_batch, nr_canale, 1, 1)
        return x * excitation #inmulteste x cu ponderile per mapa, astfel


# In[83]:


class BlocRezidual(nn.Module):
    def __init__(self, canale_input, canale_output):
        super(BlocRezidual, self).__init__()
        self.c1 = nn.Conv2d(canale_input, canale_output, kernel_size=3, padding=1)
        self.n1 = nn.BatchNorm2d(canale_output)
        self.c2 = nn.Conv2d(canale_output, canale_output, kernel_size=3, padding=1)
        self.n2 = nn.BatchNorm2d(canale_output)
        self.bloc = BlocSE(canale_output)
        self.direct = nn.Identity() # retine input-ul initial, pentru a putea fi folosit ca input

    def forward(self, x):
        residual = self.direct(x) # valoarea initiala a lui x
        x = F.relu(self.n1(self.c1(x)))
        x = self.n2(self.c2(x))
        x = self.bloc(x)
        x += residual 
        # adaug valoarea initiala a lui x la output astfel functia cautata 
        # devenind ceva de genul f(x) = h(x) - x, acest lucru se numeste skip connection
        # si poate ajuta reteaua sa invete foarte bine, chiar daca anumite straturi nu au inceput
        # inca sa invete
        return F.relu(x)


# In[90]:


# Formula pentru marime output_ului in functie de input:
# output_size = (input_size - kernel_size + 2 * paddng) / stride + 1 

class ModulInception(nn.Module):
    def __init__(self, canale_input, output1, input3, output3, input5, output5, output1x1):
        super(ModulInception, self).__init__()
        self.br1 = nn.Sequential(
            nn.Conv2d(canale_input, output1, kernel_size=1),
            nn.BatchNorm2d(output1),
            nn.ReLU(True),
            # marime_output = (canale_input - 1 + 2 * 0) / 1 + 1 = canal_input
        )
        self.br2 = nn.Sequential(
            nn.Conv2d(canale_input, input3, kernel_size=1), # kernel_size=1 pastreaza marimile
            nn.BatchNorm2d(input3),
            nn.ReLU(True),
            nn.Conv2d(input3, output3, kernel_size=3, padding=1),
            nn.BatchNorm2d(output3),
            nn.ReLU(True),
            # marime_output = (canale_input - 3 + 2 * 1) / 1 + 1 = canal_input
        )
        self.br3 = nn.Sequential(
            nn.Conv2d(canale_input, input5, kernel_size=1),# kernel_size=1 pastreaza marimile
            nn.BatchNorm2d(input5),
            nn.ReLU(True),
            nn.Conv2d(input5, output5, kernel_size=5, padding=2),
            nn.BatchNorm2d(output5),
            nn.ReLU(True),
            # marime_output = (canale_input - 5 + 2 * 2) / 1 + 1 = canal_input
        )
        self.br4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(canale_input, output1x1, kernel_size=1),
            nn.BatchNorm2d(output1x1),
            nn.ReLU(True),
            # marime_output = (canale_input - 3 + 2 * 1) / 1 + 1 = canal_input
        )
        self.bloc = BlocSE(output1 + output3 + output5 + output1x1)
        # le concatenez si le adjustez ponderile folosind BlocSE
    
    def forward(self, x):
        br1 = self.br1(x)
        br2 = self.br2(x)
        br3 = self.br3(x)
        br4 = self.br4(x)
        output = [br1, br2, br3, br4]
        x = torch.cat(output, 1)
        x = self.bloc(x)
        return x


# In[91]:


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.c1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
        self.n1 = nn.BatchNorm2d(32)
        
        self.inc1 = ModulInception(canale_input=32, output1=32, input3=32, output3=32, input5=32, output5=32, output1x1=32)
        self.inc2 = ModulInception(canale_input=128, output1=64, input3=64, output3=64, input5=64, output5=64, output1x1=64)
        
        self.res1 = BlocRezidual(canale_input=256, canale_output=256)
        self.res2 = BlocRezidual(canale_input=256, canale_output=256)
        
        self.c2 = nn.Conv2d(in_channels=256, out_channels=64, kernel_size=3, padding=1)
        self.n2 = nn.BatchNorm2d(64)
        
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.avg_global = nn.AdaptiveAvgPool2d(1)
        
        self.dropout = nn.Dropout(0.5)
        self.f1 = nn.Linear(64, 128)
        self.f2 = nn.Linear(128, 256)
        self.output = nn.Linear(256, 3)
        
    def forward(self, x):
        x = self.max_pool(F.relu(self.n1(self.c1(x)))) # (3x80x80) -> (32x40x40)
        x = self.inc1(x) # (32x40x40) -> (128x40x40)
        x = self.max_pool(x) # (128x40x40) -> (128x20x20)
        x = self.inc2(x) # (128x20x20) -> (256x20x20)
        
        x = self.res1(x) # (256x20x20) -> (256x20x20)
        x = self.max_pool(x) # (256x20x20) -> (256x10x10)
        x = self.res2(x) # (256x10x10) -> (256x10x10)
        
        x = self.max_pool(F.relu(self.n2(self.c2(x))))  #(256x10x10) -> (64x5x5)
        x = self.avg_global(x)# fiecare mapa -> media pixelilor ei (64x5x5) -> (64x1x1)
        x = x.view(x.size(0), -1)# pastrez prima dimensiune, restul le concatenez pentru mini-NN ce urmeaza (64x1x1) -> (64)
        
        x = F.relu(self.f1(x)) # clasic NN (64) -> (128)
        x = self.dropout(x)
        x = F.relu(self.f2(x)) # (128) -> (256)
        x = self.dropout(x)
        x = self.output(x) # (256) -> (3)
        
        return x


# In[92]:


model2 = CNN()
print(model2)


# In[ ]:


criteriu_loss = nn.CrossEntropyLoss()
optimizator = optim.Adam(model2.parameters(), lr=0.001)
lr_changer = StepLR(optimizator, step_size=7)

NR_EPOCI = 25
for nr_epoca in range(NR_EPOCI):
    print(f"---{nr_epoca}---")
    model2.train()
    predictii_corecte_epoca = 0
    nr_predictii_epoca = 0
    loss_epoca = 0.0
    
    for batch_imagini, batch_labels in dataloader_train:
        optimizator.zero_grad()
        predictie_batch = model2(batch_imagini)
        loss_batch = criteriu_loss(predictie_batch, batch_labels)
        
        loss_batch.backward()
        optimizator.step()
        loss_epoca += loss_batch.item()
        
        predictie = predictie_batch.argmax(dim=1)
        predictii_corecte_epoca += (predictie == batch_labels).sum().item()
        nr_predictii_epoca += batch_labels.shape[0]

    acurate_epoca = (predictii_corecte_epoca / nr_predictii_epoca) * 100
    loss_epoca = loss_epoca / len(dataloader_train)
    print(f"Train Loss: {loss_epoca}")
    print(f"Train Acc: {acurate_epoca}%")
    
    model2.eval()
    nr_predictii_corecte_validare = 0
    nr_predictii_totale_validare = 0
    loss_validare = 0.0

    with torch.no_grad():
        for imagini_validare, labels_validare in dataloader_validare:            
            predictie = model2(imagini_validare)
            loss = criteriu_loss(predictie, labels_validare)
            loss_validare += loss.item()
            
            predictie = predictie.argmax(dim=1)
            nr_predictii_corecte_validare += (predictie == labels_validare).sum().item()
            nr_predictii_totale_validare += labels_validare.shape[0]
    
    loss_validare = loss_validare / len(dataloader_validare)
    acuratete_validare = 100 * (nr_predictii_corecte_validare / nr_predictii_totale_validare)
    print(f"Validare loss: {loss_validare}")
    print(f"Validare Acc: {acuratete_validare}%")
    
    lr_changer.step()


# In[ ]:





# In[26]:


predictii_finale = []
model2.eval()
with torch.no_grad():
    for imagini, nume_imagini in dataloader_test:
        predictii = model2(imagini)
        predictii = predictii.argmax(dim=1)
        for nume_imagine, predictie in zip(nume_imagini, predictii.numpy()):
            predictii_finale.append((nume_imagine, predictie))


# In[27]:


predictii_df = pd.DataFrame(predictii_finale, columns=['image_id', 'label'])
predictii_df.to_csv('test_predictions_FULL_CNN_INCEPTION_SE.csv', index=False)


# In[ ]:




