# %%
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import os
from torch.nn.modules.pooling import AvgPool2d
from torchsummary import summary
from torchvision.transforms import transforms
from PIL import Image
from tqdm import tqdm
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision.ops import nms
import random
import torchvision.transforms.functional as TF
import pickle
import sys
# %%

MAXLENGTH = 47


class SkullDataset(torch.utils.data.Dataset):
    def __init__(self, path, mode, low, high):
        self.path = path
        self.mode = mode
        self.padding = False
        if mode == 'train':
            with open(os.path.join(path, 'records_train.json')) as f:
                self.labels = json.load(f)['datainfo']
            self.label = []
            self.mask = []
        self.study_list = sorted(os.listdir(
            os.path.join(path, mode)))[low:high]
        self.data = []
        for case_id in tqdm(self.study_list):
            labels = []
            for slice_file in os.listdir(os.path.join(self.path, self.mode, case_id)):
                labels.append(self.labels[slice_file[:-4]]['label'])
            if self.padding:
                for i in range(47-len(labels)):
                    if labels[0] == 0:
                        labels.append(0)
                    else:
                        labels.append(-1)
            if mode == 'train':
                self.label.append(torch.tensor(labels, dtype=torch.float32))
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])

        ]
        )

    def __getitem__(self, idx):
        case_id = self.study_list[idx]
        imgs = []
        angle=random.randint(-180,180)
        for slice_file in os.listdir(os.path.join(self.path, self.mode, case_id)):
            img = np.load(os.path.join(
                self.path, self.mode, case_id, slice_file))
            img = np.clip((img+1024)/4095, 0, 255)
            img = Image.fromarray(img)
            img = self.transform(img)
            img = TF.rotate(img, angle, fill=-1.0)
            imgs.append(img)
        if self.padding:
            for i in range(47-len(imgs)):
                imgs.append(-torch.ones(1, 512, 512))

        if self.mode == 'train':
            return torch.stack(imgs, dim=0), self.label[idx]
        else:
            return torch.stack(imgs, dim=0)

    def __len__(self):
        return len(self.study_list)


# %%
MAXLENGTH = 47


class SkullDataset(torch.utils.data.Dataset):
    def __init__(self, path, mode, low, high):
        self.path = path
        self.mode = mode
        self.padding=False
        if mode == 'train':
            with open(os.path.join(path, 'records_train.json')) as f:
                self.labels = json.load(f)['datainfo']
            self.label = []
            self.mask = []
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ]
        )
        self.study_list = sorted(os.listdir(
            os.path.join(path, mode)))[low:high]
        self.data = []
        for case_id in tqdm(self.study_list):
            imgs = []
            labels = []
            for slice_file in os.listdir(os.path.join(self.path, self.mode, case_id)):
                img = np.load(os.path.join(
                    self.path, self.mode, case_id, slice_file))
                img = np.clip((img+1024)/4095, 0, 255)
                img = Image.fromarray(img)
                img = self.transform(img)
                imgs.append(img)
                if mode == 'train':
                    labels.append(self.labels[slice_file[:-4]]['label'])
            if self.padding:
                for i in range(47-len(imgs)):
                    imgs.append(-torch.ones(1, 512, 512))
                    if mode == 'train':
                        if labels[0] == 0:
                            labels.append(0)
                        else:
                            labels.append(-1)
            self.data.append(torch.stack(imgs, dim=0))
            if mode == 'train':
                self.label.append(torch.tensor(labels, dtype=torch.float32))
    def __getitem__(self, idx):
        if self.mode == 'train':

            #angle = random.choice([-30, -15, 0, 15, 30])
            #TF.crop(img,)
            angle=random.randint(-180,180)
            angle=0
            img = TF.rotate(self.data[idx], angle, fill=-1)
            return img, self.label[idx]
        else:
            return self.data[idx]

    def __len__(self):
        return len(self.study_list)


# %%
train_data = SkullDataset('./skull', 'train',558,1116)
val_data = SkullDataset('./skull', 'train', 0, 50)
# %%
batchsize = 1
train_dataloader = torch.utils.data.DataLoader(
    train_data, batch_size=batchsize, shuffle=True)
# %%
val_dataloader = torch.utils.data.DataLoader(
    val_data, batch_size=batchsize, shuffle=False)
# %%


class PositionalEmbedding1D(nn.Module):

    def __init__(self, seq_len, dim):
        super().__init__()
        self.pos_embedding = nn.Parameter(torch.zeros(1, seq_len, dim))

    def forward(self, x):
        return x + self.pos_embedding[:,:x.shape[1],:]


class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(DoubleConv, self).__init__()
        self.dc = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 5, padding=2),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(),
            nn.Conv2d(out_ch, out_ch, 5, padding=2),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(),
        )

    def forward(self, x):
        x = self.dc(x)
        return x


class Skull_CNN_Transformer(nn.Module):
    def __init__(self, in_ch, depth, trans_layers):
        super().__init__()
        d_model = in_ch*64
        self.trans_layers = trans_layers
        self.down_list = nn.ModuleList()
        self.down_list.append(DoubleConv(1, in_ch))
        ratio = 2
        for i in range(depth):
            self.down_list.append(nn.Sequential(
                nn.MaxPool2d(4),
                DoubleConv(in_ch, in_ch*ratio)
            ))
            in_ch *= ratio
        trans_encode_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=8, dim_feedforward=d_model, dropout=0.2, batch_first=True)
        self.InterTransformer = nn.TransformerEncoder(
            trans_encode_layer, num_layers=trans_layers)
        self.inter_pos_embedding = PositionalEmbedding1D(
            48, d_model)
        self.diagnosis_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.diagnosis_fc = nn.Linear(d_model, 1)
        self.slice_diagnosis_fc = nn.Linear(d_model, 1)
        self.sigmoid = nn.Sigmoid()
        self.dropout=nn.Dropout(0.2)

    def forward(self, x):
        feature = x.reshape(x.shape[0]*x.shape[1], 1, 512, 512)
        for down_layer in self.down_list:
            feature = down_layer(feature)
            # b*l c w h
        # b*l c d_model -> b*l d_model
        transformer_out = feature.reshape(x.shape[0], x.shape[1], -1)
        transformer_out = self.InterTransformer(self.inter_pos_embedding(
            torch.cat([self.diagnosis_token, transformer_out], dim=1)))
        diagnosis = self.diagnosis_fc(self.dropout(transformer_out[:, 0, :]))
        diagnosis = self.sigmoid(diagnosis)
        slice_diagnosis = self.slice_diagnosis_fc(self.dropout(
            transformer_out[:, 1:, :])) # b l 1
        slice_diagnosis = self.sigmoid(slice_diagnosis)

        return diagnosis, slice_diagnosis



# %%
model = Skull_CNN_Transformer(8, 4, 2).cuda()
model.load_state_dict(torch.load('./model/predict/model_1116_.bin'))
optimizer = torch.optim.Adam(
    model.parameters(), lr=0.00002, betas=(0.9, 0.99))
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.95)
best_val_acc = 0.0
case_criterion = nn.BCELoss()
slice_criterion = nn.BCELoss()
# %%
for n in range(100):
    case_acc = 0.0
    slice_acc = 0.0
    mask_acc = 0.0
    train_loss = 0.0
    tp = 0
    p = 0
    fn = 0
    recall=0
    precision=0
    model.train()
    train = tqdm(train_dataloader)
    for num, (img, label) in enumerate(train):

        diagnosis, slice_diagnosis= model(img.cuda())
        case_loss = case_criterion(diagnosis.squeeze(-1), torch.tensor(
            abs(label[:, 0]), dtype=torch.float).cuda())
        slice_loss = slice_criterion(
            slice_diagnosis.squeeze(-1), torch.clamp(label, 0, 1).cuda())
        total_loss = case_loss+slice_loss
        train_loss += total_loss.item()
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        case_acc += (torch.round(diagnosis.detach().cpu())
                     == abs(label[:, 0])).numpy().mean()

        slice_acc += (torch.round(slice_diagnosis.squeeze(-1).detach().cpu())
                      == torch.clamp(label, 0, 1)).numpy().mean()
        train.set_postfix(
            {'n': n, 'loss': f'{train_loss/(num+1): .4f}', 's_acc': f'{slice_acc/(num+1):.4f}', 'c_acc': f'{case_acc/(num+1): .4f}'})
    val = tqdm(val_dataloader)
    val_case_acc = 0.0
    val_slice_acc = 0.0
    with torch.no_grad():
        for num, (img, label) in enumerate(val):
            diagnosis, slice_diagnosis = model(img.cuda())
            val_case_acc += (torch.round(diagnosis.detach().cpu())
                         == abs(label[:, 0])).numpy().mean()

            val_slice_acc += (torch.round(slice_diagnosis.squeeze(-1).detach().cpu())
                          == torch.clamp(label, 0, 1)).numpy().mean()
            val.set_postfix(
                {'n': n,'s_acc': f'{val_slice_acc/(num+1):.4f}', 'c_acc': f'{val_case_acc/(num+1): .4f}'})
    
    if best_val_acc >= val_case_acc:
        best_val_acc = val_case_acc
        torch.save(model.state_dict(), './model/predict/model_1116.bin')
        torch.save(optimizer.state_dict(), './model/predict/opt_1116.bin')
        torch.save(scheduler.state_dict(), './model/predict/sch_1116.bin')

    scheduler.step()
# %%
    torch.save(model.state_dict(), './model/predict/model_1116_.bin')
    torch.save(optimizer.state_dict(), './model/predict/opt_1116_.bin')
    torch.save(scheduler.state_dict(), './model/predict/sch_1116_.bin')

# %%

test_data = SkullDataset('./skull', 'test',0,130)
# %%
batchsize = 1
test_dataloader = torch.utils.data.DataLoader(
    test_data, batch_size=batchsize, shuffle=False)
# %%
models = [Skull_CNN_Transformer(8, 4, 2).cuda() for _ in range(2)]
name = ['558','1116']
for i, m in enumerate(models):
    m.load_state_dict( torch.load('./model/predict/model_'+name[i]+'_.bin'))
    m.eval()
# %%
sliding_width = 11
test = tqdm(test_dataloader)
with open('./prediction.csv', 'w') as file:
    file.write('id,label,coords\n')
    for idx, img in enumerate(test):
        fname = os.listdir(os.path.join(
            test_data.path, test_data.mode, test_data.study_list[idx]))
        with torch.no_grad():
            diagnosis = 0
            slice_diagnosis = 0
            for m in models:
                d, s = m(img.cuda())
                diagnosis += d/len(models)
                slice_diagnosis += s/len(models)
                    
            case_diagnosis = torch.round(diagnosis.detach().cpu())
            slice_diagnosis = torch.round(
                slice_diagnosis.squeeze().detach().cpu())
           
        for slice_num in range(len(fname)):
            slice_diag = int(slice_diagnosis[slice_num])

            if case_diagnosis:
                if slice_diag == 0:

                    slice_diag = -1
                else:
                    slice_diag=-1
                  
            else:
                slice_diag = 0
            file.write(f'{fname[slice_num][:-4]},{slice_diag},')
            file.write('\n')
# %%

val_data = SkullDataset('./skull', 'train',0,558)
# %%
batchsize = 1
val_dataloader = torch.utils.data.DataLoader(
    test_data, batch_size=batchsize, shuffle=False)
# %%
model = Skull_CNN_Transformer(8, 4, 2).cuda()
model.load_state_dict(torch.load('./model/predict/model_1116_.bin'))
val = tqdm(val_dataloader)
val_case_acc = 0.0
val_slice_acc = 0.0
with torch.no_grad():
    for num, (img, label) in enumerate(val):
        diagnosis, slice_diagnosis = model(img.cuda())
        val_case_acc += (torch.round(diagnosis.detach().cpu())
                        == abs(label[:, 0])).numpy().mean()

        val_slice_acc += (torch.round(slice_diagnosis.squeeze(-1).detach().cpu())
                        == torch.clamp(label, 0, 1)).numpy().mean()
        val.set_postfix(
            {'s_acc': f'{val_slice_acc/(num+1):.4f}', 'c_acc': f'{val_case_acc/(num+1): .4f}'})
# %%
