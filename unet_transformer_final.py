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
        masks = []
        angle = random.choice([-30, -15, 0, 15, 30])
        #angle=random.randint(-180,180)
        for slice_file in os.listdir(os.path.join(self.path, self.mode, case_id)):
            img = np.load(os.path.join(
                self.path, self.mode, case_id, slice_file))
            img = np.clip((img+1024)/4095, 0, 255)
            img = Image.fromarray(img)
            img = self.transform(img)
            img = TF.rotate(img, angle, fill=-1.0)
            imgs.append(img)
            if self.mode == 'train':
                mask = torch.zeros(1, 512, 512)
                for coor in self.labels[slice_file[:-4]]['coords']:
                    margin = 5
                    for i in range(coor[0]-margin, coor[0]+margin+1):
                        for j in range(coor[1]-margin, coor[1]+margin+1):
                            try:
                                mask[0, i, j] = 1
                            except:
                                continue
                mask = TF.rotate(mask, angle)
                masks.append(mask)
        if self.padding:
            for i in range(47-len(imgs)):
                imgs.append(-torch.ones(1, 512, 512))
                if self.mode == 'train':
                    masks.append(torch.zeros(1, 512, 512))

        if self.mode == 'train':
            return torch.stack(imgs, dim=0), self.label[idx], torch.cat(masks, dim=0)
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
            masks = []
            labels = []
            slice_file=os.listdir(os.path.join(self.path, self.mode, case_id))[0]
            if self.labels[slice_file[:-4]]['label']==0:
                continue
            for slice_file in os.listdir(os.path.join(self.path, self.mode, case_id)):
                img = np.load(os.path.join(
                    self.path, self.mode, case_id, slice_file))
                img = np.clip((img+1024)/4095, 0, 255)
                img = Image.fromarray(img)
                img = self.transform(img)
                imgs.append(img)
                if mode == 'train':
                    mask = torch.zeros(1, 512, 512)
                    for coor in self.labels[slice_file[:-4]]['coords']:
                        margin = 10
                        for i in range(coor[0]-margin, coor[0]+margin+1):
                            for j in range(coor[1]-margin, coor[1]+margin+1):
                                try:
                                    mask[0, i, j] = 1
                                except:
                                    continue
                    masks.append(mask)
                    labels.append(self.labels[slice_file[:-4]]['label'])
            self.data.append(torch.stack(imgs, dim=0))
            if mode == 'train':
                self.label.append(torch.tensor(labels, dtype=torch.float32))
                self.mask.append(torch.cat(masks, dim=0))
        self.transform=transforms.RandomApply(
            nn.ModuleList(
                []
            )
            ,p=0.3
        )

    def __getitem__(self, idx):
        if self.mode == 'train':

            angle = random.choice([-30, -15, 0, 15, 30])
            img = TF.rotate(self.data[idx], angle, fill=-1)
            mask = TF.rotate(self.mask[idx], angle)
            return img, self.label[idx], mask
        else:
            return self.data[idx]

    def __len__(self):
        return len(self.data)


# %%
train_data = SkullDataset('./skull', 'train', 0, 400)
val_data = SkullDataset('./skull', 'train', 1076, 1116)
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
        return x + self.pos_embedding[:, :x.shape[1], :]


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


class Up(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, in_ch//2, 4, 4)
        self.dc = DoubleConv(in_ch, out_ch)

    def forward(self, lower, upper):
        lower = self.up(lower)
        out = self.dc(torch.cat([lower, upper], dim=1))
        return out


class Skull_Unet_Transformer(nn.Module):
    def __init__(self, in_ch, depth, trans_layers):
        super().__init__()
        d_model = in_ch*64
        num_class = 2
        self.trans_layers = trans_layers
        self.down_list = nn.ModuleList()
        self.up_list = nn.ModuleList()
        self.down_list.append(DoubleConv(1, in_ch))
        ratio = 2
        for i in range(depth):
            self.down_list.append(nn.Sequential(
                nn.MaxPool2d(4),
                DoubleConv(in_ch, in_ch*ratio)
            ))
            in_ch *= ratio
        for i in range(depth):
            self.up_list.append(Up(in_ch, in_ch//ratio))
            in_ch //= ratio
        self.up_list.append(nn.Sequential(nn.Dropout(0.2),
        nn.Conv2d(in_ch, num_class, 1)))
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
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        temp = x.reshape(x.shape[0]*x.shape[1], 1, 512, 512)
        out_x = []
        for down_layer in self.down_list:
            temp = down_layer(temp)
            out_x.append(temp)
            # b*l c w h
        # b*l c d_model -> b*l d_model
        transformer_out = out_x[-1].reshape(x.shape[0], x.shape[1], -1)
        transformer_out = self.InterTransformer(self.inter_pos_embedding(
            torch.cat([self.diagnosis_token, transformer_out], dim=1)))
        diagnosis = self.diagnosis_fc(self.dropout(transformer_out[:, 0, :]))
        diagnosis = self.sigmoid(diagnosis)
        slice_diagnosis = self.slice_diagnosis_fc(
            self.dropout(transformer_out[:, 1:, :]))  # b l 1
        slice_diagnosis = self.sigmoid(slice_diagnosis)
        # b l c w h
        out_x[-1] = transformer_out[:, 1:,
                                    :].reshape(out_x[-1].shape)  # b*l c w h
        for i, up_layer in enumerate(self.up_list):
            if i != (len(self.up_list)-1):
                out_x[-1] = up_layer(out_x[-1], out_x[-i-2])
            else:
                out_x[-1] = up_layer(out_x[-1])

        return diagnosis, slice_diagnosis, out_x[-1].reshape(x.shape[0], x.shape[1], 2, 512, 512).permute(0, 2, 1, 3, 4)


class FocalLoss(nn.modules.loss._WeightedLoss):
    def __init__(self, weight=torch.tensor([0.01, 0.99]).cuda(), gamma=3, reduction='mean'):
        super(FocalLoss, self).__init__(weight, reduction=reduction)
        self.gamma = gamma
        self.weight = weight

    def forward(self, input, target):

        ce_loss = F.cross_entropy(
            input, target, reduction='none', weight=self.weight)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma * ce_loss).mean()
        return focal_loss


# %%
model = Skull_Unet_Transformer(8, 4, 2).cuda()
optimizer = torch.optim.AdamW(
    model.parameters(), lr=0.0002, betas=(0.1, 0.9))
#scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.95)
#model.load_state_dict(torch.load('./model/second/model_300.bin'))
#optimizer.load_state_dict(torch.load('./model/opt.bin'))
#scheduler.load_state_dict(torch.load('./model/sch.bin'))
best_f1 = 0.0
case_criterion = nn.BCELoss()
slice_criterion = nn.BCELoss()
total_epoch=0
# %%
fracture_criterion = FocalLoss(torch.tensor([0.15, 0.85]).cuda(), gamma=3)
#fracture_criterion=nn.CrossEntropyLoss(weight=torch.tensor([0.01,0.99]).cuda())
# %%
for n in range(100):
    case_acc = 0.0
    slice_acc = 0.0
    mask_acc = 0.0
    train_loss = 0.0
    fracture_n = 0
    tp = 0
    p = 0
    fn = 0
    model.train()
    train = tqdm(train_dataloader)
    for num, (img, label, mask) in enumerate(train):
        diagnosis, slice_diagnosis, fracture_mask = model(img.cuda())
        case_loss = case_criterion(diagnosis.squeeze(-1), torch.tensor(
            abs(label[:, 0]), dtype=torch.float).cuda())
        slice_loss = slice_criterion(
            slice_diagnosis.squeeze(-1), torch.clamp(label, 0, 1).cuda())
        fracture_loss = fracture_criterion(
            fracture_mask, torch.tensor(mask, dtype=torch.int64).cuda())
        #total_loss = case_loss+slice_loss+fracture_loss*20
        total_loss = fracture_loss*100
        train_loss += total_loss.item()
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        case_acc += (torch.round(diagnosis.detach().cpu())
                     == abs(label[:, 0])).numpy().mean()

        slice_acc += (torch.round(slice_diagnosis.squeeze(-1).detach().cpu())
                      == torch.clamp(label, 0, 1)).numpy().mean()
        fracture_slice = torch.clamp(label, 0, 1) == 1
        if fracture_slice.sum() != 0:
            f_mask = torch.max(fracture_mask.detach().cpu(), 1)[
                1][fracture_slice]
            mask = mask[fracture_slice]
            mask_acc += (f_mask == mask).numpy().mean()
            fracture_n += 1
            # f1 =tp/2tp+fp+fn=tp/tp+p+fn
            tp += (mask[f_mask == 1] == 1).numpy().sum()
            p += f_mask.numpy().sum()
            fn += (f_mask[mask == 1] == 0).numpy().sum()
            f1 = 2*tp/(tp+p+fn)
        try:
            train.set_postfix(
                {'n': n, 'loss': f'{train_loss/(num+1): .4f}', 's_acc': f'{slice_acc/(num+1):.4f}', 'c_acc': f'{case_acc/(num+1): .4f}','mask_acc': f'{mask_acc/fracture_n:.4f}','pre': f'{tp/p:.4f}', 'rec': f'{tp/(tp+fn):.4f}'})
        except ZeroDivisionError:
            train.set_postfix(
                {'n': n, 'loss': f'{train_loss/(num+1): .4f}', 's_acc': f'{slice_acc/(num+1):.4f}', 'c_acc': f'{case_acc/(num+1): .4f}'})

    case_acc = 0.0
    slice_acc = 0.0
    mask_acc = 0.0
    train_loss = 0.0
    fracture_n = 0
    tp = 0
    p = 0
    fn = 0
    model.eval()
    
    val = tqdm(val_dataloader)
    with torch.no_grad():
        for num, (img, label, mask) in enumerate(val):
            diagnosis, slice_diagnosis, fracture_mask = model(img.cuda())
            case_acc += (torch.round(diagnosis.detach().cpu())
                         == abs(label[:, 0])).numpy().mean()

            slice_acc += (torch.round(slice_diagnosis.squeeze(-1).detach().cpu())
                          == torch.clamp(label, 0, 1)).numpy().mean()
            fracture_slice = torch.clamp(label, 0, 1) == 1
            if fracture_slice.sum() != 0:
                f_mask = torch.max(fracture_mask.detach().cpu(), 1)[
                    1][fracture_slice]
                mask = mask[fracture_slice]
                mask_acc += (f_mask == mask).numpy().mean()
                fracture_n += 1
                tp += (mask[f_mask == 1] == 1).numpy().sum()
                p += f_mask.numpy().sum()
                fn += (f_mask[mask == 1] == 0).numpy().sum()
                f1 = 2*tp/(tp+p+fn)
            try:
                val.set_postfix(
                    {'n': n,'s_acc': f'{slice_acc/(num+1):.4f}', 'c_acc': f'{case_acc/(num+1): .4f}','mask_acc': f'{mask_acc/fracture_n:.4f}', 'pre': f'{tp/p:.4f}', 'rec': f'{tp/(tp+fn):.4f}'})
            except ZeroDivisionError:
                val.set_postfix(
                    {'n': n, 's_acc': f'{slice_acc/(num+1):.4f}', 'c_acc': f'{case_acc/(num+1): .4f}'})
    if best_f1 < f1:
        best_f1 = f1
    torch.save(model.state_dict(), './model/final/model_200.bin')
    torch.save(optimizer.state_dict(), './model/opt.bin')
    #torch.save(scheduler.state_dict(), './model/sch.bin')
    total_epoch+=1
    #pos_weight=np.max([0.5,0.9-total_epoch*0.010])
    #fracture_criterion = FocalLoss(torch.tensor([1-pos_weight, pos_weight],dtype=torch.float32).cuda())
    
   # scheduler.step()
# %%
torch.save(model.state_dict(), './model/test.bin')
torch.save(optimizer.state_dict(), './model/opt.bin')
torch.save(scheduler.state_dict(), './model/sch.bin')

# %%
# model.load_state_dict(torch.load('./model/model200.bin'))
img, label, mask = train_data[3]
img = img.unsqueeze(dim=0)
slice_id = 15
print(label[slice_id])
plt.subplot(1, 3, 1)
plt.imshow(img[0][slice_id][0], cmap='gray')
plt.axis('off')
plt.subplot(1, 3, 2)
model.eval()
with torch.no_grad():
    diagnosis, slice_diagnosis, fracture_mask = model(img.cuda())
    plt.imshow(torch.max(fracture_mask.detach().cpu(), 1)[1]
               [0][slice_id], cmap='binary')
    # print(diagnosis)
    # print((torch.max(fracture_mask.detach().cpu(), 1)[1]
    #    ).numpy().mean())
plt.axis('off')
plt.subplot(1, 3, 3)
plt.imshow(mask[slice_id], cmap='binary')
plt.axis('off')
# plt.savefig(f'{slice_id}.png')
# %%
print(np.argwhere(torch.max(fracture_mask[0], 1)[1].cpu().numpy() == 1))
print(np.argwhere(mask.numpy() == 1))
# %%

test_data = SkullDataset('./skull', 'test')
# %%
batchsize = 1
test_dataloader = torch.utils.data.DataLoader(
    test_data, batch_size=batchsize, shuffle=False)
# %%
# model = Skull_Unet_Transformer(8, 4, 2).cuda()
# model.load_state_dict(torch.load('./model/model800.bin'))
# model.eval()
models = [Skull_Unet_Transformer(8, 4, 2).cuda() for _ in range(1)]
name = ['900']
for i, m in enumerate(models):
    m.load_state_dict(
        torch.load('./model/model_'+name[i]+'.bin'))
    m.eval()
# %%
sliding_width = 11
avg = nn.AvgPool2d(5, 1, padding=2)
test = tqdm(test_dataloader)
with open('./prediction.csv', 'w') as file:
    file.write('id,label,coords\n')
    for idx, img in enumerate(test):
        fname = os.listdir(os.path.join(
            test_data.path, test_data.mode, test_data.study_list[idx]))
        with torch.no_grad():
            diagnosis = 0
            slice_diagnosis = 0
            fracture_mask = 0
            with torch.no_grad():
                for m in models:
                    d, s, f = m(img.cuda())
                    diagnosis += d/len(models)
                    slice_diagnosis += s/len(models)
                    fracture_mask += f/len(models)
            case_diagnosis = torch.round(diagnosis.detach().cpu())
            slice_diagnosis = torch.round(
                slice_diagnosis.squeeze().detach().cpu())
            f_mask = torch.max(fracture_mask, 1)[1][0]
        for slice_num in range(len(fname)):
            slice_diag = int(slice_diagnosis[slice_num])

            if case_diagnosis:
                if slice_diag == 0:

                    slice_diag = -1
                else:
                    # c_f_mask = avg(torch.tensor(
                    # f_mask[slice_num].unsqueeze(0), dtype=torch.float32))
                    c_f_mask = torch.tensor(
                        f_mask[slice_num].unsqueeze(0), dtype=torch.float32)
                    coords = np.argwhere(c_f_mask[0].cpu().numpy() == 1)

                    coords = torch.tensor(coords)
                    boxes = torch.stack([coords[:, 0]-sliding_width//2, coords[:, 1]-sliding_width//2,
                                         coords[:, 0]+sliding_width//2, coords[:, 1]+sliding_width//2], dim=1)
                    if boxes.shape[0] == 0:
                        slice_diag = -1
            else:
                slice_diag = 0
            file.write(f'{fname[slice_num][:-4]},{slice_diag},')
            if slice_diag == 1:

                reduced_boxes = nms(
                    boxes.detach().cpu().float(), c_f_mask[0, coords[:, 0], coords[:, 1]].detach().cpu(), 0.5)

                if len(reduced_boxes > 10):
                    reduced_boxes = reduced_boxes[:10]
                for boxid in reduced_boxes.numpy():
                    file.write(f'{boxes[boxid][0]} {boxes[boxid][1]} ')
            file.write('\n')
# %%
train_data = SkullDataset('./skull', 'train')
# %%
models = [Skull_Unet_Transformer(8, 4, 2).cuda() for _ in range(5)]
name = ['300', '900']
for i, m in enumerate(models):
    m.load_state_dict(
        torch.load('./model/model_'+name[i]+'.bin'))
    m.eval()
# %%
batchsize = 1
train_dataloader = torch.utils.data.DataLoader(
    train_data, batch_size=batchsize, shuffle=True)
# %%
train = tqdm(train_dataloader)
case_acc = 0.0
slice_acc = 0.0
mask_acc = 0.0
for num, (img, label, mask) in enumerate(train):
    diagnosis = 0
    slice_diagnosis = 0
    fracture_mask = 0
    with torch.no_grad():
        for m in models:
            d, s, f = m(img.cuda())
            diagnosis += d/len(models)
            slice_diagnosis += s/len(models)
            fracture_mask += f/len(models)
    case_acc += (torch.round(diagnosis.detach().cpu())
                 == abs(label[:, 0])).numpy().mean()
    slice_acc += (torch.round(slice_diagnosis.squeeze(-1).detach().cpu())
                  == torch.clamp(label, 0, 1)).numpy().mean()
    mask_acc += (torch.max(fracture_mask.detach().cpu(), 1)[1]
                 == mask).numpy().mean()
    train.set_postfix(
        {'slice_acc': f'{slice_acc/(num+1):.4f}', 'case_acc': f'{case_acc/(num+1): .4f}', 'mask_acc': f'{mask_acc/(num+1): .4f}'})

# %%
