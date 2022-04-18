# %%
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import os
from torchvision.transforms import transforms
from PIL import Image
from tqdm import tqdm
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision.ops import nms
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection import FasterRCNN
import torchvision
# %%

MAXLENGTH = 47


class SkullDataset(torch.utils.data.Dataset):
    def __init__(self, path, mode,low,high):
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
        self.study_list = sorted(os.listdir(os.path.join(path, mode)))[low:high]
        self.data = []
        for case_id in tqdm(self.study_list):
            imgs = []
            masks = []
            labels = []
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
            for i in range(47-len(imgs)):
                imgs.append(-torch.ones(1, 512, 512))
                if mode == 'train':
                    masks.append(torch.zeros(1, 512, 512))
                    if labels[0] == 0:
                        labels.append(0)
                    else:
                        labels.append(-1)
            self.data.append(torch.stack(imgs, dim=0))
            if mode == 'train':
                self.label.append(torch.tensor(labels, dtype=torch.float32))
                self.mask.append(torch.cat(masks, dim=0))

    def __getitem__(self, idx):
        if self.mode == 'train':
            return self.data[idx], self.label[idx], self.mask[idx]
        else:
            return self.data[idx]

    def __len__(self):
        return len(self.study_list)


# %%


class PositionalEmbedding1D(nn.Module):

    def __init__(self, seq_len, dim):
        super().__init__()
        self.pos_embedding = nn.Parameter(torch.zeros(1, seq_len, dim))

    def forward(self, x):
        return x + self.pos_embedding


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
        self.up_list.append(nn.Conv2d(in_ch, num_class, 1))
        trans_encode_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=8, dim_feedforward=d_model, dropout=0.1, batch_first=True)
        self.InterTransformer = nn.TransformerEncoder(
            trans_encode_layer, num_layers=trans_layers)
        self.inter_pos_embedding = PositionalEmbedding1D(
            48, d_model)
        self.diagnosis_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.diagnosis_fc = nn.Linear(d_model, 1)
        self.slice_diagnosis_fc = nn.Linear(d_model, 1)
        self.sigmoid = nn.Sigmoid()
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
        diagnosis = self.diagnosis_fc(transformer_out[:, 0, :])
        diagnosis = self.sigmoid(diagnosis)
        slice_diagnosis = self.slice_diagnosis_fc(
            transformer_out[:, 1:, :]) # b l 1
        slice_diagnosis = self.sigmoid(slice_diagnosis)
        # b l c w h
        out_x[-1] = transformer_out[:, 1:,
                                    :].reshape(out_x[-1].shape)  # b*l c w h
        for i, up_layer in enumerate(self.up_list):
            if i != (len(self.up_list)-1):
                out_x[-1] = up_layer(out_x[-1], out_x[-i-2])
            else:
                out_x[-1] = up_layer(out_x[-1])

        return diagnosis, slice_diagnosis, out_x[-1].reshape(x.shape[0], MAXLENGTH, 2, 512, 512).permute(0, 2, 1, 3, 4)


class FocalLoss(nn.modules.loss._WeightedLoss):
    def __init__(self, weight=torch.tensor([0.01, 0.99]).cuda(), gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__(weight, reduction=reduction)
        self.gamma = gamma
        self.weight = weight

    def forward(self, input, target):

        ce_loss = F.cross_entropy(
            input, target, reduction=self.reduction, weight=self.weight)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma * ce_loss).mean()
        return focal_loss


# %%

test_data = SkullDataset('./skull', 'test')
# %%
batchsize = 1
test_dataloader = torch.utils.data.DataLoader(
    test_data, batch_size=batchsize, shuffle=False)
models = [Skull_Unet_Transformer(8, 4, 2).cuda() for _ in range(4)]
name = ['300','600','900','1200']
for i, m in enumerate(models):
    m.load_state_dict(
        torch.load('./model/model_'+name[i]+'.bin'))
    m.eval()
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
    pretrained=True).cuda()
num_classes = 2
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes).cuda()
model.load_state_dict(torch.load('./model/final/fastrcnn.bin'))
model.eval()
# %%
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
                    d, s, _ = m(img.cuda())
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
                    img = list(i.cuda() for i in img)
                    file.write(f'{fname[slice_num][:-4]},{slice_diag},')

                    outputs = model(img)
                    boxes=outputs[idx]['boxes'].detach().cpu().numpy()
                    scores=outputs[idx]['scores'].detach().cpu().numpy()
                    
                    #reduced_boxes = nms( boxes,scores, 0.5)
                    if len(boxes>10):
                        boxes=boxes[:10]
                    for boxid in boxes.numpy():
                        file.write(f'{(boxes[:][0]+boxes[:][2])//2} {(boxes[:][1]+boxes[:][3])/2} ')
            else:
                slice_diag=0
                file.write(f'{fname[slice_num][:-4]},{slice_diag},')
            file.write('\n')