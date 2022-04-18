# %%
import torchvision
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import os
from torch.nn.modules.pooling import AvgPool2d
from torchvision.transforms import transforms
from PIL import Image
from tqdm import tqdm
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision.ops import nms
import random
import torchvision.transforms.functional as TF
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection import FasterRCNN
# %%
MAXLENGTH = 47


class SkullDataset(torch.utils.data.Dataset):
    def __init__(self, path, mode, low, high,m):
        self.path = path
        self.mode = m
        if m == 'train':
            with open(os.path.join(path, 'records_train.json')) as f:
                self.labels = json.load(f)['datainfo']
            self.label = []
            self.bboxes = []
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ]
        )
        self.study_list = sorted(os.listdir(
            os.path.join(path, mode)))[low:high]
        self.imgs = []
        for case_id in tqdm(self.study_list):
            bboxes = []
            for slice_file in os.listdir(os.path.join(self.path, mode, case_id)):
                if m=='train':
                    if self.labels[slice_file[:-4]]['label'] != 1:
                        continue
                    for coor in self.labels[slice_file[:-4]]['coords']:
                        margin = 10
                        box = (coor[0]-margin, coor[1]-margin,
                            coor[0]+margin, coor[1]+margin)
                        bboxes.append(box)
                        self.bboxes.append(torch.tensor(
                            bboxes, dtype=torch.float32))
                img = np.load(os.path.join(
                    self.path, mode, case_id, slice_file))
                img = np.clip((img+1024)/4095, 0, 255)
                img = Image.fromarray(img)
                img = self.transform(img)

                self.imgs.append(img)

    def __getitem__(self, idx):
        if self.mode == 'train':

            #angle = random.choice([-30, -15, 0, 15, 30])
            #img = TF.rotate(self.data[idx], angle, fill=-1)
            #mask = TF.rotate(self.mask[idx], angle)
            target = {}
            target['boxes'] = self.bboxes[idx]
            target['labels'] = torch.ones(
                (self.bboxes[idx].shape[0],), dtype=torch.int64)
            target['iscrowd'] = torch.zeros(
                (self.bboxes[idx].shape[0],), dtype=torch.int64)
            target['image_id'] = torch.tensor([idx])
            target['area'] = torch.tensor(100, dtype=torch.float32)
            return self.imgs[idx], target
        else:
            return self.imgs[idx],None

    def __len__(self):
        return len(self.imgs)


# %%
train_data = SkullDataset('./skull', 'train', 0, 20)
#val_data = SkullDataset('./skull', 'train', 1110, 1116)
# %%
batchsize = 2


def collate_fn(batch):
    return tuple(zip(*batch))


train_dataloader = torch.utils.data.DataLoader(
    train_data, batch_size=batchsize, shuffle=True, collate_fn=collate_fn)
# %%
val_dataloader = torch.utils.data.DataLoader(
    val_data, batch_size=batchsize, shuffle=False, collate_fn=collate_fn)

# %%
device = 'cuda'
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
    pretrained=True).to(device)
optimizer = torch.optim.AdamW(
    model.parameters(), lr=0.0002, betas=(0.9, 0.99))
best_f1 = 0.0
# %%
schduler=torch.optim.lr_scheduler.ExponentialLR(optimizer,0.9)
# %%
num_classes = 2
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes).cuda()
model.load_state_dict(torch.load('./model/final/test.bin'))
# %%
for n in range(2):
    case_acc = 0.0
    slice_acc = 0.0
    mask_acc = 0.0
    train_loss = 0.0
    fracture_n = 0
    model.train()
    train = tqdm(train_dataloader)
    for num, (img, targets) in enumerate(train):

        img = list(i.to(device) for i in img)
        target = [{k: v.to(device) for k, v in t.items()} for t in targets]
        loss_dict = model(img, target)
        losses = sum(loss for loss in loss_dict.values())
        train_loss += losses.item()
        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        train.set_postfix(
            {'n': n, 'loss': f'{train_loss/(num+1): .4f}'})
    schduler.step()
    torch.save(model.state_dict(), './model/final/test.bin')

# %%
torch.save(model.state_dict(), './model/test.bin')
torch.save(optimizer.state_dict(), './model/opt.bin')
torch.save(schduler.state_dict(), './model/sch.bin')

# %%
# model.load_state_dict(torch.load('./model/model200.bin'))
img, gt_bboxes =next(iter(train_dataloader))
img = list(i.to(device) for i in img)
# %%
slice_id = 15
plt.subplot(1, 3, 1)
idx=0
plt.imshow(img[idx][0].detach().cpu().numpy(), cmap='gray')
plt.axis('off')
plt.subplot(1, 3, 2)
model.eval()
with torch.no_grad():
    outputs = model(img)
    boxes=outputs[idx]['boxes'][:10].detach().cpu().numpy()
    scores=outputs[idx]['scores'].detach().cpu().numpy()
    plt.imshow(np.zeros((512,512)),cmap='binary')
    plt.plot([(boxes[:,0]+boxes[:,2])/2],[(boxes[:,1]+boxes[:,3])/2],'o')
    print(scores)
plt.axis('off')
plt.subplot(1, 3, 3)
gt_boxes=gt_bboxes[idx]['boxes']
plt.imshow(np.zeros((512,512)),cmap='binary')
plt.plot([(gt_boxes[:,0]+gt_boxes[:,2]).detach().cpu().numpy()/2],[(gt_boxes[:,1]+gt_boxes[:,3]).detach().cpu().numpy()/2],'o')
plt.axis('off')
# %%
test_data = SkullDataset('./skull', 'test', 0, 130,'test')
def collate_fn(batch):
    return tuple(zip(*batch))
test_dataloader = torch.utils.data.DataLoader(
    test_data, batch_size=1, shuffle=False, collate_fn=collate_fn)
# %%
device = 'cuda'
model = torchvision.models.detection.fasterrcnn_resnet50_fpn().to(device)
# %%
num_classes = 2
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes).cuda()
model.load_state_dict(torch.load('./model/final/test.bin'))
model.eval()
# %%
with open('./voting_baseline.csv', 'w') as file:
    with open('./voting.csv','r') as slice_pred:
        file.write('id,label,coords\n')
        for slice_prediction,(test_img,_) in tqdm(zip(slice_pred.readlines()[1:],test_dataloader)):
            if slice_prediction.split(',')[1].strip()=='-1':
                test_img = list(i.cuda() for i in test_img)
                file.write(slice_prediction.strip().split(',')[0])
                file.write(',')

                with torch.no_grad():
                    outputs = model(test_img)
                    boxes=outputs[0]['boxes'].detach()#.cpu().numpy()
                    scores=outputs[0]['scores'].detach()#.cpu().numpy()
                boxes_id= nms(boxes+torch.tensor([-5,-5,+5,+5]).cuda(),scores, 0.5).cpu().numpy()
                boxes=boxes[boxes_id].cpu().numpy()
                scores=scores[:len(boxes)].cpu().numpy()
                if len(boxes>10):
                    boxes=boxes[:10]
                elif len(boxes)==0 :
                    file.write('-1,\n')
                    continue
                threshold=0.93
                if scores[0]<threshold:
                    file.write('-1,\n')
                    continue
                file.write('1,')
                for box,score in zip(boxes,scores):
                    if score>threshold:
                        file.write(f'{int((box[:][0]+box[:][2])//2)} {int((box[:][1]+box[:][3])//2)} ')
                file.write('\n')
            else:
                file.write(slice_prediction)
# %%
