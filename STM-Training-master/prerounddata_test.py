import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from torch.utils.data import Dataset

from libs.dataset.transform import TrainTransform, TestTransform

import os
import math
import numpy as np
from PIL import Image
import glob

def multibatch_collate_fn(batch):

    min_time = min([sample[0].shape[0] for sample in batch])
    frames = torch.stack([sample[0] for sample in batch])
    masks = torch.stack([sample[1] for sample in batch])

    objs = [sample[2] for sample in batch]

    try:
        info = [sample[3] for sample in batch]
    except IndexError as ie:
        info = None

    return frames, masks, objs, info


class BaseData(Dataset):

    def increase_max_skip(self):
        pass

    def set_max_skip(self):
        pass

class PreRoundData(BaseData):

    def __init__(self, train=True, sampled_frames=3, 
        transform=None, max_skip=5, increment=5, samples_per_video=12):
        
        data_dir = os.path.join(ROOT, 'PreRoundData')

        self.image_dir = os.path.join(data_dir, 'JPEGImages')
        self.mask_dir = os.path.join(data_dir, 'Annotations')
        _imset_f = os.path.join(data_dir, 'ImageSets', 'train1.txt')

        self.root = data_dir
        self.max_obj = 0

        
        self.videos = []
        self.num_frames = {}
        self.num_objects = {}
        self.shape = {}
        # extract annotation information
        with open(os.path.join(_imset_f), "r") as lines:
            # 遍历每一行文件名
            for line in lines:
                _video = line.rstrip('\n') # 除去换行符，只留下文件名
                self.videos.append(_video) # 将文件名添加进文件名列表
                self.num_frames[_video] = len(glob.glob(os.path.join(self.image_dir, _video, '*.jpg'))) # 索引每一帧图像并添加进帧字典
                # P(调色板)模式：读到的图片每个像素的值是一个索引,映射到其对应的调色板上(调色板长度一般为256*3)
                _mask = np.array(Image.open(os.path.join(self.mask_dir, _video, '00001.png')).convert("P")) 
                self.num_objects[_video] = np.max(_mask) # np.max求序列中元素的最大值
                self.max_obj = max(self.num_objects[_video], self.max_obj)
                self.shape[_video] = np.shape(_mask)


        self.samples_per_video = samples_per_video
        self.sampled_frames = sampled_frames
        self.length = samples_per_video * len(self.videos)
        self.max_skip = max_skip
        self.increment = increment
        
        self.transform = transform
        self.train = train

    def increase_max_skip(self):
        self.max_skip = min(self.max_skip + self.increment, MAX_TRAINING_SKIP)

    def set_max_skip(self, max_skip):
        self.max_skip = max_skip

    def To_onehot(self, mask):
        M = np.zeros((self.samples_per_video, mask.shape[0], mask.shape[1]), dtype=np.uint8)
        for k in range(self.samples_per_video):
            M[k] = (mask == k).astype(np.uint8)
        return M
    
    def All_to_onehot(self, masks):
        Ms = np.zeros((self.samples_per_video, masks.shape[0], masks.shape[1], masks.shape[2]), dtype=np.uint8)
        for n in range(masks.shape[0]):
            Ms[:,n] = self.To_onehot(masks[n])
        return Ms

    def __len__(self):

        return self.length

    def __getitem__(self, index):
        video = self.videos[index]
        info = {}
        info['name'] = video
        info['num_frames'] = self.num_frames[video]
        info['size'] = self.shape[video]

        N_frames = np.empty((self.num_frames[video],) + self.shape[video] + (3,), dtype=np.float32) # (视频帧数,width,height,rgb)
        N_masks = np.empty((self.num_frames[video],) + self.shape[video], dtype=np.uint8) # (视频帧数,width,height)

        for i in range(1, self.num_frames[video]):
            img_file = os.path.join(self.image_dir, video, '{:05d}.jpg'.format(i)) # /1 -> /00001
            N_frames[i] = np.array(Image.open(img_file).convert('RGB'))/255.
            try:
                mask_file = os.path.join(self.mask_dir, video, '{:05d}.png'.format(i))  
                N_masks[i] = np.array(Image.open(mask_file).convert('P'), dtype=np.uint8)
            except:
                N_masks[i] = 255

        Fs = torch.from_numpy(np.transpose(N_frames.copy(), (3, 0, 1, 2)).copy()).float()

        Ms = torch.from_numpy(self.All_to_onehot(N_masks).copy()).float()
        num_objects = torch.LongTensor([int(self.num_objects[video])])
        '''
        if self.transform is None:
            raise RuntimeError('Lack of proper transformation')
        Fs, Ms = self.transform(Fs, Ms, False)
        '''
        return Fs, Ms, num_objects, info





ROOT = 'C:\\Users\\Admin\\Desktop\\data'

DATA_CONTAINER = {}
DATA_CONTAINER['PreRoundData'] = PreRoundData

# 设置训练的 max skip time length 
sampled_frames = 1
train_transformer = None
#train_transformer = TrainTransform(size=(720, 1280)) # 训练数据处理中包含了多种增强策略
max_skip = 5
samples_per_video = 1 

trainset = DATA_CONTAINER['PreRoundData'](
    train=True, 
    sampled_frames=sampled_frames, 
    transform=train_transformer, 
    max_skip=max_skip, 
    samples_per_video=samples_per_video
    )

''' Load data '''
trainloader = data.DataLoader(trainset, batch_size=1, shuffle=True, num_workers=0,
                              collate_fn=multibatch_collate_fn)
for index, patchdata in enumerate(trainloader):
    print(index)
    print(len(patchdata))
    print(type(patchdata[0])) 
    print(patchdata[0].size())
    print(type(patchdata[1])) 
    print(patchdata[1].size())
    print(type(patchdata[2])) 
    print(patchdata[2])
    print(type(patchdata[3])) 
    print(patchdata[3])
