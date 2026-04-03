from pathlib import Path
import random,cv2
import glob,os
from torch.utils.data import Dataset
import numpy as np
import os,glob
import torch
from torchvision import transforms
import sys



class MyDataset(Dataset):
    def __init__(self, input_rgb_path, input_depth_path):
        super(MyDataset, self).__init__()
        self.input_rgb_list = []
        self.input_depth_list = []
        self.rgb_name_list = []
        self.depth_name_list = []

        self.num = 0
        for _ in range(1):
            for i in os.listdir(input_rgb_path):
                input_rgb_img = input_rgb_path + i
                input_rgb_name = i
                self.input_rgb_list.append(input_rgb_img)
                self.rgb_name_list.append(input_rgb_name)
                input_depth_name = str(int(input_rgb_name[6:-4])) + '.png'
                input_depth_img = input_depth_path + input_depth_name
                self.input_depth_list.append(input_depth_img)
                self.depth_name_list.append(input_depth_name)
                self.num = self.num + 1

    def __len__(self):
        return self.num

    def __getitem__(self, idx):
        img_rgb = np.array(cv2.imread(self.input_rgb_list[idx]))
        img_depth = np.array(cv2.imread(self.input_depth_list[idx]))
        rgb_name = self.rgb_name_list[idx]
        depth_name = self.depth_name_list[idx]

        x = np.random.randint(0, img_rgb.shape[0] - 256)
        y = np.random.randint(0, img_rgb.shape[1] - 256)

        input_rgb_np = img_rgb[x:x + 256, y:y + 256, :].astype(np.float32).transpose(2, 0, 1) / 255.0
        input_depth_np = img_depth[x:x + 256, y:y + 256, :].astype(np.float32).transpose(2, 0, 1) / 255.0

        input_rgb_tensor = torch.from_numpy(input_rgb_np)
        input_depth_tensor = torch.from_numpy(input_depth_np)

        return input_depth_tensor, depth_name, input_rgb_tensor, rgb_name
        
        
        
class MyTestDataset(Dataset):
    def __init__(self, input_rgb_path, input_depth_path):
        super(MyTestDataset, self).__init__()
        self.input_rgb_list = []
        self.input_depth_list = []
        self.rgb_name_list = []
        self.depth_name_list = []

        self.num = 0
        for _ in range(1):
            for i in os.listdir(input_rgb_path):
                input_rgb_img = input_rgb_path + i
                input_rgb_name = i
                self.input_rgb_list.append(input_rgb_img)
                self.rgb_name_list.append(input_rgb_name)
                input_depth_name = str(int(input_rgb_name[6:-4])) + '.png'
                input_depth_img = input_depth_path + input_depth_name
                self.input_depth_list.append(input_depth_img)
                self.depth_name_list.append(input_depth_name)
                self.num = self.num + 1


    def __len__(self):
        return self.num

    def __getitem__(self, idx):
        img_rgb = np.array(cv2.imread(self.input_rgb_list[idx]))
        img_depth = np.array(cv2.imread(self.input_depth_list[idx]))
        rgb_name = self.rgb_name_list[idx]
        depth_name = self.depth_name_list[idx]


        input_rgb_np = img_rgb.astype(np.float32).transpose(2, 0, 1) / 255.0
        input_depth_np = img_depth.astype(np.float32).transpose(2, 0, 1) / 255.0

        input_rgb_tensor = torch.from_numpy(input_rgb_np)
        input_depth_tensor = torch.from_numpy(input_depth_np)

        return input_depth_tensor, depth_name, input_rgb_tensor, rgb_name




class NYUDataset(Dataset):
    def __init__(self, input_rgb_path, input_depth_path):
        super(NYUDataset, self).__init__()
        self.input_rgb_list = []
        self.input_depth_list = []
        self.rgb_name_list = []
        self.depth_name_list = []

        self.num = 0
        for _ in range(1):
            for i in os.listdir(input_rgb_path):
                input_rgb_img = input_rgb_path + i
                input_rgb_name = i
                self.input_rgb_list.append(input_rgb_img)
                self.rgb_name_list.append(input_rgb_name)
                input_depth_name = str(int(input_rgb_name[:-4])) + '.png'
                input_depth_img = input_depth_path + input_depth_name
                self.input_depth_list.append(input_depth_img)
                self.depth_name_list.append(input_depth_name)
                self.num = self.num + 1

    def __len__(self):
        return self.num

    def __getitem__(self, idx):
        img_rgb = np.array(cv2.imread(self.input_rgb_list[idx]))
        img_depth = np.array(cv2.imread(self.input_depth_list[idx]))
        rgb_name = self.rgb_name_list[idx]
        depth_name = self.depth_name_list[idx]
        x = np.random.randint(0, img_rgb.shape[0] - 256)
        y = np.random.randint(0, img_rgb.shape[1] - 256)
        input_rgb_np = img_rgb[x:x + 256, y:y + 256, :].astype(np.float32).transpose(2, 0, 1) / 255.0
        input_depth_np = img_depth[x:x + 256, y:y + 256, :].astype(np.float32).transpose(2, 0, 1) / 255.0
        input_rgb_tensor = torch.from_numpy(input_rgb_np)
        input_depth_tensor = torch.from_numpy(input_depth_np)
        return input_depth_tensor, depth_name, input_rgb_tensor, rgb_name



class NYUTestDataset(Dataset):
    def __init__(self, input_rgb_path, input_depth_path):
        super(NYUTestDataset, self).__init__()
        self.input_rgb_list = []
        self.input_depth_list = []
        self.rgb_name_list = []
        self.depth_name_list = []

        self.num = 0
        for _ in range(1):
            for i in os.listdir(input_rgb_path):
                input_rgb_img = input_rgb_path + i
                input_rgb_name = i
                self.input_rgb_list.append(input_rgb_img)
                self.rgb_name_list.append(input_rgb_name)
                input_depth_name = str(int(input_rgb_name[:-4])) + '.png'
                input_depth_img = input_depth_path + input_depth_name
                self.input_depth_list.append(input_depth_img)
                self.depth_name_list.append(input_depth_name)
                self.num = self.num + 1


    def __len__(self):
        return self.num

    def __getitem__(self, idx):
        img_rgb = np.array(cv2.imread(self.input_rgb_list[idx]))
        img_depth = np.array(cv2.imread(self.input_depth_list[idx]))
        rgb_name = self.rgb_name_list[idx]
        depth_name = self.depth_name_list[idx]
        input_rgb_np = img_rgb.astype(np.float32).transpose(2, 0, 1) / 255.0
        input_depth_np = img_depth.astype(np.float32).transpose(2, 0, 1) / 255.0
        input_rgb_tensor = torch.from_numpy(input_rgb_np)
        input_depth_tensor = torch.from_numpy(input_depth_np)
        return input_depth_tensor, depth_name, input_rgb_tensor, rgb_name

        
        
def save_checkpoint(state, epoch, is_best=False, log_dir=None, filename="ckpt.pth.tar"):
    save_file = os.path.join(log_dir, filename)
    print("save model in:", save_file)
    if epoch % 10 == 0:
        new_filename = "ckpt_" + str(epoch) + ".pth.tar"
        new_save_file = os.path.join(log_dir, new_filename)
        torch.save(state, new_save_file)
    if is_best:
        torch.save(state, os.path.join(log_dir, filename.replace(".pth.tar", ".best.pth.tar")))


class AverageMeter(object):

    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


def get_output_folder(parent_dir, env_name, output_current_folder=False):
    os.makedirs(parent_dir, exist_ok=True)
    experiment_id = 0
    for folder_name in os.listdir(parent_dir):
        if not os.path.isdir(os.path.join(parent_dir, folder_name)):
            continue
        try:
            folder_name = int(folder_name.split('-run')[-1])
            if folder_name > experiment_id:
                experiment_id = folder_name
        except:
            pass
    if not output_current_folder: 
        experiment_id += 1

    parent_dir = os.path.join(parent_dir, env_name)
    parent_dir = parent_dir + '-run{}'.format(experiment_id)
    os.makedirs(parent_dir, exist_ok=True)
    return parent_dir, experiment_id

class MinimalCrop:  
    def __init__(self, min_div=16):
        self.min_div = min_div
        
    def __call__(self, image):
        w, h = image.size
        
        h_new = h - (h % self.min_div)
        w_new = w - (w % self.min_div)
        
        if h_new == 0 and w_new == 0:
            return image
        else:    
            h_diff = h-h_new
            w_diff = w-w_new

            top = int(h_diff/2)
            bottom = h_diff-top
            left = int(w_diff/2)
            right = w_diff-left

            return image.crop((left, top, w-right, h-bottom))






