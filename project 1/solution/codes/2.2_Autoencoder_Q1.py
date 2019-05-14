##########################################
# Stat231 Project 1:
# Autoencoder
# Author: Xiaohan Wang
##########################################

import os
import numpy as np
import argparse
import matplotlib.pyplot as plt
import skimage
from skimage import io, transform, img_as_ubyte
import scipy.io as sio
from easydict import EasyDict as edict

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from PIL import Image

from ae import autoencoder
import mywarper

# parser = argparse.ArgumentParser(description='stat231_project1')
# parser.add_argument('--epochs', type=int, default=300)
# parser.add_argument('--batch_size', type=int, default=100)
# parser.add_argument('--seed', type=int, default=12345)
# parser.add_argument('--device', type=int, default=0)
# parser.add_argument('--image_dir', type=str, default='./data/images/')
# parser.add_argument('--landmark_dir', type=str, default='./data/landmarks/')
# parser.add_argument('--male_img_dir', type=str, default='./data/male_images/')
# parser.add_argument('--female_img_dir', type=str, default='./data/female_images/')
# parser.add_argument('--male_landmark', type=str, default='./data/male_landmarks/')
# parser.add_argument('--female_landmark', type=str, default='./data/female_landmarks/')
# parser.add_argument('--path', type=str, default='./results/model/')
# parser.add_argument('--log', type=str, default='./results/log/')
# parser.add_argument('--appear_lr', type=float, default=7e-4)
# parser.add_argument('--landmark_lr', type=float, default=1e-4)
args = edict({
    "image_dir": '/content/drive/My Drive/stat231/project1/dataset/images/',
    "landmark_dir": '/content/drive/My Drive/stat231/project1/dataset/landmarks/',
    "epochs": 350,
    "batch_size": 100,
    "device": 0,
    "appear_lr": 5e-4,
    "landmark_lr": 4e-4,
    "seed": 12345
})


# Read Dataset
class data_reader(object):
    def __init__(self, root_dir, file_str_len, origin_name, file_format):
        self.root_dir = root_dir
        self.file_str_len = file_str_len
        self.origin_name = origin_name
        self.file_format = file_format

    def read(self, split, read_type):
        files_len = len([name for name in os.listdir(self.root_dir) 
                        if os.path.isfile(os.path.join(self.root_dir, name))])
        counter = 0
        idx = counter
        dataset = []
        train_dataset = []
        test_dataset = []
        while counter < files_len:
            name = self.origin_name + str(idx)
            if len(name) > self.file_str_len:
                name = name[len(name)-self.file_str_len:]
            try:
                if read_type == 'image':
                    data = io.imread(self.root_dir + name + self.file_format)
                elif read_type == 'landmark':
                    mat_data = sio.loadmat(self.root_dir + name + self.file_format)

                    data = mat_data['lms']
                dataset.append(data)
                counter += 1
            except FileNotFoundError:
                pass
            idx += 1
        train_dataset = dataset[:split]
        test_dataset = dataset[split:]
        return train_dataset, test_dataset

# Construct Dataset
class ImgToTensor(object):
    def __call__(self, sample):
        sample = sample.transpose((2, 0, 1))
        return torch.tensor(sample, dtype=torch.float32)/255

class LandmarkToTensor(object):
    def __call__(self, sample):
        return torch.tensor(sample, dtype=torch.float32)/128

class dataset_constructor(Dataset):
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sample_data = self.dataset[idx]
        if self.transform:
            sample_data = self.transform(sample_data)
        return sample_data


def run_autoencoder():
    face_trainset = dataset_constructor(face_images_train_warped, transform=transforms.Compose([
                                                                    ImgToTensor()]))
    face_testset = dataset_constructor(face_images_test_warped, transform=transforms.Compose([
                                                                    ImgToTensor()]))
    face_trainloader = torch.utils.data.DataLoader(face_trainset, \
                                                    batch_size=args.batch_size, \
                                                    shuffle=True, \
                                                    num_workers=0)
    face_testloader = torch.utils.data.DataLoader(face_testset, \
                                                    batch_size=args.batch_size, \
                                                    shuffle=False, \
                                                    num_workers=0)

    landmark_trainset = dataset_constructor(face_landmark_train, transform=transforms.Compose([
                                                                    LandmarkToTensor()]))
    landmark_testset = dataset_constructor(face_landmark_test, transform=transforms.Compose([
                                                                    LandmarkToTensor()]))
    landmark_trainloader = torch.utils.data.DataLoader(landmark_trainset, \
                                                        batch_size=args.batch_size, \
                                                        shuffle=True, \
                                                        num_workers=0)
    landmark_testloader = torch.utils.data.DataLoader(landmark_testset, \
                                                        batch_size=args.batch_size, \
                                                        shuffle=False, \
                                                        num_workers=0)

    auto_encoder = autoencoder(args.appear_lr, args.landmark_lr, True)
    # ============ ae: train appearance model ============== #
    auto_encoder.train_appear_model(args.epochs, face_trainloader, True)

    # ============ ae: train landmark model ============== #
    auto_encoder.train_landmark_model(args.epochs, landmark_trainloader, True)

    # ============ ae: test appearance ============== #
    recons_face_mean_pos = auto_encoder.test_appear_model(face_testloader, True)

    # ============ ae: test landmark ============== #
    recons_landmark = auto_encoder.test_landmark_model(landmark_testloader, True)

    # ============ warp reconstructed face from mean to reconstructed landmarks ============== #
    plt.figure(figsize=(16, 27))
    plt.suptitle('First 20 Reconstructed Faces (Autoencoder)', fontsize=24, x=0.5, y=0.9)
    for i in range(20):
        cur_recons_face_mean_pos = np.transpose(recons_face_mean_pos[i].data.cpu().numpy(), (1, 2, 0))
        cur_recons_landmark = recons_landmark[i].data.cpu().numpy() * 128
        recons_face = mywarper.warp(cur_recons_face_mean_pos, mean_landmark, cur_recons_landmark)
        pos = 1 + i
        if i >= 15:
            pos = 16 + i
        elif i >= 10:
            pos = 11 + i
        elif i >= 5:
            pos = 6 + i
        plt.subplot(8, 5, pos)
        plt.axis('off')
        plt.title('Reconstructed Face ' + str(i+1), fontsize=12)
        plt.imshow(recons_face)
        # plot original image
        plt.subplot(8, 5, pos+5)
        plt.axis('off')
        plt.title('Original Face ' + str(i+1), fontsize=12)
        plt.imshow(face_images_test[i])
    plt.subplots_adjust(wspace=0.2, hspace=0)
    plt.show()

def main():
    run_autoencoder()

# args = parser.parse_args()
args.cuda = torch.cuda.is_available()
torch.cuda.set_device(args.device)

# if not os.path.exists(args.path):
#     os.makedirs(args.path)
# if not os.path.exists(args.log):
#     os.makedirs(args.log)

if args.cuda:
    torch.cuda.manual_seed(args.seed)

    
# ============ read original data ============== #
face_images_reader = data_reader(args.image_dir, 6, '000000', '.jpg')
face_images_train, face_images_test = face_images_reader.read(split=800, read_type='image')
face_landmark_reader = data_reader(args.landmark_dir, 6, '000000', '.mat')
face_landmark_train, face_landmark_test = face_landmark_reader.read(split=800, read_type='landmark')

# ============ calculate mean landmark of train set ============== #
mean_landmark = np.zeros((68, 2))
for i in range(800):
    landmark = face_landmark_train[i]
    mean_landmark += landmark
mean_landmark /= 800

# ============ warp train set and test set to mean position ============== #
face_images_train_warped = []
face_images_test_warped = []
for i in range(800):
    warped_face = mywarper.warp(face_images_train[i], face_landmark_train[i], mean_landmark)
    face_images_train_warped.append(warped_face)

for i in range(200):
    warped_face = mywarper.warp(face_images_test[i], face_landmark_test[i], mean_landmark)
    face_images_test_warped.append(warped_face)


if __name__ == "main":
    main() 
