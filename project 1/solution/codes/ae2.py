##########################################
# Stat231 Project 1:
# Autoencoder
# Author: Xiaohan Wang
##########################################

import torch
import torch.nn as nn
import torch.optim as optim


class appearance_autoencoder(nn.Module):
    def __init__(self):
        super(appearance_autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            # TODO: Fill in the encoder structure
            nn.Conv2d(3, 16, kernel_size=5, stride=2, padding=2), # conv1 -> 64 x 64 x 16
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1), # conv2 -> 32 x 32 x 32
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1), # conv3 -> 16 x 16 x 64
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1), # conv4 -> 8 x 8 x 128
            nn.ReLU(),
        )

        self.fc1 = nn.Sequential(                    
            # TODO: Fill in the FC layer structure
            nn.Linear(128 * 8 * 8, 50), # fc
            nn.ReLU(),
        )

        self.fc2 = nn.Sequential(                    
            # TODO: Fill in the FC layer structure
            nn.Linear(50, 128 * 8 * 8), # fc
            nn.ReLU(),
        )

        self.decoder = nn.Sequential(
            # TODO: Fill in the decoder structure
            # Hint: De-Conv in PyTorch: ConvTranspose2d 
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1), # deconv1 -> 16 x 16 x 64
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1), # deconv2 -> 32 x 32 x 32
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1), # deconv3 -> 64 x 64 x 16
            nn.ReLU(),
            nn.ConvTranspose2d(16, 3, kernel_size=5, stride=2, padding=2, output_padding=1), # deconv4 -> 128 x 128 x 3
            nn.Sigmoid(),
        )

    def forward(self, x, only_decoder):
        # TODO: Fill in forward pass
        if not only_decoder:
            x_encoder = self.encoder(x)   # (N, 8, 8, 128)
            x_reshape = x_encoder.view(x_encoder.size(0), -1)  # (N, 8 * 8 * 128)
            x = self.fc1(x_reshape)  # (N, 50)
        x_fcd = self.fc2(x)   # (N, 8 * 8 * 128)
        x_reshape2 =  x_fcd.view(x_fcd.size(0), 128, 8, 8)  # (N, 8, 8, 128)
        x_recon = self.decoder(x_reshape2)
        return x, x_recon


class landmark_autoencoder(nn.Module):
    def __init__(self):
        super(landmark_autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            # TODO: Fill in the encoder structure
            nn.Linear(68 * 2, 100),
            nn.ReLU(),
            nn.Linear(100, 10),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            # TODO: Fill in the decoder structure
            nn.Linear(10, 100),
            nn.ReLU(),
            nn.Linear(100, 68 * 2),
            nn.Sigmoid(),
        )

    def forward(self, x, only_decoder):
        # TODO: Fill in forward pass
        if not only_decoder:
            x_flatten = x.view(x.size(0), -1)
            x = self.encoder(x_flatten)
        x_decoder = self.decoder(x)
        x_recon = x_decoder.view(x.size(0), 68, 2)
        return x, x_recon


class autoencoder(object):
    def __init__(self, appear_lr, landmark_lr, use_cuda):
        self.appear_model = appearance_autoencoder()
        self.landmark_model = landmark_autoencoder()
        self.use_cuda = use_cuda
        if use_cuda:
            self.appear_model.cuda()
            self.landmark_model.cuda()
        self.criterion = nn.MSELoss()
        self.appear_optim = optim.Adam(self.appear_model.parameters(), lr=appear_lr)
        self.landmark_optim = optim.Adam(self.landmark_model.parameters(), lr=landmark_lr)
        
    def train(self, model, epoch, trainloader, use_cuda, optim):
        all_encoders = []
        for batch_idx, x in enumerate(trainloader):
            if use_cuda:
                x = x.cuda()
            
            optim.zero_grad()
            encoders, output = model(x, False)
            loss = self.criterion(output, x)
            loss.backward()
            optim.step()

            for encoder in encoders:
                all_encoders.append(encoder)

        print('Training Epoch: {}'.format(epoch))
        return all_encoders


    def test(self, model, testloader, use_cuda):
        recons = None
        for batch_idx, x in enumerate(testloader):
            if use_cuda:
                x = x.cuda()
            _, output = model(x, True)  # nx3x128x128
            recons = output[1]  # only check result on first image
            break  

        return recons


    def train_appear_model(self, epochs, trainloader, use_cuda):
        self.appear_model.train()
        epoch = 0
        # TODO: Train appearance autoencoder
        all_encoders = []
        while epoch < epochs:
            epoch += 1
            all_encoders = self.train(self.appear_model, epoch, trainloader, use_cuda, self.appear_optim)
        return all_encoders

    def train_landmark_model(self, epochs, trainloader, use_cuda):
        self.landmark_model.train()
        epoch = 0
        # TODO: Train landmark autoencoder
        all_encoders = []
        while epoch < epochs:
            epoch += 1
            all_encoders = self.train(self.landmark_model, epoch, trainloader, use_cuda, self.landmark_optim)
        return all_encoders

    def test_appear_model(self, testloader, use_cuda):
        self.appear_model.eval()
        # TODO: Test appearance autoencoder
        recons_face = self.test(self.appear_model, testloader, use_cuda)
        return recons_face
    
    def test_landmark_model(self, testloader, use_cuda):
        self.landmark_model.eval()
        # TODO: Test landmark autoencoder
        recons_landmark = self.test(self.landmark_model, testloader, use_cuda)
        return recons_landmark

