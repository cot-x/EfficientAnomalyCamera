import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import numpy as np

from torchvision import transforms
from torchvision.utils import save_image

from tqdm import tqdm
from pickle import load, dump

import os
import cv2
import random
import datetime
import argparse


from model import *


class Solver:
    def __init__(self, args):
        use_cuda = torch.cuda.is_available() if not args.cpu else False
        self.device = torch.device("cuda" if use_cuda else "cpu")
        torch.backends.cudnn.benchmark = True
        print(f'Use Device: {self.device}')
        
        def num_fmap(stage):
            base_size = self.args.image_size
            fmap_base = base_size * 4
            fmap_max = base_size // 2
            fmap_decay = 1.0
            return min(int(fmap_base / (2.0 ** (stage * fmap_decay))), fmap_max)
        
        self.args = args
        self.feed_dim = num_fmap(0)
        self.max_depth = int(np.log2(self.args.image_size)) - 1
        self.pseudo_aug = 0.0
        self.epoch = 0
        self.scorelist = []
        
        self.netG = Generator(self.max_depth, num_fmap).to(self.device)
        self.netD = Discriminator(self.max_depth, num_fmap).to(self.device)
        self.netE = Encoder(self.max_depth, num_fmap).to(self.device)
        
        self.netG.apply(self.weights_init)
        self.netD.apply(self.weights_init)
        self.netE.apply(self.weights_init)
        
        self.optimizer_G = optim.Adam(self.netG.parameters(), lr=self.args.lr, betas=(0, 0.9))
        self.optimizer_D = optim.Adam(self.netD.parameters(), lr=self.args.lr * self.args.mul_lr_dis, betas=(0, 0.9))
        self.optimizer_E = optim.Adam(self.netE.parameters(), lr=self.args.lr * self.args.mul_lr_dis, betas=(0, 0.9))
    
    def weights_init(self, module):
        if type(module) == nn.Linear or type(module) == nn.Conv2d or type(module) == nn.ConvTranspose2d:
            nn.init.kaiming_normal_(module.weight)
            if module.bias is not None:
                module.bias.data.fill_(0)
            
    def save_state(self):
        self.netG.cpu(), self.netD.cpu()
        torch.save(self.netG.state_dict(), os.path.join(self.args.weight_dir, f'weight_G.pth'))
        torch.save(self.netD.state_dict(), os.path.join(self.args.weight_dir, f'weight_D.pth'))
        self.netG.to(self.device), self.netD.to(self.device)
        
    def load_state(self):
        if (os.path.exists('weight_G.pth') and os.path.exists('weight_D.pth')):
            self.netG.load_state_dict(torch.load('weight_G.pth', map_location=self.device))
            self.netD.load_state_dict(torch.load('weight_D.pth', map_location=self.device))
            self.state_loaded = True
            print('Loaded network state.')
    
    def save_resume(self):
        with open(os.path.join('.', f'resume.pkl'), 'wb') as f:
            dump(self, f)
    
    def load_resume(self):
        if os.path.exists('resume.pkl'):
            with open(os.path.join('.', 'resume.pkl'), 'rb') as f:
                print('Load resume.')
                return load(f)
        else:
            return self
        
    def trainGAN(self, epoch, real_img, a=0, b=1, c=1):
        ### Train with LSGAN.
        ### for example, (a, b, c) = 0, 1, 1 or (a, b, c) = -1, 1, 0
        
        # ================================================================================ #
        #                             Train the discriminator                              #
        # ================================================================================ #
        
        random_data = torch.randn(real_img.size(0), self.feed_dim, 2, 2).to(self.device)
        
        # Compute loss with real images.
        real_z_score = self.netE(real_img)
        real_src_score, _ = self.netD(real_img, real_z_score)
        real_src_loss = torch.sum((real_src_score - b) ** 2)
        
        # Compute loss with fake images.
        fake_img = self.netG(random_data)
        fake_src_score, _ = self.netD(fake_img, random_data)
        
        p = random.uniform(0, 1)
        if 1 - self.pseudo_aug < p:
            fake_src_loss = torch.sum((fake_src_score - b) ** 2) # Pseudo: fake is real.
        else:
            fake_src_loss = torch.sum((fake_src_score - a) ** 2)
        
        # Update Probability Augmentation.
        lz = (torch.sign(torch.logit(real_src_score)).mean()
              - torch.sign(torch.logit(fake_src_score)).mean()) / 2
        if lz > self.args.aug_threshold:
            self.pseudo_aug += 0.01
        else:
            self.pseudo_aug -= 0.01
        self.pseudo_aug = min(1, max(0, self.pseudo_aug))
        
        # Backward and optimize.
        d_loss = 0.5 * (real_src_loss + fake_src_loss) / self.batch_size
        self.optimizer_D.zero_grad()
        d_loss.backward()
        self.optimizer_D.step()
        
        # Logging.
        loss = {}
        loss['D/loss'] = d_loss.item()
        loss['Augment/prob'] = self.pseudo_aug
        
        # ================================================================================ #
        #                               Train the generator                                #
        # ================================================================================ #
        
        random_data = torch.randn(real_img.size(0), self.feed_dim, 2, 2).to(self.device)
        
        # Compute loss with fake images.
        fake_img = self.netG(random_data)
        fake_src_score, _ = self.netD(fake_img, random_data)
        fake_src_loss = torch.sum((fake_src_score - c) ** 2)
        
        # Backward and optimize.
        g_loss = 0.5 * fake_src_loss / self.batch_size
        self.optimizer_G.zero_grad()
        g_loss.backward()
        self.optimizer_G.step()
        
        # Logging.
        loss['G/loss'] = g_loss.item()
        
        # ================================================================================ #
        #                               Train the encoder                                  #
        # ================================================================================ #
        
        real_z_score = self.netE(real_img)
        real_src_score, _ = self.netD(real_img, real_z_score)
        real_src_loss = torch.sum((real_src_score - c) ** 2)
        
        # Backward and optimize.
        e_loss = 0.5 * real_src_loss / self.batch_size
        self.optimizer_E.zero_grad()
        e_loss.backward()
        self.optimizer_E.step()
        
        # Logging.
        loss['E/loss'] = e_loss.item()
        
        # Save
        self.save_state()
        img_name = 'generator_last.png'
        img_path = os.path.join(self.args.result_dir, img_name)
        save_image(fake_img, img_path)
        
        return loss
    
    def score(self, image, lambda_anomaly=0.1):
        self.netG.eval()
        self.netD.eval()
        self.netE.eval()
        
        z = self.netE(image)
        fake = self.netG(z)
        fake_score, fake_feature = self.netD(fake, z)
        real_score, real_feature = self.netD(image, z)
        
        residual_loss = torch.abs(image - fake)
        residual_loss = residual_loss.view(residual_loss.shape[0], -1)
        residual_loss = residual_loss.sum(dim=1)
        
        discrimination_loss = torch.abs(real_feature - fake_feature)
        discrimination_loss = discrimination_loss.view(discrimination_loss.shape[0], -1)
        discrimination_loss = discrimination_loss.sum(dim=1)
        
        score = (1 - lambda_anomaly) * residual_loss + lambda_anomaly * discrimination_loss
        return score.item()
    
    def capture(self):
        hyper_params = {}
        hyper_params['Device ID'] = self.args.device_id
        hyper_params['Result Dir'] = self.args.result_dir
        hyper_params['Weight Dir'] = self.args.weight_dir
        hyper_params['Image Size'] = self.args.image_size
        hyper_params['Learning Rate'] = self.args.lr
        hyper_params["Mul Discriminator's LR"] = self.args.mul_lr_dis
        hyper_params['Num ScoreList'] = self.args.num_scorelist
        hyper_params['Probability Aug-Threshold'] = self.args.aug_threshold
        
        self.batch_size = 1
        grayscale = transforms.Grayscale(num_output_channels=1)
        resize = transforms.Resize((self.args.image_size, self.args.image_size))
        
        now = datetime.datetime.now()
        log_lotate = datetime.datetime(now.year, now.month, now.day + 1)
        
        while True:
            self.netG.train()
            self.netD.train()
            self.netE.train()
            
            capture = cv2.VideoCapture(self.args.device_id)
            try:
                _, frame = capture.read()
            finally:
                capture.release()
            timestamp = datetime.datetime.now()
            
            if timestamp >= log_lotate:
                if not os.path.exists(f'log_{log_lotate.year}-{log_lotate.month}-{log_lotate.day}'):
                    os.rename(self.args.result_dir, f'log_{log_lotate.year}-{log_lotate.month}-{log_lotate.day}')
                    os.mkdir(self.args.result_dir)
                log_lotate = datetime.datetime(now.year, now.month, now.day + 1)
            
            self.epoch += 1
            epoch_loss_G = 0.0
            epoch_loss_D = 0.0
            epoch_loss_E = 0.0

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = torch.Tensor(rgb).permute(2, 0, 1) / 255
            image = image.unsqueeze(0).to(self.device)
            image = grayscale(image)
            image = resize(image)

            loss = self.trainGAN(self.epoch, image)

            epoch_loss_D += loss['D/loss']
            epoch_loss_G += loss['G/loss']
            epoch_loss_E += loss['E/loss']

            epoch_loss = epoch_loss_G + epoch_loss_D + epoch_loss_E
                
            score = self.score(image)
            self.scorelist += [score]
            self.scorelist = self.scorelist[-self.args.num_scorelist:]
            
            print(f'{timestamp}: AnomalyScore {score} (TotalLoss {epoch_loss})')
            
            if score >= max(self.scorelist) and len(self.scorelist) >= self.args.num_scorelist:
                img_name = f'{timestamp}_{score}.jpg'
                img_path = os.path.join(self.args.result_dir, img_name)
                cv2.imwrite(img_path, frame)
            
            if not self.args.noresume:
                self.save_resume()
    
    def generate(self, num=100):
        self.netG.eval()
        
        for _ in range(num):
            random_data = torch.randn(1, self.feed_dim, 2, 2).to(self.device)
            fake_img = self.netG(random_data)[0][0,:]
            save_image(fake_img, os.path.join(self.args.result_dir, f'generated_{time.time()}.png'))
        print('New picture was generated.')