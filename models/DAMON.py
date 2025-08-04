import time,os,sys

import torch.nn.functional as F
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from src.DAMONnetwork import *

class Discriminator_rec(nn.Module):

    def __init__(self, opt):
        super(Discriminator_rec, self).__init__()
        model = Encoder(opt.ngpu,opt,1)
        layers = list(model.main.children())

        self.features = nn.Sequential(*layers[:])
        self.classifier = nn.Sequential(
            nn.Conv1d(opt.ndf*16, 1, 4, 1, 0),
            nn.Sigmoid()
        )

    def forward(self, x):
        features = self.features(x)
        features = features
        classifier = self.classifier(features)
        classifier = classifier.view(-1, 1).squeeze(1)

        return classifier, features

class Discriminator_latent(nn.Module):

    def __init__(self, opt):
        super(Discriminator_latent, self).__init__()

        self.features = nn.Sequential(
            nn.Conv1d(1, opt.ndf, 4, 2, 1),
            nn.BatchNorm1d(opt.ndf),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv1d(opt.ndf, opt.ndf * 2, 4, 2, 1),
            nn.BatchNorm1d(opt.ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv1d(opt.ndf * 2, opt.ndf * 4, 4, 2, 1),
            nn.BatchNorm1d(opt.ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv1d(opt.ndf * 4, opt.ndf * 8, 4, 2, 1),
            nn.BatchNorm1d(opt.ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv1d(opt.ndf * 8, opt.ndf * 16, 4, 2, 1),
            nn.BatchNorm1d(opt.ndf * 16),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.classifier = nn.Sequential(
            nn.Conv1d(opt.ndf * 16, 1, 4, 1, 0),
            nn.Sigmoid()
        )

    def forward(self, x):
        features = self.features(x)
        features = features
        classifier = self.classifier(features)
        classifier = classifier.view(-1, 1).squeeze(1)

        return classifier, features


class Generator(nn.Module):

    def __init__(self, opt):
        super(Generator, self).__init__()
        self.encoder = Encoder(opt.ngpu,opt,opt.nz)
        self.decoder = Decoder(opt.ngpu,opt)

    def reparameter(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, log_var = self.encoder(x)
        latent_z = self.reparameter(mu, log_var)
        output = self.decoder(latent_z)
        return output, latent_z, mu, log_var


class DAEMON(DAEMON_MODEL):


    def __init__(self, opt, train_dataloader, val_dataloader, test_data, label, device):
        super(DAEMON, self).__init__(opt)

        self.early_stopping = EarlyStopping(opt, patience=opt.patience, verbose=False)

        self.opt = opt
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.test_data = test_data
        self.label = label
        self.device = device


        self.train_batchsize = opt.train_batchsize
        self.val_batchsize = opt.val_batchsize
        self.test_batchsize = opt.test_batchsize
        self.nz = opt.nz
        self.niter = opt.niter

        self.G = Generator(opt).to(device)
        self.G.apply(weights_init)

        self.D_rec = Discriminator_rec(opt).to(device)
        self.D_rec.apply(weights_init)

        self.D_lat = Discriminator_latent(opt).to(device)
        self.D_lat.apply(weights_init)


        self.bce_criterion = nn.BCELoss()
        self.mse_criterion=nn.MSELoss()
        self.l1loss = nn.L1Loss()


        self.optimizer_D_rec = optim.Adam(self.D_rec.parameters(), lr=opt.lr_d, betas=(opt.beta1, 0.999))
        self.optimizer_D_lat = optim.Adam(self.D_lat.parameters(), lr=opt.lr_d, betas=(opt.beta1, 0.999))
        self.optimizer_G = optim.Adam(self.G.parameters(), lr=opt.lr_g, betas=(opt.beta1, 0.999))


        self.cur_epoch = 0
        self.input =  None

        self.p_z = None
        self.real_label = 1.0
        self.fake_label= 0.0
        #
        #output of discriminator_rec
        self.out_d_rec_real = None
        self.feat_rec_real = None
        self.out_d_rec_fake = None
        self.feat_rec_fake = None

        #output of discriminator_lat
        self.out_d_lat_real = None
        self.feat_lat_real = None
        self.out_d_lat_fake = None
        self.feat_lat_fake = None

        #output of generator
        self.mu = None
        self.log_var = None
        self.out_g_fake = None
        self.latent_z = None

        #loss
        self.loss_d_rec = None
        self.loss_d_rec_real = None
        self.loss_d_rec_fake = None

        self.loss_d_lat = None
        self.loss_d_lat_real = None
        self.loss_d_lat_fake = None

        self.loss_g = None
        self.loss_g_rs = None
        self.loss_g_rec = None
        self.loss_g_lat = None

