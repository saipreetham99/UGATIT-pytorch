

import os
import time
import itertools
import cv2
import numpy as np
import torch
import torch.nn as nn
from glob import glob
from dataset import ImageFolder
from torchvision import transforms
from torch.utils.data import DataLoader
from networks import *
from utils import *


class UGATIT(object):
    def __init__(self, args):
        # ----- basic cfg -----
        self.light = args.light
        self.model_name = 'UGATIT_light' if self.light else 'UGATIT'
        self.result_dir = args.result_dir
        self.dataset = args.dataset
        self.iteration = args.iteration
        self.decay_flag = args.decay_flag
        self.batch_size = args.batch_size
        self.print_freq = args.print_freq
        self.save_freq = args.save_freq
        self.lr = args.lr
        self.weight_decay = args.weight_decay
        self.ch = args.ch
        self.adv_weight = args.adv_weight
        self.cycle_weight = args.cycle_weight
        self.identity_weight = args.identity_weight
        self.cam_weight = args.cam_weight
        self.n_res = args.n_res
        self.n_dis = args.n_dis
        self.img_size = args.img_size
        self.img_ch = args.img_ch
        self.device = args.device
        self.benchmark_flag = args.benchmark_flag
        self.resume = args.resume

        # ----- cudnn perf tweak -----
        if torch.backends.cudnn.enabled and self.benchmark_flag:
            torch.backends.cudnn.benchmark = True

        # ----- console info -----
        print("\n##### Information #####")
        print("# light          :", self.light)
        print("# dataset        :", self.dataset)
        print("# batch_size     :", self.batch_size)
        print("# iteration      :", self.iteration)
        print("# residual blocks:", self.n_res)
        print("# disc layers    :", self.n_dis)
        print("# weights        :", self.adv_weight,
              self.cycle_weight, self.identity_weight, self.cam_weight)

    # ------------------------------------------------------------------ #
    # Setup
    # ------------------------------------------------------------------ #
    def build_model(self):
        """Create dataloaders, nets, loss funcs, opts."""
        train_tf = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.Resize((self.img_size + 30, self.img_size + 30)),
            transforms.RandomCrop(self.img_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5,)*3, (0.5,)*3)
        ])
        test_tf = transforms.Compose([
            transforms.Resize((self.img_size, self.img_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,)*3, (0.5,)*3)
        ])

        root = 'dataset'
        self.trainA_loader = DataLoader(
            ImageFolder(os.path.join(root, self.dataset, 'trainA'), train_tf),
            batch_size=self.batch_size, shuffle=True)
        self.trainB_loader = DataLoader(
            ImageFolder(os.path.join(root, self.dataset, 'trainB'), train_tf),
            batch_size=self.batch_size, shuffle=True)
        self.testA_loader = DataLoader(
            ImageFolder(os.path.join(root, self.dataset, 'testA'),  test_tf),
            batch_size=1, shuffle=False)
        self.testB_loader = DataLoader(
            ImageFolder(os.path.join(root, self.dataset, 'testB'),  test_tf),
            batch_size=1, shuffle=False)

        # ----- nets -----
        self.genA2B = ResnetGenerator(3, 3, self.ch, self.n_res,
                                      self.img_size, self.light).to(self.device)
        self.genB2A = ResnetGenerator(3, 3, self.ch, self.n_res,
                                      self.img_size, self.light).to(self.device)
        self.disGA = Discriminator(3, self.ch, 7).to(self.device)
        self.disGB = Discriminator(3, self.ch, 7).to(self.device)
        self.disLA = Discriminator(3, self.ch, 5).to(self.device)
        self.disLB = Discriminator(3, self.ch, 5).to(self.device)

        # ----- losses -----
        self.L1_loss = nn.L1Loss().to(self.device)
        self.MSE_loss = nn.MSELoss().to(self.device)
        self.BCE_loss = nn.BCEWithLogitsLoss().to(self.device)

        # ----- opts -----
        self.G_optim = torch.optim.Adam(
            itertools.chain(self.genA2B.parameters(),
                            self.genB2A.parameters()),
            lr=self.lr, betas=(0.5, 0.999), weight_decay=self.weight_decay)
        self.D_optim = torch.optim.Adam(
            itertools.chain(self.disGA.parameters(), self.disGB.parameters(),
                            self.disLA.parameters(), self.disLB.parameters()),
            lr=self.lr, betas=(0.5, 0.999), weight_decay=self.weight_decay)

        # ----- clipper -----
        self.Rho_clipper = RhoClipper(0, 1)

    # ------------------------------------------------------------------ #
    # Train
    # ------------------------------------------------------------------ #
    def train(self):
        nets = [self.genA2B, self.genB2A,
                self.disGA, self.disGB, self.disLA, self.disLB]
        for n in nets:
            n.train()

        # resume --------------------------------------------------------
        start_iter = 1
        if self.resume:
            ckpts = glob(os.path.join(self.result_dir, self.dataset,
                                      'model', '*.pt'))
            if ckpts:
                ckpts.sort()
                start_iter = int(ckpts[-1].split('_')[-1].split('.')[0])
                self.load(os.path.join(self.result_dir,
                                       self.dataset, 'model'), start_iter)
                print(" [*] resumed from iter", start_iter)

        # training loop -------------------------------------------------
        print('training start!')
        start_time = time.time()

        # build iterators once
        trainA_iter = iter(self.trainA_loader)
        trainB_iter = iter(self.trainB_loader)
        testA_iter = iter(self.testA_loader)
        testB_iter = iter(self.testB_loader)

        for step in range(start_iter, self.iteration + 1):

            # lr decay
            if self.decay_flag and step > (self.iteration // 2):
                lr_step = self.lr / (self.iteration // 2)
                self.G_optim.param_groups[0]['lr'] -= lr_step
                self.D_optim.param_groups[0]['lr'] -= lr_step

            # grab next batch (reset when exhausted)
            try:
                real_A, _ = next(trainA_iter)
            except StopIteration:
                trainA_iter = iter(self.trainA_loader)
                real_A, _ = next(trainA_iter)

            try:
                real_B, _ = next(trainB_iter)
            except StopIteration:
                trainB_iter = iter(self.trainB_loader)
                real_B, _ = next(trainB_iter)

            real_A, real_B = real_A.to(self.device), real_B.to(self.device)

            # ----- Discriminator pass -----
            self.D_optim.zero_grad()

            fake_A2B, _, _ = self.genA2B(real_A)
            fake_B2A, _, _ = self.genB2A(real_B)

            real_GA_logit, real_GA_cam_logit, _ = self.disGA(real_A)
            real_LA_logit, real_LA_cam_logit, _ = self.disLA(real_A)
            real_GB_logit, real_GB_cam_logit, _ = self.disGB(real_B)
            real_LB_logit, real_LB_cam_logit, _ = self.disLB(real_B)

            fake_GA_logit, fake_GA_cam_logit, _ = self.disGA(fake_B2A.detach())
            fake_LA_logit, fake_LA_cam_logit, _ = self.disLA(fake_B2A.detach())
            fake_GB_logit, fake_GB_cam_logit, _ = self.disGB(fake_A2B.detach())
            fake_LB_logit, fake_LB_cam_logit, _ = self.disLB(fake_A2B.detach())

            D_loss_A = self.adv_weight * (
                self.MSE_loss(real_GA_logit, torch.ones_like(real_GA_logit)) +
                self.MSE_loss(fake_GA_logit, torch.zeros_like(fake_GA_logit)) +
                self.MSE_loss(real_LA_logit, torch.ones_like(real_LA_logit)) +
                self.MSE_loss(fake_LA_logit, torch.zeros_like(fake_LA_logit)) +
                self.MSE_loss(real_GA_cam_logit, torch.ones_like(real_GA_cam_logit)) +
                self.MSE_loss(fake_GA_cam_logit, torch.zeros_like(fake_GA_cam_logit)) +
                self.MSE_loss(real_LA_cam_logit, torch.ones_like(real_LA_cam_logit)) +
                self.MSE_loss(fake_LA_cam_logit,
                              torch.zeros_like(fake_LA_cam_logit))
            )

            D_loss_B = self.adv_weight * (
                self.MSE_loss(real_GB_logit, torch.ones_like(real_GB_logit)) +
                self.MSE_loss(fake_GB_logit, torch.zeros_like(fake_GB_logit)) +
                self.MSE_loss(real_LB_logit, torch.ones_like(real_LB_logit)) +
                self.MSE_loss(fake_LB_logit, torch.zeros_like(fake_LB_logit)) +
                self.MSE_loss(real_GB_cam_logit, torch.ones_like(real_GB_cam_logit)) +
                self.MSE_loss(fake_GB_cam_logit, torch.zeros_like(fake_GB_cam_logit)) +
                self.MSE_loss(real_LB_cam_logit, torch.ones_like(real_LB_cam_logit)) +
                self.MSE_loss(fake_LB_cam_logit,
                              torch.zeros_like(fake_LB_cam_logit))
            )

            (D_loss_A + D_loss_B).backward()
            self.D_optim.step()

            # ----- Generator pass -----
            self.G_optim.zero_grad()

            fake_A2B, fake_A2B_cam_logit, _ = self.genA2B(real_A)
            fake_B2A, fake_B2A_cam_logit, _ = self.genB2A(real_B)

            fake_A2B2A, _, _ = self.genB2A(fake_A2B)
            fake_B2A2B, _, _ = self.genA2B(fake_B2A)

            fake_A2A, fake_A2A_cam_logit, _ = self.genB2A(real_A)
            fake_B2B, fake_B2B_cam_logit, _ = self.genA2B(real_B)

            fake_GA_logit, fake_GA_cam_logit, _ = self.disGA(fake_B2A)
            fake_LA_logit, fake_LA_cam_logit, _ = self.disLA(fake_B2A)
            fake_GB_logit, fake_GB_cam_logit, _ = self.disGB(fake_A2B)
            fake_LB_logit, fake_LB_cam_logit, _ = self.disLB(fake_A2B)

            G_loss_A = self.adv_weight * (
                self.MSE_loss(fake_GA_logit, torch.ones_like(fake_GA_logit)) +
                self.MSE_loss(fake_LA_logit, torch.ones_like(fake_LA_logit)) +
                self.MSE_loss(fake_GA_cam_logit, torch.ones_like(fake_GA_cam_logit)) +
                self.MSE_loss(fake_LA_cam_logit,
                              torch.ones_like(fake_LA_cam_logit))
            ) + self.cycle_weight * self.L1_loss(fake_A2B2A, real_A) + \
                self.identity_weight * self.L1_loss(fake_A2A, real_A) + \
                self.cam_weight * (self.BCE_loss(fake_B2A_cam_logit, torch.ones_like(fake_B2A_cam_logit)) +
                                   self.BCE_loss(fake_A2A_cam_logit, torch.zeros_like(fake_A2A_cam_logit)))

            G_loss_B = self.adv_weight * (
                self.MSE_loss(fake_GB_logit, torch.ones_like(fake_GB_logit)) +
                self.MSE_loss(fake_LB_logit, torch.ones_like(fake_LB_logit)) +
                self.MSE_loss(fake_GB_cam_logit, torch.ones_like(fake_GB_cam_logit)) +
                self.MSE_loss(fake_LB_cam_logit,
                              torch.ones_like(fake_LB_cam_logit))
            ) + self.cycle_weight * self.L1_loss(fake_B2A2B, real_B) + \
                self.identity_weight * self.L1_loss(fake_B2B, real_B) + \
                self.cam_weight * (self.BCE_loss(fake_A2B_cam_logit, torch.ones_like(fake_A2B_cam_logit)) +
                                   self.BCE_loss(fake_B2B_cam_logit, torch.zeros_like(fake_B2B_cam_logit)))

            (G_loss_A + G_loss_B).backward()
            self.G_optim.step()

            # clip AdaILN / ILN rho
            self.genA2B.apply(self.Rho_clipper)
            self.genB2A.apply(self.Rho_clipper)

            # ----- log -----
            if step % self.print_freq == 0:
                print("[%5d/%5d] time %.1f  d %.5f  g %.5f" %
                      (step, self.iteration, time.time()-start_time,
                       (D_loss_A+D_loss_B).item(),
                       (G_loss_A+G_loss_B).item()))

            # ----- save image samples -----
            if step % self.print_freq == 0:
                self._save_samples(step, trainA_iter, trainB_iter,
                                   testA_iter, testB_iter)

            # ----- save ckpt -----
            if step % self.save_freq == 0:
                self.save(os.path.join(self.result_dir,
                          self.dataset, 'model'), step)

    # ------------------------------------------------------------------ #
    # Helper: sample viz
    # ------------------------------------------------------------------ #
    def _save_samples(self, step, trainA_iter, trainB_iter, testA_iter, testB_iter):
        def grab_next(loader_iter, loader):
            try:
                x, _ = next(loader_iter)
            except StopIteration:
                loader_iter = iter(loader)
                x, _ = next(loader_iter)
            return x.to(self.device), loader_iter

        self.genA2B.eval()
        self.genB2A.eval()
        A2B_grid, B2A_grid = np.zeros(
            (self.img_size*7, 0, 3)), np.zeros((self.img_size*7, 0, 3))

        # 5 train + 5 test
        for loader_pair in [(trainA_iter, self.trainA_loader),
                            (trainB_iter, self.trainB_loader)]*5 + \
            [(testA_iter, self.testA_loader),
             (testB_iter, self.testB_loader)]*5:
            # iterate A then B
            real_A, trainA_iter = grab_next(loader_pair[0], loader_pair[1])
            real_B, trainB_iter = grab_next(
                loader_pair[0], loader_pair[1])  # placeholder

        # re‑enter train mode
        self.genA2B.train()
        self.genB2A.train()

    # ------------------------------------------------------------------ #
    # Save / load
    # ------------------------------------------------------------------ #
    def save(self, dir_path, step):
        params = dict(genA2B=self.genA2B.state_dict(),
                      genB2A=self.genB2A.state_dict(),
                      disGA=self.disGA.state_dict(),
                      disGB=self.disGB.state_dict(),
                      disLA=self.disLA.state_dict(),
                      disLB=self.disLB.state_dict())
        torch.save(params, os.path.join(
            dir_path, f"{self.dataset}_params_{step:07d}.pt"))

    def load(self, dir_path, step):
        params = torch.load(os.path.join(
            dir_path, f"{self.dataset}_params_{step:07d}.pt"))
        self.genA2B.load_state_dict(params['genA2B'])
        self.genB2A.load_state_dict(params['genB2A'])
        self.disGA.load_state_dict(params['disGA'])
        self.disGB.load_state_dict(params['disGB'])
        self.disLA.load_state_dict(params['disLA'])
        self.disLB.load_state_dict(params['disLB'])

    def test(self):
        model_list = glob(os.path.join(
            self.result_dir, self.dataset, 'model', '*.pt'))
        if not len(model_list) == 0:
            model_list.sort()
            iter = int(model_list[-1].split('_')[-1].split('.')[0])
            self.load(os.path.join(self.result_dir,
                      self.dataset, 'model'), iter)
            print(" [*] Load SUCCESS")
        else:
            print(" [*] Load FAILURE")
            return

        self.genA2B.eval(), self.genB2A.eval()
        for n, (real_A, _) in enumerate(self.testA_loader):
            real_A = real_A.to(self.device)

            fake_A2B, _, fake_A2B_heatmap = self.genA2B(real_A)

            fake_A2B2A, _, fake_A2B2A_heatmap = self.genB2A(fake_A2B)

            fake_A2A, _, fake_A2A_heatmap = self.genB2A(real_A)

            A2B = np.concatenate((RGB2BGR(tensor2numpy(denorm(real_A[0]))),
                                  cam(tensor2numpy(
                                      fake_A2A_heatmap[0]), self.img_size),
                                  RGB2BGR(tensor2numpy(denorm(fake_A2A[0]))),
                                  cam(tensor2numpy(
                                      fake_A2B_heatmap[0]), self.img_size),
                                  RGB2BGR(tensor2numpy(denorm(fake_A2B[0]))),
                                  cam(tensor2numpy(
                                      fake_A2B2A_heatmap[0]), self.img_size),
                                  RGB2BGR(tensor2numpy(denorm(fake_A2B2A[0])))), 0)

            cv2.imwrite(os.path.join(self.result_dir, self.dataset,
                        'test', 'A2B_%d.png' % (n + 1)), A2B * 255.0)

        for n, (real_B, _) in enumerate(self.testB_loader):
            real_B = real_B.to(self.device)

            fake_B2A, _, fake_B2A_heatmap = self.genB2A(real_B)

            fake_B2A2B, _, fake_B2A2B_heatmap = self.genA2B(fake_B2A)

            fake_B2B, _, fake_B2B_heatmap = self.genA2B(real_B)

            B2A = np.concatenate((RGB2BGR(tensor2numpy(denorm(real_B[0]))),
                                  cam(tensor2numpy(
                                      fake_B2B_heatmap[0]), self.img_size),
                                  RGB2BGR(tensor2numpy(denorm(fake_B2B[0]))),
                                  cam(tensor2numpy(
                                      fake_B2A_heatmap[0]), self.img_size),
                                  RGB2BGR(tensor2numpy(denorm(fake_B2A[0]))),
                                  cam(tensor2numpy(
                                      fake_B2A2B_heatmap[0]), self.img_size),
                                  RGB2BGR(tensor2numpy(denorm(fake_B2A2B[0])))), 0)

            cv2.imwrite(os.path.join(self.result_dir, self.dataset,
                        'test', 'B2A_%d.png' % (n + 1)), B2A * 255.0)
