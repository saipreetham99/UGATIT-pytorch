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
from networks import ResnetGenerator, Discriminator
from utils import *

# Import for AMP
from torch.cuda.amp import GradScaler, autocast


class UGATIT(object):
    def __init__(self, args):
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

        # Enable TF32 on Ampere GPUs
        if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8:
            print('Enabling TF32 for Ampere GPU.')
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

        if torch.backends.cudnn.enabled and self.benchmark_flag:
            print('Setting benchmark flag to True for improved performance.')
            torch.backends.cudnn.benchmark = True

        print("\n##### Information #####")
        print(f"# light version: {self.light}")
        print(f"# dataset: {self.dataset}")
        print(f"# batch_size: {self.batch_size}")
        print(f"# iteration: {self.iteration}")
        print(f"# residual blocks: {self.n_res}")
        print(f"# discriminator layers: {self.n_dis}")
        print(f"# weights (adv, cycle, identity, cam): {self.adv_weight}")
        print(f"{self.cycle_weight}, {self.identity_weight}")
        print(f"{self.cam_weight}")

        print("#######################\n")

    def build_model(self):
        """Create dataloaders, networks, loss functions, and optimizers."""
        print("Building model...")
        # Image transformations
        train_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.Resize((self.img_size + 30, self.img_size + 30)),
            transforms.RandomCrop(self.img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        ])
        test_transform = transforms.Compose([
            transforms.Resize((self.img_size, self.img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        ])

        # Data loaders
        dataset_root = os.path.join('dataset', self.dataset)
        self.trainA_loader = DataLoader(
            ImageFolder(os.path.join(dataset_root, 'trainA'), train_transform),
            batch_size=self.batch_size, shuffle=True, num_workers=4, pin_memory=True)
        self.trainB_loader = DataLoader(
            ImageFolder(os.path.join(dataset_root, 'trainB'), train_transform),
            batch_size=self.batch_size, shuffle=True, num_workers=4, pin_memory=True)
        self.testA_loader = DataLoader(
            ImageFolder(os.path.join(dataset_root, 'testA'), test_transform),
            batch_size=1, shuffle=False, num_workers=4, pin_memory=True)
        self.testB_loader = DataLoader(
            ImageFolder(os.path.join(dataset_root, 'testB'), test_transform),
            batch_size=1, shuffle=False, num_workers=4, pin_memory=True)

        # Networks
        self.genA2B = nn.DataParallel(ResnetGenerator(input_nc=3, output_nc=3, ngf=self.ch,
                                      n_blocks=self.n_res, img_size=self.img_size, light=self.light).to(self.device))
        self.genB2A = nn.DataParallel(ResnetGenerator(input_nc=3, output_nc=3, ngf=self.ch,
                                      n_blocks=self.n_res, img_size=self.img_size, light=self.light).to(self.device))
        self.disGA = nn.DataParallel(Discriminator(
            input_nc=3, ndf=self.ch, n_layers=7).to(self.device))
        self.disGB = nn.DataParallel(Discriminator(
            input_nc=3, ndf=self.ch, n_layers=7).to(self.device))
        self.disLA = nn.DataParallel(Discriminator(
            input_nc=3, ndf=self.ch, n_layers=5).to(self.device))
        self.disLB = nn.DataParallel(Discriminator(
            input_nc=3, ndf=self.ch, n_layers=5).to(self.device))

        # Loss functions
        self.L1_loss = nn.L1Loss().to(self.device)
        self.MSE_loss = nn.MSELoss().to(self.device)
        self.BCE_loss = nn.BCEWithLogitsLoss().to(self.device)

        # Optimizers
        self.G_optim = torch.optim.Adam(itertools.chain(self.genA2B.parameters(
        ), self.genB2A.parameters()), lr=self.lr, betas=(0.5, 0.999), weight_decay=self.weight_decay)
        self.D_optim = torch.optim.Adam(itertools.chain(self.disGA.parameters(), self.disGB.parameters(
        ), self.disLA.parameters(), self.disLB.parameters()), lr=self.lr, betas=(0.5, 0.999), weight_decay=self.weight_decay)

        # GradScaler for AMP
        self.G_scaler = GradScaler()
        self.D_scaler = GradScaler()

        # Rho clipper for AdaILN
        self.Rho_clipper = RhoClipper(0, 1)
        print("Model built successfully.")

    def train(self):
        """Main training loop."""
        # Set models to train mode
        self.genA2B.train()
        self.genB2A.train()
        self.disGA.train()
        self.disGB.train()
        self.disLA.train()
        self.disLB.train()

        start_iter = 1
        # Resume from checkpoint if requested
        if self.resume:
            model_list = glob(os.path.join(
                self.result_dir, self.dataset, 'model', '*.pt'))
            if model_list:
                model_list.sort(key=lambda x: int(
                    os.path.basename(x).split('_')[-1].split('.')[0]))
                latest_checkpoint = model_list[-1]
                start_iter = int(os.path.basename(
                    latest_checkpoint).split('_')[-1].split('.')[0])
                self.load(os.path.join(self.result_dir,
                          self.dataset, 'model'), start_iter)
                print(f"[*] Resuming training from iteration {start_iter}")

        print('Training started!')
        start_time = time.time()

        trainA_iter = iter(self.trainA_loader)
        trainB_iter = iter(self.trainB_loader)

        for step in range(start_iter, self.iteration + 1):
            # Dynamic learning rate decay
            if self.decay_flag and step > (self.iteration // 2):
                decay_amount = self.lr / (self.iteration // 2)
                self.G_optim.param_groups[0]['lr'] -= decay_amount
                self.D_optim.param_groups[0]['lr'] -= decay_amount

            # Get next batch of real images
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

            # --- Update Discriminators ---
            self.D_optim.zero_grad()

            with autocast():
                fake_A2B, _, _ = self.genA2B(real_A)
                fake_B2A, _, _ = self.genB2A(real_B)

                real_GA_logit, real_GA_cam_logit, _ = self.disGA(real_A)
                real_LA_logit, real_LA_cam_logit, _ = self.disLA(real_A)
                real_GB_logit, real_GB_cam_logit, _ = self.disGB(real_B)
                real_LB_logit, real_LB_cam_logit, _ = self.disLB(real_B)

                fake_GA_logit, fake_GA_cam_logit, _ = self.disGA(
                    fake_B2A.detach())
                fake_LA_logit, fake_LA_cam_logit, _ = self.disLA(
                    fake_B2A.detach())
                fake_GB_logit, fake_GB_cam_logit, _ = self.disGB(
                    fake_A2B.detach())
                fake_LB_logit, fake_LB_cam_logit, _ = self.disLB(
                    fake_A2B.detach())

                D_adv_loss_A = self.MSE_loss(real_GA_logit, torch.ones_like(
                    real_GA_logit)) + self.MSE_loss(fake_GA_logit, torch.zeros_like(fake_GA_logit))
                D_adv_loss_B = self.MSE_loss(real_GB_logit, torch.ones_like(
                    real_GB_logit)) + self.MSE_loss(fake_GB_logit, torch.zeros_like(fake_GB_logit))
                D_adv_loss_LA = self.MSE_loss(real_LA_logit, torch.ones_like(
                    real_LA_logit)) + self.MSE_loss(fake_LA_logit, torch.zeros_like(fake_LA_logit))
                D_adv_loss_LB = self.MSE_loss(real_LB_logit, torch.ones_like(
                    real_LB_logit)) + self.MSE_loss(fake_LB_logit, torch.zeros_like(fake_LB_logit))
                D_cam_loss_A = self.MSE_loss(real_GA_cam_logit, torch.ones_like(
                    real_GA_cam_logit)) + self.MSE_loss(fake_GA_cam_logit, torch.zeros_like(fake_GA_cam_logit))
                D_cam_loss_B = self.MSE_loss(real_GB_cam_logit, torch.ones_like(
                    real_GB_cam_logit)) + self.MSE_loss(fake_GB_cam_logit, torch.zeros_like(fake_GB_cam_logit))
                D_cam_loss_LA = self.MSE_loss(real_LA_cam_logit, torch.ones_like(
                    real_LA_cam_logit)) + self.MSE_loss(fake_LA_cam_logit, torch.zeros_like(fake_LA_cam_logit))
                D_cam_loss_LB = self.MSE_loss(real_LB_cam_logit, torch.ones_like(
                    real_LB_cam_logit)) + self.MSE_loss(fake_LB_cam_logit, torch.zeros_like(fake_LB_cam_logit))

                D_loss_A = self.adv_weight * \
                    (D_adv_loss_A + D_adv_loss_LA + D_cam_loss_A + D_cam_loss_LA)
                D_loss_B = self.adv_weight * \
                    (D_adv_loss_B + D_adv_loss_LB + D_cam_loss_B + D_cam_loss_LB)

                Discriminator_loss = D_loss_A + D_loss_B

            self.D_scaler.scale(Discriminator_loss).backward()
            self.D_scaler.step(self.D_optim)
            self.D_scaler.update()

            # --- Update Generators ---
            self.G_optim.zero_grad()

            with autocast():
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

                G_adv_loss_A = self.MSE_loss(fake_GA_logit, torch.ones_like(
                    fake_GA_logit)) + self.MSE_loss(fake_LA_logit, torch.ones_like(fake_LA_logit))
                G_adv_loss_B = self.MSE_loss(fake_GB_logit, torch.ones_like(
                    fake_GB_logit)) + self.MSE_loss(fake_LB_logit, torch.ones_like(fake_LB_logit))
                G_cam_loss_A = self.MSE_loss(fake_GA_cam_logit, torch.ones_like(
                    fake_GA_cam_logit)) + self.MSE_loss(fake_LA_cam_logit, torch.ones_like(fake_LA_cam_logit))
                G_cam_loss_B = self.MSE_loss(fake_GB_cam_logit, torch.ones_like(
                    fake_GB_cam_logit)) + self.MSE_loss(fake_LB_cam_logit, torch.ones_like(fake_LB_cam_logit))
                G_cycle_loss_A = self.L1_loss(fake_A2B2A, real_A)
                G_cycle_loss_B = self.L1_loss(fake_B2A2B, real_B)
                G_identity_loss_A = self.L1_loss(fake_A2A, real_A)
                G_identity_loss_B = self.L1_loss(fake_B2B, real_B)
                G_cam_logit_loss_A = self.BCE_loss(fake_B2A_cam_logit, torch.ones_like(
                    fake_B2A_cam_logit)) + self.BCE_loss(fake_A2A_cam_logit, torch.zeros_like(fake_A2A_cam_logit))
                G_cam_logit_loss_B = self.BCE_loss(fake_A2B_cam_logit, torch.ones_like(
                    fake_A2B_cam_logit)) + self.BCE_loss(fake_B2B_cam_logit, torch.zeros_like(fake_B2B_cam_logit))

                G_loss_A = self.adv_weight * (G_adv_loss_A + G_cam_loss_A) + self.cycle_weight * G_cycle_loss_A + \
                    self.identity_weight * G_identity_loss_A + self.cam_weight * G_cam_logit_loss_A
                G_loss_B = self.adv_weight * (G_adv_loss_B + G_cam_loss_B) + self.cycle_weight * G_cycle_loss_B + \
                    self.identity_weight * G_identity_loss_B + self.cam_weight * G_cam_logit_loss_B

                Generator_loss = G_loss_A + G_loss_B

            self.G_scaler.scale(Generator_loss).backward()
            self.G_scaler.step(self.G_optim)
            self.G_scaler.update()

            # Clip Rho in AdaILN/ILN
            self.genA2B.apply(self.Rho_clipper)
            self.genB2A.apply(self.Rho_clipper)

            # --- Logging and Saving ---
            if step % self.print_freq == 0:
                elapsed_time = time.time() - start_time
                print(f"[{step:6d}/{self.iteration:6d}] time: {elapsed_time:.1f}s, D_loss: {Discriminator_loss.item():.4f}, G_loss: {Generator_loss.item():.4f}")

            if step % self.save_freq == 0:
                self.save(os.path.join(self.result_dir,
                          self.dataset, 'model'), step)
                self._save_intermediate_images(step)

    def _save_intermediate_images(self, step):
        """Saves a grid of translated images."""
        print(f"Saving intermediate images for iteration {step}...")
        img_dir = os.path.join(self.result_dir, self.dataset, 'img')

        self.genA2B.eval()
        self.genB2A.eval()

        # Grab a single test image from each domain
        real_A, _ = next(iter(self.testA_loader))
        real_B, _ = next(iter(self.testB_loader))
        real_A, real_B = real_A.to(self.device), real_B.to(self.device)

        with torch.no_grad():
            with autocast():  # Use autocast for inference as well
                fake_A2B, _, _ = self.genA2B(real_A)
                fake_B2A, _, _ = self.genB2A(real_B)

                # Cycle consistency
                fake_A2B2A, _, _ = self.genB2A(fake_A2B)
                fake_B2A2B, _, _ = self.genA2B(fake_B2A)

        # Create image grids
        A2B_grid = np.concatenate([tensor2numpy(denorm(real_A[0])), tensor2numpy(
            denorm(fake_A2B[0])), tensor2numpy(denorm(fake_A2B2A[0]))], axis=1)
        B2A_grid = np.concatenate([tensor2numpy(denorm(real_B[0])), tensor2numpy(
            denorm(fake_B2A[0])), tensor2numpy(denorm(fake_B2A2B[0]))], axis=1)

        # Save images
        cv2.imwrite(os.path.join(img_dir, f'A2B_{step:07d}.png'), RGB2BGR(A2B_grid * 255.0))
        cv2.imwrite(os.path.join(img_dir, f'B2A_{step:07d}.png'), RGB2BGR(B2A_grid * 255.0))

        # Restore train mode
        self.genA2B.train()
        self.genB2A.train()
        print("Intermediate images saved.")

    def save(self, dir_path, step):
        """Saves model checkpoints."""
        print(f"Saving checkpoint for iteration {step}...")
        os.makedirs(dir_path, exist_ok=True)
        params = {
            'genA2B': self.genA2B.state_dict(),
            'genB2A': self.genB2A.state_dict(),
            'disGA': self.disGA.state_dict(),
            'disGB': self.disGB.state_dict(),
            'disLA': self.disLA.state_dict(),
            'disLB': self.disLB.state_dict(),
            'G_optim': self.G_optim.state_dict(),
            'D_optim': self.D_optim.state_dict(),
            'G_scaler': self.G_scaler.state_dict(),
            'D_scaler': self.D_scaler.state_dict()
        }
        torch.save(params, os.path.join(
            dir_path, f'{self.dataset}_params_{step:07d}.pt'))
        print("Checkpoint saved.")

    def load(self, dir_path, step):
        """Loads model checkpoints."""
        print(f"Loading checkpoint from iteration {step}...")
        try:
            params = torch.load(os.path.join(dir_path, f'{self.dataset}_params_{step:07d}.pt'), map_location=self.device)
            self.genA2B.load_state_dict(params['genA2B'])
            self.genB2A.load_state_dict(params['genB2A'])
            self.disGA.load_state_dict(params['disGA'])
            self.disGB.load_state_dict(params['disGB'])
            self.disLA.load_state_dict(params['disLA'])
            self.disLB.load_state_dict(params['disLB'])
            self.G_optim.load_state_dict(params['G_optim'])
            self.D_optim.load_state_dict(params['D_optim'])
            self.G_scaler.load_state_dict(params['G_scaler'])
            self.D_scaler.load_state_dict(params['D_scaler'])
            print("Checkpoint loaded successfully.")
        except FileNotFoundError:
            print(f"Error: Checkpoint file not found at {dir_path}. Starting from scratch.")
        except Exception as e:
            print(f"An error occurred while loading the checkpoint: {e}")

    def test(self):
        """Tests the model and saves results."""
        model_list = glob(os.path.join(self.result_dir, self.dataset, 'model', '*.pt'))
        if not model_list:
            print("Error: No trained model found!")
            return

        model_list.sort(key=lambda x: int(os.path.basename(x).split('_')[-1].split('.')[0]))
        latest_checkpoint = model_list[-1]
        iteration = int(os.path.basename(latest_checkpoint).split('_')[-1].split('.')[0])

        self.load(os.path.join(self.result_dir,self.dataset, 'model'), iteration)
        print(f"[*] Loaded model from iteration {iteration} for testing.")

        self.genA2B.eval()
        self.genB2A.eval()

        test_img_dir = os.path.join(self.result_dir, self.dataset, 'test')

        for n, (real_A, _) in enumerate(self.testA_loader):
            real_A = real_A.to(self.device)
            with torch.no_grad():
                with autocast():
                    fake_A2B, _, _ = self.genA2B(real_A)

            A_out_path = os.path.join(test_img_dir, f'A2B_{n + 1}.png')
            cv2.imwrite(A_out_path, RGB2BGR(tensor2numpy(denorm(fake_A2B[0])) * 255.0))

        for n, (real_B, _) in enumerate(self.testB_loader):
            real_B = real_B.to(self.device)
            with torch.no_grad():
                with autocast():
                    fake_B2A, _, _ = self.genB2A(real_B)

            B_out_path = os.path.join(test_img_dir, f'B2A_{n + 1}.png')
            cv2.imwrite(B_out_path, RGB2BGR(tensor2numpy(denorm(fake_B2A[0])) * 255.0))

        print(f"[*] Test results saved in {test_img_dir}")
