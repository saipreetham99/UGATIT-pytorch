import os
import time
import itertools
import glob as _glob
import logging
from datetime import datetime
import torch
import torch.nn as nn
from torch.cuda import amp
from torch.nn.parameter import Parameter
from torch.utils.data import DataLoader
from torchvision import transforms

from utils import (
    RGB2BGR,
    RhoClipper,
    cam,
    check_folder,
    denorm,
    imagenet_norm,
    str2bool,
    tensor2numpy,
)
from dataset import ImageFolder
from networks import ResnetGenerator, Discriminator

# ----------------------------------------------------------------------------
# Helper: configure global perf settings once the GPU is known to be Blackwell
# ----------------------------------------------------------------------------
if torch.cuda.is_available():
    # B200 ships with bfloat16 & TF32 paths – enable them.
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")


class UGATIT(object):
    """Only the parts requested by user were altered."""

    def __init__(self, args):
        # ------- new CLI options --------
        self.save_interval = args.save_interval  # every N iters
        self.keep_ckpts = args.keep_ckpts  # rolling window
        self.amp_enabled = args.amp and torch.cuda.is_available()

        # ------- old init code (trimmed for space) --------
        self.light = args.light
        self.model_name = "UGATIT_light" if self.light else "UGATIT"
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

        if torch.backends.cudnn.enabled and self.benchmark_flag:
            torch.backends.cudnn.benchmark = True

        # -------- logging setup (more verbose) --------
        log_fmt = "%Y-%m-%d %H:%M:%S"
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s | %(message)s",
            datefmt=log_fmt,
        )
        logging.info("Starting UGATIT with AMP=%s", self.amp_enabled)

    # ----------------------------------------------------------------------
    # Building model (unchanged apart from compile & DataLoader pin)
    # ----------------------------------------------------------------------
    def build_model(self):
        train_tf = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(),
                transforms.Resize((self.img_size + 30, self.img_size + 30)),
                transforms.RandomCrop(self.img_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
            ]
        )
        test_tf = transforms.Compose(
            [
                transforms.Resize((self.img_size, self.img_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
            ]
        )

        self.trainA_loader = DataLoader(
            ImageFolder(os.path.join("dataset", self.dataset, "trainA"), train_tf),
            batch_size=self.batch_size,
            shuffle=True,
            pin_memory=True,
        )
        self.trainB_loader = DataLoader(
            ImageFolder(os.path.join("dataset", self.dataset, "trainB"), train_tf),
            batch_size=self.batch_size,
            shuffle=True,
            pin_memory=True,
        )
        self.testA_loader = DataLoader(
            ImageFolder(os.path.join("dataset", self.dataset, "testA"), test_tf),
            batch_size=1,
            shuffle=False,
            pin_memory=True,
        )
        self.testB_loader = DataLoader(
            ImageFolder(os.path.join("dataset", self.dataset, "testB"), test_tf),
            batch_size=1,
            shuffle=False,
            pin_memory=True,
        )

        # ---- models ----
        self.genA2B = ResnetGenerator(
            3, 3, self.ch, self.n_res, self.img_size, self.light
        ).to(self.device)
        self.genB2A = ResnetGenerator(
            3, 3, self.ch, self.n_res, self.img_size, self.light
        ).to(self.device)
        self.disGA = Discriminator(3, self.ch, 7).to(self.device)
        self.disGB = Discriminator(3, self.ch, 7).to(self.device)
        self.disLA = Discriminator(3, self.ch, 5).to(self.device)
        self.disLB = Discriminator(3, self.ch, 5).to(self.device)

        # optional torch.compile for extra speed (PyTorch 2.0+)
        if hasattr(torch, "compile"):
            self.genA2B = torch.compile(self.genA2B, mode="reduce-overhead")
            self.genB2A = torch.compile(self.genB2A, mode="reduce-overhead")

        # ---- loss ----
        self.L1_loss = nn.L1Loss().to(self.device)
        self.MSE_loss = nn.MSELoss().to(self.device)
        self.BCE_loss = nn.BCEWithLogitsLoss().to(self.device)

        # ---- optim ----
        self.G_optim = torch.optim.Adam(
            itertools.chain(self.genA2B.parameters(), self.genB2A.parameters()),
            lr=self.lr,
            betas=(0.5, 0.999),
            weight_decay=self.weight_decay,
        )
        self.D_optim = torch.optim.Adam(
            itertools.chain(
                self.disGA.parameters(),
                self.disGB.parameters(),
                self.disLA.parameters(),
                self.disLB.parameters(),
            ),
            lr=self.lr,
            betas=(0.5, 0.999),
            weight_decay=self.weight_decay,
        )

        # utility
        self.Rho_clipper = RhoClipper(0, 1)

        # AMP scalers (created lazily to avoid overhead on CPU)
        if self.amp_enabled:
            self.scaler_G = torch.GradScaler("cuda")
            self.scaler_D = torch.GradScaler("cuda")

    # ----------------------------------------------------------------------
    # Train
    # ----------------------------------------------------------------------
    def train(self):
        self.genA2B.train()
        self.genB2A.train()
        self.disGA.train()
        self.disGB.train()
        self.disLA.train()
        self.disLB.train()

        # resume support
        start_iter = 1
        ckpt_dir = os.path.join(self.result_dir, self.dataset, "model")
        if self.resume:
            ckpts = sorted(_glob.glob(os.path.join(ckpt_dir, "*.pt")))
            if ckpts:
                latest = ckpts[-1]
                start_iter = int(os.path.basename(latest).split("_")[-1].split(".")[0])
                self.load(ckpt_dir, start_iter)
                logging.info("Resumed from step %d", start_iter)

        # iterators
        trainA_iter = iter(self.trainA_loader)
        trainB_iter = iter(self.trainB_loader)

        start_time = time.time()
        logging.info("Training kicks off 🚀")

        if hasattr(torch.compiler, "cudagraph_mark_step_begin"):
            torch.compiler.cudagraph_mark_step_begin()

        for step in range(start_iter, self.iteration + 1):
            # LR decay (old behaviour kept)
            if self.decay_flag and step > (self.iteration // 2):
                self.G_optim.param_groups[0]["lr"] -= self.lr / (self.iteration // 2)
                self.D_optim.param_groups[0]["lr"] -= self.lr / (self.iteration // 2)

            # fetch batch (restart dataloader if needed)
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

            real_A = real_A.to(self.device, non_blocking=True)
            real_B = real_B.to(self.device, non_blocking=True)

            # ----------------------
            #   1. Update D
            # ----------------------
            self.D_optim.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", enabled=self.amp_enabled):
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

                # adversarial losses (kept same)
                D_ad_loss_GA = self.MSE_loss(
                    real_GA_logit, torch.ones_like(real_GA_logit)
                ) + self.MSE_loss(fake_GA_logit, torch.zeros_like(fake_GA_logit))
                D_ad_cam_loss_GA = self.MSE_loss(
                    real_GA_cam_logit, torch.ones_like(real_GA_cam_logit)
                ) + self.MSE_loss(
                    fake_GA_cam_logit, torch.zeros_like(fake_GA_cam_logit)
                )
                D_ad_loss_LA = self.MSE_loss(
                    real_LA_logit, torch.ones_like(real_LA_logit)
                ) + self.MSE_loss(fake_LA_logit, torch.zeros_like(fake_LA_logit))
                D_ad_cam_loss_LA = self.MSE_loss(
                    real_LA_cam_logit, torch.ones_like(real_LA_cam_logit)
                ) + self.MSE_loss(
                    fake_LA_cam_logit, torch.zeros_like(fake_LA_cam_logit)
                )
                D_ad_loss_GB = self.MSE_loss(
                    real_GB_logit, torch.ones_like(real_GB_logit)
                ) + self.MSE_loss(fake_GB_logit, torch.zeros_like(fake_GB_logit))
                D_ad_cam_loss_GB = self.MSE_loss(
                    real_GB_cam_logit, torch.ones_like(real_GB_cam_logit)
                ) + self.MSE_loss(
                    fake_GB_cam_logit, torch.zeros_like(fake_GB_cam_logit)
                )
                D_ad_loss_LB = self.MSE_loss(
                    real_LB_logit, torch.ones_like(real_LB_logit)
                ) + self.MSE_loss(fake_LB_logit, torch.zeros_like(fake_LB_logit))
                D_ad_cam_loss_LB = self.MSE_loss(
                    real_LB_cam_logit, torch.ones_like(real_LB_cam_logit)
                ) + self.MSE_loss(
                    fake_LB_cam_logit, torch.zeros_like(fake_LB_cam_logit)
                )

                D_loss_A = self.adv_weight * (
                    D_ad_loss_GA + D_ad_cam_loss_GA + D_ad_loss_LA + D_ad_cam_loss_LA
                )
                D_loss_B = self.adv_weight * (
                    D_ad_loss_GB + D_ad_cam_loss_GB + D_ad_loss_LB + D_ad_cam_loss_LB
                )
                Discriminator_loss = D_loss_A + D_loss_B

            if self.amp_enabled:
                self.scaler_D.scale(Discriminator_loss).backward()
                self.scaler_D.step(self.D_optim)
                self.scaler_D.update()
            else:
                Discriminator_loss.backward()
                self.D_optim.step()

            # ----------------------
            #   2. Update G
            # ----------------------
            self.G_optim.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", enabled=self.amp_enabled):
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

                G_ad_loss_GA = self.MSE_loss(
                    fake_GA_logit, torch.ones_like(fake_GA_logit)
                )
                G_ad_cam_loss_GA = self.MSE_loss(
                    fake_GA_cam_logit, torch.ones_like(fake_GA_cam_logit)
                )
                G_ad_loss_LA = self.MSE_loss(
                    fake_LA_logit, torch.ones_like(fake_LA_logit)
                )
                G_ad_cam_loss_LA = self.MSE_loss(
                    fake_LA_cam_logit, torch.ones_like(fake_LA_cam_logit)
                )
                G_ad_loss_GB = self.MSE_loss(
                    fake_GB_logit, torch.ones_like(fake_GB_logit)
                )
                G_ad_cam_loss_GB = self.MSE_loss(
                    fake_GB_cam_logit, torch.ones_like(fake_GB_cam_logit)
                )
                G_ad_loss_LB = self.MSE_loss(
                    fake_LB_logit, torch.ones_like(fake_LB_logit)
                )
                G_ad_cam_loss_LB = self.MSE_loss(
                    fake_LB_cam_logit, torch.ones_like(fake_LB_cam_logit)
                )

                G_recon_loss_A = self.L1_loss(fake_A2B2A, real_A)
                G_recon_loss_B = self.L1_loss(fake_B2A2B, real_B)
                G_identity_loss_A = self.L1_loss(fake_A2A, real_A)
                G_identity_loss_B = self.L1_loss(fake_B2B, real_B)

                G_cam_loss_A = self.BCE_loss(
                    fake_B2A_cam_logit, torch.ones_like(fake_B2A_cam_logit)
                ) + self.BCE_loss(
                    fake_A2A_cam_logit, torch.zeros_like(fake_A2A_cam_logit)
                )
                G_cam_loss_B = self.BCE_loss(
                    fake_A2B_cam_logit, torch.ones_like(fake_A2B_cam_logit)
                ) + self.BCE_loss(
                    fake_B2B_cam_logit, torch.zeros_like(fake_B2B_cam_logit)
                )

                G_loss_A = (
                    self.adv_weight
                    * (
                        G_ad_loss_GA
                        + G_ad_cam_loss_GA
                        + G_ad_loss_LA
                        + G_ad_cam_loss_LA
                    )
                    + self.cycle_weight * G_recon_loss_A
                    + self.identity_weight * G_identity_loss_A
                    + self.cam_weight * G_cam_loss_A
                )
                G_loss_B = (
                    self.adv_weight
                    * (
                        G_ad_loss_GB
                        + G_ad_cam_loss_GB
                        + G_ad_loss_LB
                        + G_ad_cam_loss_LB
                    )
                    + self.cycle_weight * G_recon_loss_B
                    + self.identity_weight * G_identity_loss_B
                    + self.cam_weight * G_cam_loss_B
                )

                Generator_loss = G_loss_A + G_loss_B

            if self.amp_enabled:
                self.scaler_G.scale(Generator_loss).backward()
                self.scaler_G.step(self.G_optim)
                self.scaler_G.update()
            else:
                Generator_loss.backward()
                self.G_optim.step()

            # clip AdaILN / ILN
            self.genA2B.apply(self.Rho_clipper)
            self.genB2A.apply(self.Rho_clipper)

            # verbose print
            if step % self.print_freq == 0:
                elapsed = time.time() - start_time
                mem = torch.cuda.memory_allocated() / (1024**3)
                logging.info(
                    "[%d/%d] %.1fs | d_loss %.4f | g_loss %.4f | GPU %.2f GB",
                    step,
                    self.iteration,
                    elapsed,
                    Discriminator_loss.item(),
                    Generator_loss.item(),
                    mem,
                )

            # save rolling ckpt every save_interval
            if step % self.save_interval == 0:
                self.save(ckpt_dir, step)
                self._prune_checkpoints(ckpt_dir)

            # final save based on original save_freq
            if step % self.save_freq == 0:
                self.save(ckpt_dir, step)
                self._prune_checkpoints(ckpt_dir)

    # ----------------------------------------------------------------------
    # Checkpoint helpers
    # ----------------------------------------------------------------------
    def _ckpt_name(self, step: int):
        return f"{self.dataset}_params_{step:07d}.pt"

    def save(self, dir_path, step):
        os.makedirs(dir_path, exist_ok=True)
        params = {
            "genA2B": self.genA2B.state_dict(),
            "genB2A": self.genB2A.state_dict(),
            "disGA": self.disGA.state_dict(),
            "disGB": self.disGB.state_dict(),
            "disLA": self.disLA.state_dict(),
            "disLB": self.disLB.state_dict(),
        }
        ckpt_path = os.path.join(dir_path, self._ckpt_name(step))
        torch.save(params, ckpt_path)
        logging.info("Saved ckpt %s", ckpt_path)

    def _prune_checkpoints(self, dir_path):
        """Keep only the newest `self.keep_ckpts` checkpoints."""
        ckpts = sorted(
            _glob.glob(os.path.join(dir_path, f"{self.dataset}_params_*.pt"))
        )
        if len(ckpts) > self.keep_ckpts:
            for p in ckpts[: -self.keep_ckpts]:
                try:
                    os.remove(p)
                    logging.info("Pruned old ckpt %s", p)
                except OSError:
                    logging.warning("Could not remove %s", p)

    def load(self, dir_path, step):
        ckpt_path = os.path.join(dir_path, self._ckpt_name(step))
        params = torch.load(ckpt_path, map_location=self.device)
        self.genA2B.load_state_dict(params["genA2B"])
        self.genB2A.load_state_dict(params["genB2A"])
        self.disGA.load_state_dict(params["disGA"])
        self.disGB.load_state_dict(params["disGB"])
        self.disLA.load_state_dict(params["disLA"])
        self.disLB.load_state_dict(params["disLB"])
        logging.info("Loaded ckpt %s", ckpt_path)
