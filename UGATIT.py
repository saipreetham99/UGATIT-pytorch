import time, itertools
from dataset import ImageFolder
from torchvision import transforms
from torch.utils.data import DataLoader
from networks import *
from utils import *
from glob import glob
import os

# Import for AMP
# from torch.cuda.amp import GradScaler, autocast


class UGATIT(object):
    def __init__(self, args):
        self.light = args.light

        if self.light:
            self.model_name = "UGATIT_light"
        else:
            self.model_name = "UGATIT"

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

        """ Weight """
        self.adv_weight = args.adv_weight
        self.cycle_weight = args.cycle_weight
        self.identity_weight = args.identity_weight
        self.cam_weight = args.cam_weight

        """ Generator """
        self.n_res = args.n_res

        """ Discriminator """
        self.n_dis = args.n_dis

        """ Checkpoint """
        self.checkpoint_window = args.checkpoint_window

        self.img_size = args.img_size
        self.img_ch = args.img_ch

        self.device = args.device
        self.benchmark_flag = args.benchmark_flag
        self.resume = args.resume

        # Check for CUDA availability
        if self.device == "cuda" and not torch.cuda.is_available():
            print("CUDA is not available, switching to CPU mode.")
            self.device = "cpu"

        if torch.backends.cudnn.enabled and self.benchmark_flag:
            print("set benchmark !")
            torch.backends.cudnn.benchmark = True

        print()

        print("##### Information #####")
        print("# light : ", self.light)
        print("# dataset : ", self.dataset)
        print("# batch_size : ", self.batch_size)
        print("# iteration : ", self.iteration)
        print("# checkpoint_window :", self.checkpoint_window)
        print("# device :", self.device)

        print()

        print("##### Generator #####")
        print("# residual blocks : ", self.n_res)

        print()

        print("##### Discriminator #####")
        print("# discriminator layer : ", self.n_dis)

        print()

        print("##### Weight #####")
        print("# adv_weight : ", self.adv_weight)
        print("# cycle_weight : ", self.cycle_weight)
        print("# identity_weight : ", self.identity_weight)
        print("# cam_weight : ", self.cam_weight)

    ##################################################################################
    # Model
    ##################################################################################

    def build_model(self):
        """DataLoader"""
        train_transform = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(),
                transforms.Resize((self.img_size + 30, self.img_size + 30)),
                transforms.RandomCrop(self.img_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
            ]
        )
        test_transform = transforms.Compose(
            [
                transforms.Resize((self.img_size, self.img_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
            ]
        )

        self.trainA = ImageFolder(
            os.path.join("dataset", self.dataset, "trainA"), train_transform
        )
        self.trainB = ImageFolder(
            os.path.join("dataset", self.dataset, "trainB"), train_transform
        )
        self.testA = ImageFolder(
            os.path.join("dataset", self.dataset, "testA"), test_transform
        )
        self.testB = ImageFolder(
            os.path.join("dataset", self.dataset, "testB"), test_transform
        )
        self.trainA_loader = DataLoader(
            self.trainA,
            batch_size=self.batch_size,
            shuffle=True,
            pin_memory=True,
            num_workers=4,
        )
        self.trainB_loader = DataLoader(
            self.trainB,
            batch_size=self.batch_size,
            shuffle=True,
            pin_memory=True,
            num_workers=4,
        )
        self.testA_loader = DataLoader(
            self.testA, batch_size=1, shuffle=False, pin_memory=True, num_workers=4
        )
        self.testB_loader = DataLoader(
            self.testB, batch_size=1, shuffle=False, pin_memory=True, num_workers=4
        )

        """ Define Generator, Discriminator """
        self.genA2B = ResnetGenerator(
            input_nc=3,
            output_nc=3,
            ngf=self.ch,
            n_blocks=self.n_res,
            img_size=self.img_size,
            light=self.light,
        ).to(self.device)
        self.genB2A = ResnetGenerator(
            input_nc=3,
            output_nc=3,
            ngf=self.ch,
            n_blocks=self.n_res,
            img_size=self.img_size,
            light=self.light,
        ).to(self.device)
        self.disGA = Discriminator(input_nc=3, ndf=self.ch, n_layers=7).to(self.device)
        self.disGB = Discriminator(input_nc=3, ndf=self.ch, n_layers=7).to(self.device)
        self.disLA = Discriminator(input_nc=3, ndf=self.ch, n_layers=5).to(self.device)
        self.disLB = Discriminator(input_nc=3, ndf=self.ch, n_layers=5).to(self.device)

        """ Define Loss """
        self.L1_loss = nn.L1Loss().to(self.device)
        self.MSE_loss = nn.MSELoss().to(self.device)
        self.BCE_loss = nn.BCEWithLogitsLoss().to(self.device)

        """ Trainer """
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

        """ AMP Grad Scaler """
        # Creates a GradScaler for each optimizer
        self.G_scaler = torch.GradScaler("cuda", enabled=(self.device == "cuda"))

        self.D_scaler = torch.GradScaler("cuda", enabled=(self.device == "cuda"))

        """ Define Rho clipper to constraint the value of rho in AdaILN and ILN"""
        self.Rho_clipper = RhoClipper(0, 1)

    def train(self):
        (
            self.genA2B.train(),
            self.genB2A.train(),
            self.disGA.train(),
            self.disGB.train(),
            self.disLA.train(),
            self.disLB.train(),
        )

        start_iter = 1
        if self.resume:
            model_list = glob(
                os.path.join(self.result_dir, self.dataset, "model", "*.pt")
            )
            if not len(model_list) == 0:
                model_list.sort()
                start_iter = int(model_list[-1].split("_")[-1].split(".")[0]) + 1
                self.load(
                    os.path.join(self.result_dir, self.dataset, "model"), start_iter - 1
                )
                print(f" [*] Load SUCCESS, resuming from iteration {start_iter}")
                if self.decay_flag and start_iter > (self.iteration // 2):
                    self.G_optim.param_groups[0]["lr"] -= (
                        self.lr / (self.iteration // 2)
                    ) * (start_iter - self.iteration // 2)
                    self.D_optim.param_groups[0]["lr"] -= (
                        self.lr / (self.iteration // 2)
                    ) * (start_iter - self.iteration // 2)

        # training loop
        print("training start !")
        start_time = time.time()

        trainA_iter = iter(self.trainA_loader)
        trainB_iter = iter(self.trainB_loader)

        for step in range(start_iter, self.iteration + 1):
            if self.decay_flag and step > (self.iteration // 2):
                self.G_optim.param_groups[0]["lr"] -= self.lr / (self.iteration // 2)
                self.D_optim.param_groups[0]["lr"] -= self.lr / (self.iteration // 2)

            try:
                real_A, _ = next(trainA_iter)
            except:
                trainA_iter = iter(self.trainA_loader)
                real_A, _ = next(trainA_iter)

            try:
                real_B, _ = next(trainB_iter)
            except:
                trainB_iter = iter(self.trainB_loader)
                real_B, _ = next(trainB_iter)

            real_A, real_B = real_A.to(self.device), real_B.to(self.device)

            # Update D
            self.D_optim.zero_grad()

            with torch.autocast("cuda", enabled=(self.device == "cuda")):
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

                D_ad_loss_GA = self.MSE_loss(
                    real_GA_logit, torch.ones_like(real_GA_logit).to(self.device)
                ) + self.MSE_loss(
                    fake_GA_logit, torch.zeros_like(fake_GA_logit).to(self.device)
                )
                D_ad_cam_loss_GA = self.MSE_loss(
                    real_GA_cam_logit,
                    torch.ones_like(real_GA_cam_logit).to(self.device),
                ) + self.MSE_loss(
                    fake_GA_cam_logit,
                    torch.zeros_like(fake_GA_cam_logit).to(self.device),
                )
                D_ad_loss_LA = self.MSE_loss(
                    real_LA_logit, torch.ones_like(real_LA_logit).to(self.device)
                ) + self.MSE_loss(
                    fake_LA_logit, torch.zeros_like(fake_LA_logit).to(self.device)
                )
                D_ad_cam_loss_LA = self.MSE_loss(
                    real_LA_cam_logit,
                    torch.ones_like(real_LA_cam_logit).to(self.device),
                ) + self.MSE_loss(
                    fake_LA_cam_logit,
                    torch.zeros_like(fake_LA_cam_logit).to(self.device),
                )
                D_ad_loss_GB = self.MSE_loss(
                    real_GB_logit, torch.ones_like(real_GB_logit).to(self.device)
                ) + self.MSE_loss(
                    fake_GB_logit, torch.zeros_like(fake_GB_logit).to(self.device)
                )
                D_ad_cam_loss_GB = self.MSE_loss(
                    real_GB_cam_logit,
                    torch.ones_like(real_GB_cam_logit).to(self.device),
                ) + self.MSE_loss(
                    fake_GB_cam_logit,
                    torch.zeros_like(fake_GB_cam_logit).to(self.device),
                )
                D_ad_loss_LB = self.MSE_loss(
                    real_LB_logit, torch.ones_like(real_LB_logit).to(self.device)
                ) + self.MSE_loss(
                    fake_LB_logit, torch.zeros_like(fake_LB_logit).to(self.device)
                )
                D_ad_cam_loss_LB = self.MSE_loss(
                    real_LB_cam_logit,
                    torch.ones_like(real_LB_cam_logit).to(self.device),
                ) + self.MSE_loss(
                    fake_LB_cam_logit,
                    torch.zeros_like(fake_LB_cam_logit).to(self.device),
                )

                D_loss_A = self.adv_weight * (
                    D_ad_loss_GA + D_ad_cam_loss_GA + D_ad_loss_LA + D_ad_cam_loss_LA
                )
                D_loss_B = self.adv_weight * (
                    D_ad_loss_GB + D_ad_cam_loss_GB + D_ad_loss_LB + D_ad_cam_loss_LB
                )
                Discriminator_loss = D_loss_A + D_loss_B

            self.D_scaler.scale(Discriminator_loss).backward()
            self.D_scaler.step(self.D_optim)
            self.D_scaler.update()

            # Update G
            self.G_optim.zero_grad()

            with torch.autocast("cuda", enabled=(self.device == "cuda")):
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
                    fake_GA_logit, torch.ones_like(fake_GA_logit).to(self.device)
                )
                G_ad_cam_loss_GA = self.MSE_loss(
                    fake_GA_cam_logit,
                    torch.ones_like(fake_GA_cam_logit).to(self.device),
                )
                G_ad_loss_LA = self.MSE_loss(
                    fake_LA_logit, torch.ones_like(fake_LA_logit).to(self.device)
                )
                G_ad_cam_loss_LA = self.MSE_loss(
                    fake_LA_cam_logit,
                    torch.ones_like(fake_LA_cam_logit).to(self.device),
                )
                G_ad_loss_GB = self.MSE_loss(
                    fake_GB_logit, torch.ones_like(fake_GB_logit).to(self.device)
                )
                G_ad_cam_loss_GB = self.MSE_loss(
                    fake_GB_cam_logit,
                    torch.ones_like(fake_GB_cam_logit).to(self.device),
                )
                G_ad_loss_LB = self.MSE_loss(
                    fake_LB_logit, torch.ones_like(fake_LB_logit).to(self.device)
                )
                G_ad_cam_loss_LB = self.MSE_loss(
                    fake_LB_cam_logit,
                    torch.ones_like(fake_LB_cam_logit).to(self.device),
                )

                G_recon_loss_A = self.L1_loss(fake_A2B2A, real_A)
                G_recon_loss_B = self.L1_loss(fake_B2A2B, real_B)

                G_identity_loss_A = self.L1_loss(fake_A2A, real_A)
                G_identity_loss_B = self.L1_loss(fake_B2B, real_B)

                G_cam_loss_A = self.BCE_loss(
                    fake_B2A_cam_logit,
                    torch.ones_like(fake_B2A_cam_logit).to(self.device),
                ) + self.BCE_loss(
                    fake_A2A_cam_logit,
                    torch.zeros_like(fake_A2A_cam_logit).to(self.device),
                )
                G_cam_loss_B = self.BCE_loss(
                    fake_A2B_cam_logit,
                    torch.ones_like(fake_A2B_cam_logit).to(self.device),
                ) + self.BCE_loss(
                    fake_B2B_cam_logit,
                    torch.zeros_like(fake_B2B_cam_logit).to(self.device),
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

            self.G_scaler.scale(Generator_loss).backward()
            self.G_scaler.step(self.G_optim)
            self.G_scaler.update()

            # clip parameter of AdaILN and ILN, applied after optimizer step
            self.genA2B.apply(self.Rho_clipper)
            self.genB2A.apply(self.Rho_clipper)

            # More verbose logging
            g_lr = self.G_optim.param_groups[0]["lr"]
            d_lr = self.D_optim.param_groups[0]["lr"]
            g_adv_loss = self.adv_weight * (
                G_ad_loss_GA
                + G_ad_cam_loss_GA
                + G_ad_loss_LA
                + G_ad_cam_loss_LA
                + G_ad_loss_GB
                + G_ad_cam_loss_GB
                + G_ad_loss_LB
                + G_ad_cam_loss_LB
            )
            g_cyc_loss = self.cycle_weight * (G_recon_loss_A + G_recon_loss_B)
            g_id_loss = self.identity_weight * (G_identity_loss_A + G_identity_loss_B)
            g_cam_loss = self.cam_weight * (G_cam_loss_A + G_cam_loss_B)

            print(
                f"[{step:6d}/{self.iteration}] time: {time.time() - start_time:4.2f}s "
                f"| LRs: [G:{g_lr:.6f}, D:{d_lr:.6f}] "
                f"| Losses: [D:{Discriminator_loss.item():.4f}, G:{Generator_loss.item():.4f}] "
                f"| G_comp: [adv:{g_adv_loss.item():.4f}, cyc:{g_cyc_loss.item():.4f}, id:{g_id_loss.item():.4f}, cam:{g_cam_loss.item():.4f}]"
            )
            start_time = time.time()  # Reset timer

            if step % self.print_freq == 0:
                self.genA2B.eval(), self.genB2A.eval()
                # Simplified the image saving part to reduce code duplication
                with torch.no_grad():
                    train_sample_num = 3
                    test_sample_num = 3
                    A2B = []
                    B2A = []

                    # Process training images
                    for loader, container in [
                        (iter(self.trainA_loader), A2B),
                        (iter(self.trainB_loader), B2A),
                    ]:
                        for _ in range(train_sample_num):
                            real_img, _ = next(loader)
                            real_img = real_img.to(self.device)
                            if container is A2B:
                                result_img = self.generate_sample_image(
                                    self.genA2B, self.genB2A, real_img
                                )
                                A2B.append(result_img)
                            else:
                                result_img = self.generate_sample_image(
                                    self.genB2A, self.genA2B, real_img
                                )
                                B2A.append(result_img)

                    # Process test images
                    for loader, container in [
                        (iter(self.testA_loader), A2B),
                        (iter(self.testB_loader), B2A),
                    ]:
                        for _ in range(test_sample_num):
                            real_img, _ = next(loader)
                            real_img = real_img.to(self.device)
                            if container is A2B:
                                result_img = self.generate_sample_image(
                                    self.genA2B, self.genB2A, real_img
                                )
                                A2B.append(result_img)
                            else:
                                result_img = self.generate_sample_image(
                                    self.genB2A, self.genA2B, real_img
                                )
                                B2A.append(result_img)

                    # Concatenate and save images
                    A2B = np.concatenate(A2B, 1)
                    B2A = np.concatenate(B2A, 1)

                    cv2.imwrite(
                        os.path.join(
                            self.result_dir, self.dataset, "img", f"A2B_{step:07d}.png"
                        ),
                        A2B * 255.0,
                    )
                    cv2.imwrite(
                        os.path.join(
                            self.result_dir, self.dataset, "img", f"B2A_{step:07d}.png"
                        ),
                        B2A * 255.0,
                    )

                self.genA2B.train(), self.genB2A.train()

            if step % self.save_freq == 0:
                self.save(os.path.join(self.result_dir, self.dataset, "model"), step)

                # Sliding window for checkpoints
                if self.checkpoint_window > 0:
                    model_dir = os.path.join(self.result_dir, self.dataset, "model")
                    model_list = glob(
                        os.path.join(model_dir, f"{self.dataset}_params_*.pt")
                    )
                    if len(model_list) > self.checkpoint_window:
                        model_list.sort(key=os.path.getmtime)
                        os.remove(model_list[0])
                        print(
                            f"Removed old checkpoint: {os.path.basename(model_list[0])}"
                        )

    def generate_sample_image(self, gen_forward, gen_backward, real_img):
        """Helper to generate a column of sample images."""
        fake_B, _, fake_B_heatmap = gen_forward(real_img)
        fake_B2A, _, fake_B2A_heatmap = gen_backward(fake_B)
        fake_A, _, fake_A_heatmap = gen_backward(real_img)

        return np.concatenate(
            (
                RGB2BGR(tensor2numpy(denorm(real_img[0]))),
                cam(tensor2numpy(fake_A_heatmap[0]), self.img_size),
                RGB2BGR(tensor2numpy(denorm(fake_A[0]))),
                cam(tensor2numpy(fake_B_heatmap[0]), self.img_size),
                RGB2BGR(tensor2numpy(denorm(fake_B[0]))),
                cam(tensor2numpy(fake_B2A_heatmap[0]), self.img_size),
                RGB2BGR(tensor2numpy(denorm(fake_B2A[0]))),
            ),
            0,
        )

    def save(self, dir, step):
        params = {}
        params["genA2B"] = self.genA2B.state_dict()
        params["genB2A"] = self.genB2A.state_dict()
        params["disGA"] = self.disGA.state_dict()
        params["disGB"] = self.disGB.state_dict()
        params["disLA"] = self.disLA.state_dict()
        params["disLB"] = self.disLB.state_dict()
        params["G_optim"] = self.G_optim.state_dict()
        params["D_optim"] = self.D_optim.state_dict()
        if self.device == "cuda":
            params["G_scaler"] = self.G_scaler.state_dict()
            params["D_scaler"] = self.D_scaler.state_dict()

        torch.save(params, os.path.join(dir, self.dataset + "_params_%07d.pt" % step))
        print(f"Checkpoint saved at iteration {step}")

    def load(self, dir, step):
        params = torch.load(
            os.path.join(dir, self.dataset + "_params_%07d.pt" % step),
            map_location=self.device,
        )
        self.genA2B.load_state_dict(params["genA2B"])
        self.genB2A.load_state_dict(params["genB2A"])
        self.disGA.load_state_dict(params["disGA"])
        self.disGB.load_state_dict(params["disGB"])
        self.disLA.load_state_dict(params["disLA"])
        self.disLB.load_state_dict(params["disLB"])

        if "G_optim" in params and "D_optim" in params:
            self.G_optim.load_state_dict(params["G_optim"])
            self.D_optim.load_state_dict(params["D_optim"])

        if self.device == "cuda" and "G_scaler" in params and "D_scaler" in params:
            self.G_scaler.load_state_dict(params["G_scaler"])
            self.D_scaler.load_state_dict(params["D_scaler"])

    def test(self):
        model_list = glob(os.path.join(self.result_dir, self.dataset, "model", "*.pt"))
        if not len(model_list) == 0:
            model_list.sort()
            iter = int(model_list[-1].split("_")[-1].split(".")[0])
            self.load(os.path.join(self.result_dir, self.dataset, "model"), iter)
            print(f" [*] Load SUCCESS: loaded checkpoint from iteration {iter}")
        else:
            print(" [*] Load FAILURE: No model found")
            return

        self.genA2B.eval(), self.genB2A.eval()
        with torch.no_grad():
            for n, (real_A, _) in enumerate(self.testA_loader):
                real_A = real_A.to(self.device)

                A2B_img = self.generate_sample_image(self.genA2B, self.genB2A, real_A)
                cv2.imwrite(
                    os.path.join(
                        self.result_dir, self.dataset, "test", f"A2B_{n + 1:04d}.png"
                    ),
                    A2B_img * 255.0,
                )

            for n, (real_B, _) in enumerate(self.testB_loader):
                real_B = real_B.to(self.device)

                B2A_img = self.generate_sample_image(self.genB2A, self.genA2B, real_B)
                cv2.imwrite(
                    os.path.join(
                        self.result_dir, self.dataset, "test", f"B2A_{n + 1:04d}.png"
                    ),
                    B2A_img * 255.0,
                )
