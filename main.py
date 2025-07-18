import os
import argparse
from UGATIT import UGATIT
from utils import *

def parse_args():
    """Parses command-line arguments."""
    desc = "PyTorch implementation of U-GAT-IT"
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('--phase', type=str, default='train', choices=['train', 'test'], help='train or test phase')
    parser.add_argument('--light', type=str2bool, default=False, help='use U-GAT-IT light version')
    parser.add_argument('--dataset', type=str, default='selfie2anime', help='dataset name')
    
    # Training parameters
    parser.add_argument('--iteration', type=int, default=1000000, help='total number of training iterations')
    parser.add_argument('--batch_size', type=int, default=1, help='batch size')
    parser.add_argument('--print_freq', type=int, default=1000, help='frequency of printing logs')
    parser.add_argument('--save_freq', type=int, default=1000, help='frequency of saving checkpoints and intermediate images')
    parser.add_argument('--decay_flag', type=str2bool, default=True, help='use learning rate decay')
    
    # Hyperparameters
    parser.add_argument('--lr', type=float, default=0.0001, help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.0001, help='weight decay')
    parser.add_argument('--adv_weight', type=int, default=1, help='weight for adversarial loss')
    parser.add_argument('--cycle_weight', type=int, default=10, help='weight for cycle consistency loss')
    parser.add_argument('--identity_weight', type=int, default=10, help='weight for identity loss')
    parser.add_argument('--cam_weight', type=int, default=1000, help='weight for CAM loss')

    # Model architecture
    parser.add_argument('--ch', type=int, default=64, help='base channel number')
    parser.add_argument('--n_res', type=int, default=4, help='number of residual blocks')
    parser.add_argument('--n_dis', type=int, default=6, help='number of discriminator layers')

    # Image and environment settings
    parser.add_argument('--img_size', type=int, default=256, help='image size')
    parser.add_argument('--img_ch', type=int, default=3, help='image channels')
    parser.add_argument('--result_dir', type=str, default='results', help='directory to save results')
    parser.add_argument('--device', type=str, default='cuda', choices=['cpu', 'cuda'], help='device to use')
    parser.add_argument('--benchmark_flag', type=str2bool, default=False, help='use cudnn benchmark')
    parser.add_argument('--resume', type=str2bool, default=False, help='resume training from latest checkpoint')

    return check_args(parser.parse_args())

def check_args(args):
    """Checks and creates necessary directories."""
    check_folder(os.path.join(args.result_dir, args.dataset, 'model'))
    check_folder(os.path.join(args.result_dir, args.dataset, 'img')) # For intermediate images
    check_folder(os.path.join(args.result_dir, args.dataset, 'test'))
    
    assert args.batch_size >= 1, 'batch size must be at least 1'
    return args

def main():
    """Main function to run the training or testing process."""
    args = parse_args()
    if args is None:
        exit()

    gan = UGATIT(args)
    gan.build_model()

    if args.phase == 'train':
        print("Starting training phase...")
        gan.train()
        print("[*] Training finished!")

    elif args.phase == 'test':
        print("Starting testing phase...")
        gan.test()
        print("[*] Test finished!")

if __name__ == '__main__':
    main()