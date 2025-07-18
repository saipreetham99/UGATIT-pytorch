import argparse
from UGATIT import UGATIT
from utils import check_folder, str2bool
import os


def parse_args():
    desc = "U-GAT-IT trainer (B200‑optimised)"
    parser = argparse.ArgumentParser(description=desc)
    # unchanged args (trimmed for brevity)
    parser.add_argument("--phase", type=str, default="train", choices=["train", "test"])
    parser.add_argument("--light", type=str2bool, default=False)
    parser.add_argument("--dataset", type=str, default="YOUR_DATASET_NAME")
    parser.add_argument("--iteration", type=int, default=1_000_000)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--print_freq", type=int, default=1000)
    parser.add_argument("--save_freq", type=int, default=100_000)
    parser.add_argument("--decay_flag", type=str2bool, default=True)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--adv_weight", type=int, default=1)
    parser.add_argument("--cycle_weight", type=int, default=10)
    parser.add_argument("--identity_weight", type=int, default=10)
    parser.add_argument("--cam_weight", type=int, default=1000)
    parser.add_argument("--ch", type=int, default=64)
    parser.add_argument("--n_res", type=int, default=4)
    parser.add_argument("--n_dis", type=int, default=6)
    parser.add_argument("--img_size", type=int, default=256)
    parser.add_argument("--img_ch", type=int, default=3)
    parser.add_argument("--result_dir", type=str, default="results")
    parser.add_argument("--device", type=str, default="cuda", choices=["cpu", "cuda"])
    parser.add_argument("--benchmark_flag", type=str2bool, default=False)
    parser.add_argument("--resume", type=str2bool, default=False)

    # -------------- new knobs ---------------
    parser.add_argument(
        "--save_interval", type=int, default=2000, help="ckpt every N iters"
    )
    parser.add_argument("--keep_ckpts", type=int, default=3, help="max ckpts to keep")
    parser.add_argument(
        "--amp", type=str2bool, default=True, help="enable mixed precision"
    )

    args = parser.parse_args()

    # folders
    check_folder(os.path.join(args.result_dir, args.dataset, "model"))
    check_folder(os.path.join(args.result_dir, args.dataset, "img"))
    check_folder(os.path.join(args.result_dir, args.dataset, "test"))

    return args


def main():
    args = parse_args()
    gan = UGATIT(args)
    gan.build_model()

    if args.phase == "train":
        gan.train()
    else:
        gan.test()


if __name__ == "__main__":
    main()

