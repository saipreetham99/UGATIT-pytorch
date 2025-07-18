import os
import argparse
import cv2
import torch
from torchvision import transforms
from networks import ResnetGenerator
from utils import *

def parse_args():
    """Parses command-line arguments for single image testing."""
    desc = "Test a single image with a trained U-GAT-IT model"
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('--checkpoint', type=str, required=True, help='path to the checkpoint file')
    parser.add_argument('--input_image', type=str, required=True, help='path to the input image')
    parser.add_argument('--output_image', type=str, required=True, help='path to save the translated image')
    parser.add_argument('--direction', type=str, default='A2B', choices=['A2B', 'B2A'], help='translation direction')
    parser.add_argument('--img_size', type=int, default=256, help='image size used during training')
    parser.add_argument('--light', type=str2bool, default=False, help='was the model trained in light mode?')
    parser.add_argument('--ch', type=int, default=64, help='base channel number')
    parser.add_argument('--n_res', type=int, default=4, help='number of residual blocks')
    parser.add_argument('--device', type=str, default='cuda', choices=['cpu', 'cuda'], help='device to use')
    return parser.parse_args()

def main():
    """Main function to load model and translate an image."""
    args = parse_args()

    # --- Load Model ---
    print(f"Loading model for {args.direction} translation...")
    # Instantiate the correct generator
    generator = ResnetGenerator(input_nc=3, output_nc=3, ngf=args.ch, n_blocks=args.n_res, img_size=args.img_size, light=args.light).to(args.device)
    
    # Load the checkpoint
    try:
        checkpoint = torch.load(args.checkpoint, map_location=args.device)
        # Select the correct generator weights from the checkpoint
        gen_key = 'genA2B' if args.direction == 'A2B' else 'genB2A'
        generator.load_state_dict(checkpoint[gen_key])
        generator.eval()
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Error: Failed to load checkpoint. {e}")
        return

    # --- Image Pre-processing ---
    print(f"Loading and pre-processing input image: {args.input_image}")
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])
    
    try:
        image_bgr = cv2.imread(args.input_image)
        if image_bgr is None:
            raise FileNotFoundError(f"Could not read image file: {args.input_image}")
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        image_tensor = transform(image_rgb).unsqueeze(0).to(args.device)
    except Exception as e:
        print(f"Error: Failed to process input image. {e}")
        return

    # --- Image Translation ---
    print("Translating image...")
    with torch.no_grad():
        translated_tensor, _, _ = generator(image_tensor)

    # --- Post-processing and Saving ---
    print(f"Saving translated image to: {args.output_image}")
    output_image_numpy = denorm(translated_tensor.squeeze(0).cpu()).numpy()
    output_image_numpy = np.transpose(output_image_numpy, (1, 2, 0)) # HWC
    output_image_bgr = cv2.cvtColor(output_image_numpy, cv2.COLOR_RGB2BGR)
    
    cv2.imwrite(args.output_image, output_image_bgr * 255.0)
    print("Translation complete!")

if __name__ == '__main__':
    main()