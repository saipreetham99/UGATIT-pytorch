import os
import cv2
import torch
import numpy as np
import imageio.v2 as imageio # The fix is on this line

# ------------------------------------------------------------------ #
# RhoClipper - The missing class
# ------------------------------------------------------------------ #
class RhoClipper(object):
    """
    A utility class to clip the rho parameter in AdaILN layers.
    This is used with model.apply(clipper) to recursively apply the clipping
    to all modules in the model.
    """
    def __init__(self, min_val, max_val):
        self.min = min_val
        self.max = max_val
        assert min_val < max_val, "min_val must be less than max_val for clipping."

    def __call__(self, module):
        """
        This method is called for each module in the network.
        """
        if hasattr(module, 'rho'):
            # If the module has a 'rho' attribute, clamp its value
            module.rho.data.clamp_(self.min, self.max)


# ------------------------------------------------------------------ #
# Data and Image handling functions
# ------------------------------------------------------------------ #

def load_test_data(image_path, size=256):
    """
    Loads and preprocesses a single image for testing.
    - Updated to use imageio and cv2 for reading and resizing.
    """
    img = imageio.imread(image_path, pilmode='RGB')
    img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
    img = np.expand_dims(img, axis=0)
    img = preprocessing(img)
    return img

def preprocessing(x):
    """
    Normalizes image data to the range [-1, 1].
    """
    return x / 127.5 - 1.0

def save_images(images, size, image_path):
    """
    Saves a grid of images.
    - Updated to use imageio.imwrite.
    """
    # Inverse transform scales image data from [-1, 1] to [0, 1]
    # and merges them into a single image grid.
    merged_image = merge(inverse_transform(images), size)
    # Convert to uint8 before saving
    merged_image_uint8 = (merged_image * 255).astype(np.uint8)
    return imageio.imwrite(image_path, merged_image_uint8)

def inverse_transform(images):
    """
    Scales image data from [-1, 1] back to [0, 1].
    """
    return (images + 1.0) / 2.0

def merge(images, size):
    """
    Merges a batch of images into a single grid.
    """
    h, w = images.shape[1], images.shape[2]
    num_images = images.shape[0]

    # Calculate grid size if not fully specified
    if isinstance(size, int):
        size = (int(np.ceil(num_images / size)), size)

    img = np.zeros((h * size[0], w * size[1], 3))

    for idx, image in enumerate(images):
        i = idx % size[1]
        j = idx // size[1]
        img[j * h: j * h + h, i * w: i * w + w, :] = image

    return img

def check_folder(log_dir):
    """
    Creates a directory if it doesn't exist.
    """
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    return log_dir

def str2bool(x):
    """
    Converts a string to a boolean.
    """
    return x.lower() in ('true', '1', 't', 'y', 'yes')

def cam(x, size=256):
    """
    Applies a color map to a CAM heatmap for visualization.
    """
    x = x - np.min(x)
    cam_img = x / (np.max(x) + 1e-8) # Add epsilon for stability
    cam_img = np.uint8(255 * cam_img)
    cam_img = cv2.resize(cam_img, (size, size))
    cam_img = cv2.applyColorMap(cam_img, cv2.COLORMAP_JET)
    return cam_img / 255.0

def imagenet_norm(x):
    """
    Normalizes a tensor with ImageNet stats.
    """
    mean = torch.FloatTensor([0.485, 0.456, 0.406]).unsqueeze(0).unsqueeze(2).unsqueeze(3).to(x.device)
    std = torch.FloatTensor([0.229, 0.224, 0.225]).unsqueeze(0).unsqueeze(2).unsqueeze(3).to(x.device)
    return (x - mean) / std

def denorm(x):
    """
    Denormalizes a tensor from [-1, 1] to [0, 1].
    """
    return x * 0.5 + 0.5

def tensor2numpy(x):
    """
    Converts a PyTorch tensor to a NumPy array.
    """
    return x.detach().cpu().numpy().transpose(1, 2, 0)

def RGB2BGR(x):
    """
    Converts an RGB image (NumPy array) to BGR.
    """
    return cv2.cvtColor(x, cv2.COLOR_RGB2BGR)