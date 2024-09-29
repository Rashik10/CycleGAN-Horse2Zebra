import torch
import torch.nn as nn
import os
from torchvision.utils import save_image

def weights_init_normal(m):
    """
    Initializes the weights of the model with a normal distribution.
    This is commonly used in GANs for better convergence.
    """
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0.0)

def save_sample_images(generator_A2B, generator_B2A, real_A, real_B, epoch, batch, output_dir):
    """
    Generates and saves a set of images for visual inspection during training.
    
    Parameters:
    - generator_A2B: Generator that transforms domain A to B (e.g., horse to zebra)
    - generator_B2A: Generator that transforms domain B to A (e.g., zebra to horse)
    - real_A: Real images from domain A
    - real_B: Real images from domain B
    - epoch: Current epoch number
    - batch: Current batch number
    - output_dir: Directory to save the images
    """
    with torch.no_grad():
        fake_B = generator_A2B(real_A)
        fake_A = generator_B2A(real_B)
        rec_A = generator_B2A(fake_B)
        rec_B = generator_A2B(fake_A)

        os.makedirs(output_dir, exist_ok=True)
        
        # Save the real and generated images
        save_image(real_A, os.path.join(output_dir, f'real_A_epoch_{epoch}_batch_{batch}.png'), normalize=True)
        save_image(fake_B, os.path.join(output_dir, f'fake_B_epoch_{epoch}_batch_{batch}.png'), normalize=True)
        save_image(rec_A, os.path.join(output_dir, f'rec_A_epoch_{epoch}_batch_{batch}.png'), normalize=True)
        save_image(real_B, os.path.join(output_dir, f'real_B_epoch_{epoch}_batch_{batch}.png'), normalize=True)
        save_image(fake_A, os.path.join(output_dir, f'fake_A_epoch_{epoch}_batch_{batch}.png'), normalize=True)
        save_image(rec_B, os.path.join(output_dir, f'rec_B_epoch_{epoch}_batch_{batch}.png'), normalize=True)

def tensor2image(tensor):
    """
    Converts a PyTorch tensor into a numpy image array.
    
    Parameters:
    - tensor: PyTorch tensor to be converted
    
    Returns:
    - img: A numpy array representing the image
    """
    image = tensor.cpu().clone().detach().numpy()
    image = image.squeeze(0)
    image = (image + 1) / 2.0 * 255
    image = image.astype(np.uint8)
    return image

def save_checkpoint(state, checkpoint_dir, filename='checkpoint.pth.tar'):
    """
    Saves the model state, optimizer state, and any other information in the checkpoint.
    
    Parameters:
    - state: A dictionary containing model state and other necessary information.
    - checkpoint_dir: Directory where the checkpoint will be saved.
    - filename: Name of the checkpoint file.
    """
    filepath = os.path.join(checkpoint_dir, filename)
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    torch.save(state, filepath)

def load_checkpoint(checkpoint_path, model, optimizer=None):
    """
    Loads the model state from the checkpoint and optionally the optimizer state.
    
    Parameters:
    - checkpoint_path: Path to the checkpoint file.
    - model: Model object where the state will be loaded.
    - optimizer: Optimizer object where the state will be loaded (optional).
    
    Returns:
    - checkpoint: The checkpoint dictionary.
    """
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return checkpoint
