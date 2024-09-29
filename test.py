import os
import torch
from torchvision import transforms
from torch.utils.data import DataLoader

from models.cycle_gan import CycleGAN
from utils.dataset import HorseZebraDataset
from utils.utils import save_sample_images, load_checkpoint

# Setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
checkpoint_dir = './checkpoints/latest'
output_dir = './outputs/test_results'
os.makedirs(output_dir, exist_ok=True)

# Transformations applied to the test images
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Load test data
dataset = HorseZebraDataset(root='./data/horse2zebra', transform=transform, mode='test')
dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

# Load model
model = CycleGAN().to(device)
load_checkpoint(os.path.join(checkpoint_dir, 'model.pth'), model)
model.eval()

# Testing loop
with torch.no_grad():
    for i, batch in enumerate(dataloader):
        real_A = batch['A'].to(device)
        real_B = batch['B'].to(device)

        # Generate fake images
        fake_B = model.gen_A2B(real_A)
        fake_A = model.gen_B2A(real_B)

        # Generate cycle images (A -> B -> A) and (B -> A -> B)
        rec_A = model.gen_B2A(fake_B)
        rec_B = model.gen_A2B(fake_A)

        # Save the results
        save_sample_images(model.gen_A2B, model.gen_B2A, real_A, real_B, i+1, 0, output_dir)

        print(f"Saved image {i+1}/{len(dataloader)}")

print(f"All test images saved in {output_dir}")
