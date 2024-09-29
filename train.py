import os
import itertools
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import save_image
import matplotlib.pyplot as plt
import numpy as np

from models.cycle_gan import CycleGAN
from utils.dataset import HorseZebraDataset
from utils.utils import weights_init_normal, save_sample_images, save_checkpoint, load_checkpoint

# Hyperparameters
num_epochs = 100
batch_size = 4
learning_rate = 0.0002
save_interval = 10
checkpoint_dir = './checkpoints/latest'
output_dir = './outputs/models'
loss_file = './checkpoints/latest/losses.npz'

# Setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Load Data
dataset = HorseZebraDataset(root='./data/horse2zebra', transform=transform)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Initialize Model
model = CycleGAN().to(device)
model.apply(weights_init_normal)  # Initialize weights

# Loss functions
criterion_GAN = nn.MSELoss()
criterion_cycle = nn.L1Loss()

# Optimizers
optimizer_G = optim.Adam(itertools.chain(model.gen_A2B.parameters(), model.gen_B2A.parameters()), lr=learning_rate, betas=(0.5, 0.999))
optimizer_D = optim.Adam(itertools.chain(model.disc_A.parameters(), model.disc_B.parameters()), lr=learning_rate, betas=(0.5, 0.999))

# Load checkpoint and losses if they exist
start_epoch = 0
G_losses, D_losses = [], []

if os.path.exists(os.path.join(checkpoint_dir, 'model.pth')):
    checkpoint = load_checkpoint(os.path.join(checkpoint_dir, 'model.pth'), model, optimizer_G)
    optimizer_D.load_state_dict(checkpoint['optimizer_D_state_dict'])
    start_epoch = checkpoint['epoch'] + 1

    # Load previous losses if they exist
    if os.path.exists(loss_file):
        losses = np.load(loss_file)
        G_losses = losses['G_losses'].tolist()
        D_losses = losses['D_losses'].tolist()

# Training Loop
for epoch in range(start_epoch, num_epochs):
    for i, batch in enumerate(dataloader):
        real_A = batch['A'].to(device)
        real_B = batch['B'].to(device)

        # Get the output size of the discriminator
        # Pass real_A through disc_A to determine the output size
        disc_out = model.disc_A(real_A)
        target_size = disc_out.shape[2:]  # This gets the height and width (e.g., (30, 30))

        # Adversarial ground truths (now using the dynamically computed target size)
        valid = torch.ones((real_A.size(0), 1, *target_size), requires_grad=False).to(device)
        fake = torch.zeros((real_A.size(0), 1, *target_size), requires_grad=False).to(device)

        # ---------------------
        #  Train Generators
        # ---------------------
        optimizer_G.zero_grad()

        # Identity loss
        loss_id_A = criterion_cycle(model.gen_B2A(real_A), real_A)
        loss_id_B = criterion_cycle(model.gen_A2B(real_B), real_B)

        # GAN loss
        fake_B = model.gen_A2B(real_A)
        loss_GAN_A2B = criterion_GAN(model.disc_B(fake_B), valid)
        fake_A = model.gen_B2A(real_B)
        loss_GAN_B2A = criterion_GAN(model.disc_A(fake_A), valid)

        # Cycle loss
        recovered_A = model.gen_B2A(fake_B)
        loss_cycle_A = criterion_cycle(recovered_A, real_A)
        recovered_B = model.gen_A2B(fake_A)
        loss_cycle_B = criterion_cycle(recovered_B, real_B)

        # Total generator loss
        loss_G = loss_GAN_A2B + loss_GAN_B2A + 10.0 * (loss_cycle_A + loss_cycle_B) + 5.0 * (loss_id_A + loss_id_B)
        loss_G.backward()
        optimizer_G.step()

        # ---------------------
        #  Train Discriminators
        # ---------------------
        optimizer_D.zero_grad()

        # Discriminator A
        loss_real_A = criterion_GAN(model.disc_A(real_A), valid)
        loss_fake_A = criterion_GAN(model.disc_A(fake_A.detach()), fake)
        loss_D_A = (loss_real_A + loss_fake_A) * 0.5
        loss_D_A.backward()

        # Discriminator B
        loss_real_B = criterion_GAN(model.disc_B(real_B), valid)
        loss_fake_B = criterion_GAN(model.disc_B(fake_B.detach()), fake)
        loss_D_B = (loss_real_B + loss_fake_B) * 0.5
        loss_D_B.backward()

        optimizer_D.step()

        # Logging the losses
        G_losses.append(loss_G.item())
        D_losses.append((loss_D_A.item() + loss_D_B.item()) / 2)

        print(f"Epoch [{epoch+1}/{num_epochs}] Batch {i+1}/{len(dataloader)} "
              f"Loss G: {loss_G.item():.4f}, Loss D A: {loss_D_A.item():.4f}, Loss D B: {loss_D_B.item():.4f}")

        # Save some generated samples every 100 steps
        if i % 100 == 0:
            save_sample_images(model.gen_A2B, model.gen_B2A, real_A, real_B, epoch + 1, i + 1, 'outputs/images')

    # Save model and losses every `save_interval` epochs
    if (epoch + 1) % save_interval == 0:
        save_checkpoint({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_G_state_dict': optimizer_G.state_dict(),
            'optimizer_D_state_dict': optimizer_D.state_dict(),
        }, checkpoint_dir)

        np.savez(loss_file, G_losses=G_losses, D_losses=D_losses)

        torch.save(model.state_dict(), os.path.join(output_dir, f'cycle_gan_epoch_{epoch+1}.pth'))

        print(f"Checkpoint saved at epoch {epoch+1}")


# Plotting the loss curves
plt.figure()
plt.plot(G_losses, label='Generator Loss')
plt.plot(D_losses, label='Discriminator Loss')
plt.xlabel('Iterations')
plt.ylabel('Loss')
plt.legend()
plt.savefig('outputs/loss_curve.png')
plt.show()
