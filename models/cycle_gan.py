import torch
import torch.nn as nn
from models.networks import Generator, Discriminator

class CycleGAN(nn.Module):
    def __init__(self):
        super(CycleGAN, self).__init__()
        self.gen_A2B = Generator()
        self.gen_B2A = Generator()
        self.disc_A = Discriminator()
        self.disc_B = Discriminator()

    def forward(self, x_A, x_B):
        fake_B = self.gen_A2B(x_A)
        rec_A = self.gen_B2A(fake_B)
        fake_A = self.gen_B2A(x_B)
        rec_B = self.gen_A2B(fake_A)
        
        return fake_A, fake_B, rec_A, rec_B
