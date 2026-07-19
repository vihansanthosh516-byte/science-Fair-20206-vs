import torch
import numpy as np

# Test the reaction step
r = 4.0
dt = 5e-5
rho = torch.tensor(0.5, device='cuda')
exp_term = torch.exp(torch.tensor(-4.0 * 2.5e-5, device='cuda', dtype=torch.float32))
rho_new = 0.5 / (0.5 + 0.5 * torch.exp(torch.tensor(-4.0 * 2.5e-5, device='cuda')))
print('Reaction step test:')
print(f'  rho = 0.5')
print(f'  exp_term = {torch.exp(torch.tensor(-4.0 * 2.5e-5, device="cuda"))}')
rho_new = 0.5 / (0.5 + 0.5 * torch.exp(torch.tensor(-4.0 * 2.5e-5, device='cuda')))
print(f'  rho_new = {rho_new}')

# Test diffusion
dt = 5e-5
D = 1.0
k = np.fft.fftfreq(256) * 256 * 2 * np.pi
kx, ky = np.meshgrid(k, k)
k2 = torch.tensor(kx**2 + ky**2, device='cuda', dtype=torch.float32)
k2[0, 0] = 1.0

rho = torch.zeros(256, 256, device='cuda')
rho[118:276, 118:276] = 0.5

rho_hat = torch.fft.fft2(torch.ones(256, 256, device='cuda') * 0.5)
denom = 1.0 + 0.5 * 5e-5 * k2
denom[0, 0] = 1.0
numer = 1.0 - 0.5 * 5e-5 * k2
numer[0, 0] = 1.0

rho_new_hat = (numer / denom) * torch.fft.fft2(torch.ones(256, 256, device='cuda') * 0.5)
rho_new = torch.fft.ifft2(rho_new_hat).real.clamp(0, 1)
print(f'Mean after diffusion: {torch.mean(rho_new).item():.4f}')