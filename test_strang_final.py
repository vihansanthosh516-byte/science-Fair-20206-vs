import torch
import numpy as np
import time

device = 'cuda'
N = 256
dt = 5e-5
r = 4.0
D = 1.0
n_steps = 50000

# Precompute wavenumbers
k = np.fft.fftfreq(N) * N * 2 * np.pi
kx, ky = np.meshgrid(k, k)
k2 = torch.tensor(kx**2 + ky**2, device='cuda', dtype=torch.float32)
k2[0, 0] = 1.0

D = 1.0
r = 4.0
dt = 5e-5

# Precompute reaction exp terms
exp_half = torch.exp(torch.tensor(-4.0 * 2.5e-5, device='cuda', dtype=torch.float32))
exp_full = torch.exp(torch.tensor(-4.0 * 5e-5, device='cuda', dtype=torch.float32))

# Initialize
rho = torch.zeros(256, 256, device='cuda', dtype=torch.float32)
y = torch.arange(256, device='cuda').float() - 128
x = torch.arange(256, device='cuda').float() - 128
Y, X = torch.meshgrid(y, x, indexing='ij')
dist_sq = X**2 + Y**2
rho = torch.exp(-dist_sq / (2 * 15**2)) * 0.8

# Precompute wavenumbers
k = np.fft.fftfreq(256) * 256 * 2 * np.pi
kx, ky = np.meshgrid(k, k)
k2 = torch.tensor(kx**2 + ky**2, device='cuda', dtype=torch.float32)
k2[0, 0] = 1.0

D = 1.0
r = 4.0
dt = 5e-5

# Precompute Crank-Nicolson coefficients
denom = 1.0 + 0.5 * dt * k2
denom[0, 0] = 1.0
numer = 1.0 - 0.5 * dt * k2
numer[0, 0] = 1.0

# Precompute reaction exp terms
exp_half = torch.exp(torch.tensor(-4.0 * 2.5e-5, device='cuda', dtype=torch.float32))
exp_full = torch.exp(torch.tensor(-4.0 * 5e-5, device='cuda', dtype=torch.float32))

def detect_front(rho):
    mask = rho > 0.5
    if mask.any():
        y = torch.arange(256, device='cuda').float() - 128
        x = torch.arange(256, device='cuda').float() - 128
        Y, X = torch.meshgrid(y, x, indexing='ij')
        dist = torch.sqrt(X**2 + Y**2)
        mask = rho > 0.5
        if mask.any():
            return torch.quantile(dist[mask].float(), 0.95).item()
    return 0.0

# Initialize
rho = torch.zeros(256, 256, device='cuda', dtype=torch.float32)
y = torch.arange(256, device='cuda').float() - 128
x = torch.arange(256, device='cuda').float() - 128
Y, X = torch.meshgrid(y, x, indexing='ij')
dist_sq = X**2 + Y**2
rho = torch.exp(-dist_sq / (2 * 15**2)) * 0.8

n_steps = 20000
front_positions = []
times = []

t0 = time.perf_counter()
for step in range(20000):
    # R(dt/2)
    rho = torch.clamp(rho, 1e-12, 1.0 - 1e-12)
    exp_term = torch.exp(torch.tensor(-4.0 * dt/2, device='cuda', dtype=torch.float32))
    rho = rho / (rho + (1.0 - rho) * exp_term)
    rho = torch.clamp(rho, 0, 1)
    
    # Diffusion (Crank-Nicolson)
    rho_hat = torch.fft.fft2(rho)
    denom = 1.0 + 0.5 * dt * k2
    denom[0, 0] = 1.0
    numer = 1.0 - 0.5 * dt * k2
    numer[0, 0] = 1.0
    rho_new_hat = (numer / denom) * rho_hat
    rho = torch.fft.ifft2(rho_new_hat).real.clamp(0, 1)
    
    # R(dt/2)
    rho = rho / (rho + (1 - rho) * torch.exp(torch.tensor(-4.0 * dt/2, device='cuda', dtype=torch.float32)))
    rho = torch.clamp(rho, 0, 1)
    
    if step % 1000 == 0:
        mask = rho > 0.5
        if mask.any():
            y = torch.arange(256, device='cuda').float() - 128
            x = torch.arange(256, device='cuda').float() - 128
            Y, X = torch.meshgrid(y, x, indexing='ij')
            dist = torch.sqrt(X**2 + Y**2)
            mask = rho > 0.5
            dists = dist[mask].float()
            front = torch.quantile(dists, 0.95).item()
            print(f'Step {step}: front={front:.2f} px')

# Final measurement
mask = rho > 0.5
if mask.any():
    y = torch.arange(256, device='cuda').float() - 128
    x = torch.arange(256, device='cuda').float() - 128
    Y, X = torch.meshgrid(y, x, indexing='ij')
    dist = torch.sqrt(X**2 + Y**2)
    mask = rho > 0.5
    if mask.any():
        dists = dist[mask].float()
        front = torch.quantile(dists, 0.95).item()
        wave_speed = front / (5000 * 5e-5) / 5
        print(f'Final front: {front:.2f} px')
        print(f'Wave speed: {wave_speed:.2f} um/hr')
        print(f'Analytical: 20.0 um/hr')
        error = abs(front / 2000 / 5e-5 / 5 - 20) / 20 * 100
        print(f'Error: {error:.1f}%')

print('Done')