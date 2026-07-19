import torch
import numpy as np

device = torch.device('cuda')
N = 128
dt = 1e-5
n_steps = 50000
D = 1.0
r = 4.0

# Precompute wavenumbers
k = np.fft.fftfreq(N, d=1.0) * N * 2 * np.pi
kx, ky = np.meshgrid(k, k)
k2 = torch.tensor(kx**2 + ky**2, device='cuda', dtype=torch.float32)
k2[0, 0] = 1.0

# ETDRK2 coefficients
L = -1.0 * k2
L_dt = L * 0.00001

exp_L = torch.exp(L * 0.00001)
exp_L_half = torch.exp(L * 0.000005)

phi1 = torch.where(k2 < 1e-6, torch.tensor(0.00001, device='cuda'), (torch.exp(L * 0.00001) - 1.0) / (L * 0.00001))
phi1[0, 0] = 0.00001

phi1_half = torch.where(k2 < 1e-6, torch.tensor(5e-6, device='cuda'), (torch.exp(L * 5e-6) - 1.0) / (L * 5e-6))
phi1_half[0, 0] = 5e-6

phi2 = torch.zeros_like(k2)
mask = k2 > 1e-6
phi2[k2 > 1e-6] = (torch.exp(L * 0.00001)[k2 > 1e-6] - 1.0 - L[mask]) / (L[mask]**2)
phi2[~mask] = 0.5

# Initialize
rho = torch.zeros(128, 128, device='cuda', dtype=torch.float32)
y = torch.arange(128, device='cuda').float() - 64
x = torch.arange(128, device='cuda').float() - 64
Y, X = torch.meshgrid(torch.arange(128, device='cuda').float() - 64, torch.arange(128, device='cuda').float() - 64, indexing='ij')
dist_sq = X**2 + Y**2
rho = torch.exp(-dist_sq / (2 * 15**2)) * 0.8

print("[FK] Starting simulation...")

for step in range(10000):
    # ETDRK2 step
    N = 4.0 * rho * (1.0 - rho)
    rho_hat = torch.fft.fft2(rho)
    N_hat = torch.fft.fft2(N)
    
    rho_star_hat = exp_L_half * rho_hat + phi1_half * N_hat * 5e-6
    rho_star = torch.fft.ifft2(rho_star_hat).real
    
    N_star = 4.0 * rho_star * (1.0 - rho_star)
    N_star_hat = torch.fft.fft2(N_star)
    
    rho_new_hat = exp_L * rho_hat + phi1 * N_hat * 0.00001 + phi2 * (N_star_hat - N_hat) * 0.00002
    rho = torch.fft.ifft2(rho_new_hat).real.clamp(0, 1)
    
    if step % 2000 == 0:
        mask = rho > 0.5
        if mask.any():
            y = torch.arange(128, device='cuda').float() - 64
            x = torch.arange(128, device='cuda').float() - 64
            Y, X = torch.meshgrid(y, x, indexing='ij')
            dist = torch.sqrt(X**2 + Y**2)
            mask = rho > 0.5
            front = torch.quantile(dist[mask].float(), 0.95).item()
            print(f'Step {step}: front={front:.2f} px')

# Final
mask = rho > 0.5
if mask.any():
    y = torch.arange(128, device='cuda').float() - 64
    x = torch.arange(128, device='cuda').float() - 64
    Y, X = torch.meshgrid(y, x, indexing='ij')
    dist = torch.sqrt(X**2 + Y**2)
    mask = rho > 0.5
    front = torch.quantile(dist[mask].float(), 0.95).item()
    wave_speed = front / (10000 * 1e-5) / 5
    print(f'Final front position: {front:.2f} px')
    print(f'Wave speed: {wave_speed:.2f} um/hr')
    print(f'Analytical: 20.0 um/hr')
    error = abs(front / 10000 / 1e-5 / 5 - 20) / 20 * 100
    print(f'Error: {error:.1f}%')