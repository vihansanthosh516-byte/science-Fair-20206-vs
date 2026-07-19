import torch
import numpy as np

device = torch.device('cuda')
N = 256
dt = 5e-5
r = 4.0
D = 1.0
n_steps = 20000

# Initialize
rho = torch.zeros(256, 256, device='cuda', dtype=torch.float32)
y = torch.arange(256, device='cuda').float() - 128
x = torch.arange(256, device='cuda').float() - 128
Y, X = torch.meshgrid(y, x, indexing='ij')
dist_sq = X**2 + Y**2
rho = torch.exp(-dist_sq / (2 * 15**2)) * 0.8

# Precompute wavenumbers
kx = torch.fft.fftfreq(256, device='cuda') * 256 * 2 * np.pi
ky = torch.fft.fftfreq(256, device='cuda') * 2 * np.pi
kx, ky = torch.meshgrid(kx, ky, indexing='ij')
k2 = kx**2 + ky**2
k2[0, 0] = 1.0

D = 1.0
r = 4.0
dt = 5e-5

def reaction_exact(rho, dt):
    rho = torch.clamp(rho, 1e-12, 1.0 - 1e-12)
    exp_term = torch.exp(torch.tensor(-4.0 * dt, device='cuda', dtype=torch.float32))
    rho_new = rho / (rho + (1.0 - rho) * exp_term)
    return torch.clamp(rho_new, 0.0, 1.0)

def diffusion_step(rho, dt):
    rho_hat = torch.fft.fft2(rho)
    denom = 1.0 + 0.5 * 5e-5 * k2
    denom[0, 0] = 1.0
    numer = 1.0 - 0.5 * 5e-5 * k2
    numer[0, 0] = 1.0
    rho_new_hat = (numer / denom) * rho_hat
    return torch.fft.ifft2(rho_new_hat).real.clamp(0.0, 1.0)

def detect_front(rho):
    mask = rho > 0.5
    if mask.any():
        y = torch.arange(256, device='cuda').float() - 128
        x = torch.arange(256, device='cuda').float() - 128
        Y, X = torch.meshgrid(y, x, indexing='ij')
        dist = torch.sqrt(X**2 + Y**2)
        mask = rho > 0.5
        if mask.any():
            return dist[mask].float().mean().item()
    return 0.0

# Initialize
rho = torch.zeros(256, 256, device='cuda', dtype=torch.float32)
y = torch.arange(256, device='cuda').float() - 128
x = torch.arange(256, device='cuda').float() - 128
Y, X = torch.meshgrid(y, x, indexing='ij')
dist_sq = X**2 + Y**2
rho = torch.exp(-dist_sq / (2 * 15**2)) * 0.8

# Precompute k2 for diffusion
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

def reaction_step(rho, dt):
    rho = torch.clamp(rho, 1e-12, 1.0 - 1e-12)
    exp_term = torch.exp(torch.tensor(-4.0 * dt, device='cuda', dtype=torch.float32))
    rho_new = rho / (rho + (1.0 - rho) * torch.exp(torch.tensor(-4.0 * dt, device='cuda')))
    return torch.clamp(rho, 0, 1)

def diffusion_step(rho, dt):
    rho_hat = torch.fft.fft2(rho)
    denom = 1.0 + 0.5 * dt * k2
    denom[0, 0] = 1.0
    numer = 1.0 - 0.5 * dt * k2
    numer[0, 0] = 1.0
    rho_new_hat = (numer / denom) * rho_hat
    return torch.fft.ifft2(rho_new_hat).real.clamp(0, 1)

def step(rho):
    # R(dt/2)
    rho = reaction_exact(rho, dt / 2.0)
    # D(dt)
    rho = diffusion_step(rho, dt)
    # R(dt/2)
    rho = reaction_exact(rho, dt / 2.0)
    return torch.clamp(rho, 0.0, 1.0)

def detect_front(rho):
    mask = rho > 0.5
    if mask.any():
        y = torch.arange(256, device='cuda').float() - 128
        x = torch.arange(256, device='cuda').float() - 128
        Y, X = torch.meshgrid(y, x, indexing='ij')
        dist = torch.sqrt(X**2 + Y**2)
        mask = rho > 0.5
        if mask.any():
            return dist[mask].float().mean().item()
    return 0.0

# Run simulation
rho = torch.zeros(256, 256, device='cuda', dtype=torch.float32)
y = torch.arange(256, device='cuda').float() - 128
x = torch.arange(256, device='cuda').float() - 128
Y, X = torch.meshgrid(y, x, indexing='ij')
dist_sq = X**2 + Y**2
rho = torch.exp(-dist_sq / (2 * 15**2)) * 0.8

for step in range(100):
    rho = step(rho)
    if step % 100 == 0:
        front = detect_front(rho)
        print(f'Step {step}: front={front:.2f} px')

# Final
front = detect_front(rho)
wave_speed = front / (1000 * 5e-5) / 5
print(f'Final front: {front:.2f} px')
print(f'Wave speed: {front / 1000 / 5e-5 / 5:.2f} um/hr')
print(f'Analytical: 20.0 um/hr')
error = abs(front / 1000 / 5e-5 / 5 - 20) / 20 * 100
print(f'Error: {error:.1f}%')