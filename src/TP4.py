import numpy as np
import matplotlib.pyplot as plt

from matplotlib.pyplot import cm 

c = (2 * np.pi) / 10 

mu = (2 * np.pi) / 5 
sigma = (2 * np.pi) / 10

x0 = 0 
xN = 2 * np.pi
Nx = 64
dx = (xN - x0) / Nx 

# wave numbers mesh (-N/2,...,N/2-1)
k = np.arange(-Nx/2, Nx/2, 1) 

x = np.linspace(start=x0, stop=xN, num=Nx) 

t0 = 0
tN = 10
dt = 0.001
Nt = int((tN - t0) / dt)

t = np.linspace(start=t0, stop=tN, num=Nt)

# solution mesh in real space
u = np.ones((Nx, Nt))
# solution mesh in fourier space
u_hat = np.ones((Nx, Nt), dtype="complex")

# initial condition
u0 = np.exp(-((x - mu) ** 2) / (2 * (sigma ** 2)))

# fourier transform of initial condition
u0_hat = (1 / Nx) * np.fft.fftshift(np.fft.fft(u0))

# set initial condition in real and fourier mesh 
u[:,0] = u0
u_hat[:,0] = u0_hat

# forward euler
for j in range(0, Nt-1):
  # compute solution in Fourier space through a finite difference method
  u_hat[:,j+1] = u[:,j] - dt * c * 1j * k * u_hat[:,j]
  # go back in real space 
  u[:,j] = np.real(Nx * np.fft.ifft(np.fft.ifftshift(u_hat[:,j])))
  
fig, ax = plt.subplots()

# get meshgrid
xx, tt = np.meshgrid(x, t)
# plot contour, get colorbar
# use transpose to get t first then x
cs = ax.contourf(xx, tt, u.T, cmap=cm.coolwarm)
# plot colorbar 
plt.colorbar(cs)

ax.set_xlabel("x")
ax.set_ylabel("t")
ax.set_title(f"Advection linéaire: c = {c}")

plt.show()