#Quantum Tunnelling simulation using Split-Step Fourier Method
import numpy as np
import matplotlib.pyplot as plt
from vpython import *
import time
time.sleep(2.0)
scene=canvas()
g1=graph(width=600,height=400,xtitle="position x(m)",ytitle="wave function",title="wave function")
f1=gcurve(color=color.red,)
f2=gcurve(color=color.black,label="potential")

length=8
m=1
hbar =1
grid=np.linspace(-length,length,1000)
dx=grid[1]-grid[0]
sigma=0.2
p=4
v0=p**2/(2*m)
x0=-3
n=len(grid)
v_x=np.zeros(n)
print("Potential values initialized.")
for i in range(len(v_x)):
    if grid[i]>0 and grid[i]<0.5:
        v_x[i]=v0*2
one=np.ones(n)
intial_psi=(1/(sigma*np.sqrt(np.pi)))**(1/2)*np.exp(-(grid-x0)**2/(2*sigma**2))*np.exp(1j*p*grid/hbar)
norm_psi=(1/(sigma*np.sqrt(np.pi)))**(1/2)*np.exp(-(grid-x0)**2/(2*sigma**2))*np.exp(1j*p*grid/hbar)/(np.sqrt(np.sum(np.abs(intial_psi)**2)*dx))
print("Initial wave function initialized.")
matrix=(-np.diag(np.ones(n-1),-1)-np.diag(np.ones(n-1),1)+np.diag(2*np.ones(n),0))*hbar**2/(((dx**2)*(2* m)))+np.diag(v_x,0)

eiganvalue,eiganvector=np.linalg.eigh(matrix)
for k in range(n):
    eiganvector[:,k] /= np.sqrt(
        np.sum(np.abs(eiganvector[:,k])**2*dx))

psi = eiganvector.T
c=np.zeros(n,dtype=complex)
norm_psi=intial_psi/np.sqrt(np.sum(np.abs(intial_psi)**2*dx))
for i in range(len(v_x)):
    f2.plot(grid[i],0.3*v_x[i])
print("Potential plot initialized.")
for i in range(len(c)):
    c[i]=np.sum(np.conj(psi[i])*norm_psi*dx)

print("Coefficients initialized.")
t=0
dt=0.003
print("tunnuneling simulation started.")
while t<2:
    rate(60)  # Locked at a smooth 60 FPS
    psi_current=np.zeros(n,dtype=complex)
    value=[]
    for i in range(len(c)):
        psi_current+=c[i]*psi[i]*np.exp(-1j*eiganvalue[i]*t/hbar)
    for i in range(len(psi_current)):
        
        value=value+[[grid[i],2*np.abs(psi_current[i])**2]]
    f1.data=value
    t=t+dt
# Find the exact index where the barrier begins (where grid x tightly approaches 0)
barrier_start_idx = np.argmin(np.abs(grid - 0.0))
# Capture the simulation's wave amplitude value right at the barrier entrance boundary
amplitude_at_boundary = np.abs(psi_current[barrier_start_idx])**2

# We map our WKB evaluation against the expected packet energy level
E_packet = p**2 / (2 * m)

i=0
psi_wkb=np.zeros(n)
while i<len(grid):
    if v_x[i]>=E_packet:
        #Compute physical kappa decay factor
        kappa = np.sqrt(2 * m * (v_x[i] - E_packet)) / hbar
        #  Distance is calculated relative to the barrier edge (grid[i] - 0.0)
        relative_distance = grid[i] - 0.0
        #  Scale decay curve using the boundary amplitude benchmark
        psi_wkb[i] = amplitude_at_boundary * np.exp(-2 * kappa * relative_distance)

    else:
        psi_wkb[i]=np.abs(psi_current[i])**2
        
    i+=1
plt.plot(grid,2*psi_wkb,label="WKB approximation",color="g")
plt.plot(grid,2*np.abs(psi_current)**2,label="simulation results",color="b")
plt.xlabel("Position (m)")
plt.ylabel("Probability Densitycomparision")
plt.title("Quantum Tunnelling Simulation")
plt.legend()
plt.show()
error=(np.abs((psi_wkb-np.abs(psi_current)**2)/(np.abs(psi_current)**2)))
plt.plot(grid,error,color="r")
plt.xlabel("Position (m)")
plt.ylabel("Error (%)")
plt.title("Error between WKB approximation and simulation results") 
plt.grid(True,alpha=0.3)
plt.show()