#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun  7 15:09:21 2020
@author: ben

## LJ parameters are based off: https://aip.scitation.org/doi/abs/10.1063/1.479848
"""

import time

import jax.numpy as np
import matplotlib.pyplot as plt
import os
#os.system("rm -rf ./__pycache__/")
import my_functions_normalizer as my_f 

import numpy as onp
onp.random.seed(16807)


### read xyz data

init_file = "./input_data/updated_38_Ico.xyz"
finl_file = "./input_data/updated_38_Oct.xyz"

init_config = my_f.read_xyz_file(init_file) 
finl_config = my_f.read_xyz_file(finl_file)


## Running Parameters
n = 32 ## number of neurons in hidden layer
d = 3 ## dimensionality of the system: 2 for 2D
Np = 38 ## number of particles
sigma = 1.0 #LJ sigma parameter
kb = 125.7
T0 = 20.0/kb

report_every = 1000 ## print loss every this many epochs
num_epochs = [100000, 100000] ## training epochs for pre and main training

### setup directories:
output_dir = "./Output_files/"
fig_dir = "./figures/"

box_dim = [10.0, 10.0, 10.0] ## box length in each dimension

### initialize parameters for the network: 
Params = my_f.init_nn_params(n, d, Np)

### time parameters:
t_incr, del_t_NN = 1.0, 0.01
## time domain for NN MD:
t0_nn, t1_nn = 0.0, t_incr
t_seq_NN = np.arange(t0_nn, t1_nn+0.5*del_t_NN, del_t_NN)
print(f"Run model for {len(t_seq_NN)} timesteps for {t_incr} tau")


############# relax the initial and final structure for a certain steps:
#### initial state:
qx0, qy0, qz0 = init_config
px0, py0, pz0 = onp.random.normal(loc=0.0, scale=1.0, size=(Np*d, )).reshape(d, -1)
px0, py0, pz0 = px0-np.mean(px0), py0-np.mean(py0), pz0-np.mean(pz0)
### initial temperature:
T = np.sum(px0**2 + py0**2 + pz0**2)/(3.0*Np)
px0, py0, pz0 = px0*onp.sqrt(T0/T), py0*onp.sqrt(T0/T), pz0*onp.sqrt(T0/T)
T = np.sum(px0**2 + py0**2 + pz0**2)/(3.0*Np)
print(f"Temperature = {T*kb} K")

### initilize running parameters for the model:
my_f.init_sys_params(n, d, sigma, [qx0, qy0, qz0], [px0, py0, pz0],
                     del_t_NN, Np, box_dim)

##### run MD to relax initial structures:
Nstep_0, Nstep_f = 1000, 1000
print(f"###### Using MD to relax initial config. for {Nstep_0} steps ... ######")
XTN, PTN, ATN, Etot_MD, Ekin_MD, Epot_MD = \
        my_f.run_MD(init_config, onp.asarray([px0, py0, pz0]), steps=Nstep_0, dt=0.001)
T = np.sum(PTN[-1][0]**2 + PTN[-1][1]**2 + PTN[-1][2]**2)/(3.0*Np)
print(f"Now after relaxation the temperature becomes {T*kb} K")
### write MD trajectory to file
save_to = output_dir+f"Q0_relax_{Nstep_0}_steps_via_MD.lammptrj"
my_f.write_trj(XTN[:, 0, :], XTN[:, 1, :], XTN[:, 2, :], save_to, True)
## define Q0 after relaxation:
Q0 = XTN[-1].reshape(Np*d, )

#### final state:
print(f"###### Using MD to relax final config. for {Nstep_0} steps ... ######")
qxf, qyf, qzf = finl_config
pxf, pyf, pzf = onp.random.normal(loc=0.0, scale=1.0, size=(Np*d, )).reshape(d, -1)
pxf, pyf, pzf = my_f.get_velocities( qxf, qyf, qzf, pxf, pyf, pzf, my_f.H0)

XTN, PTN, ATN, Etot_MD, Ekin_MD, Epot_MD = \
        my_f.run_MD(finl_config, onp.asarray([px0, py0, pz0]), steps=Nstep_f, dt=0.001)
T = np.sum(PTN[-1][0]**2 + PTN[-1][1]**2 + PTN[-1][2]**2)/(3.0*Np)
print(f"Now after relaxation the temperature becomes {T*kb} K")
### write MD trajectory to file
save_to = output_dir+f"Qf_relax_{Nstep_f}_steps_via_MD.lammptrj"
my_f.write_trj(XTN[:, 0, :], XTN[:, 1, :], XTN[:, 2, :], save_to, True)
## define Qf after relaxation:
Qf = XTN[-1].reshape(Np*d, )

print(" ###### MD part is done ###### ")



"""
### observe the MD energies:
t = np.arange(0.0, 1.0001, 0.001)
fig0, ax0 = plt.subplots(figsize=(7, 6))
ax0.plot(t, Etot_MD/Np, 'r-', lw=5, label="Total")
ax0.plot(t, Ekin_MD/Np, 'g-', lw=5, label="Ekin")
ax0.plot(t, Epot_MD/Np, 'b-', lw=5, label="Epot")
ax0.legend(prop={'size': 30}, loc='upper right', frameon=False)
"""
#### ~~~~~~~~~~~~~~~~~~~ end of MD part: ~~~~~~~~~~~~~~~















################ Pre-training:
ifcs = np.asarray([Q0, Qf])
Params = my_f.train_N0(Params, t_seq_NN, num_epochs[0], ifcs)

## compute trajectories for all particles:
Qt_0 = my_f.QT(Params, t_seq_NN)
onp.save(output_dir+'Qt_0.npy', Qt_0)
## compute total energy as a functin of time:
Epot_0, Ekin_0 = my_f.LJ3D_M(Qt_0[0])[0], 0.5*(np.sum(Qt_0[1]**2,axis=1))
Etot_0 = Ekin_0 + Epot_0
## save trained energies:
Epot_0 = onp.array([ t_seq_NN, Epot_0 ]).T
onp.save(output_dir+'Epot_0.npy', Epot_0)
Ekin_0 = onp.array([ t_seq_NN, Ekin_0 ]).T
onp.save(output_dir+'Ekin_0.npy', Ekin_0)
Etot_0 = onp.array([ t_seq_NN, Etot_0 ]).T
onp.save(output_dir+'Etot_0.npy', Etot_0)
#### write Qt[0] to file:
#my_f.write_Qt_to_file(Qt_0[0], output_dir+"Qt_0.lammpstrj")
my_f.write_Qt_Pt_to_file(Qt_0[0], Qt_0[1], output_dir+"Qt_0.lammpstrj")


## ~~~~~~~~~~~~~~~ start the main training:
### remember to use the 'real' qxf, qyf:
### store trajectory, energy vs t for all time segments:
Etot_NN, Ekin_NN, Epot_NN = [], [], []
### loss vs epoch:
loss_epoch = []
### start the main training:
t0 = time.time()
Params,loss_epoch = my_f.run_training(Params, t_seq_NN, ifcs, report_every,
                                      num_epochs[1], output_dir)
## compute x, y, vx, vy, ax, ay for all particles:
Qt = my_f.QT(Params, t_seq_NN)
onp.save(output_dir+'Qt_NN.npy', Qt)
save_as = output_dir+"Qt_NN.lammpstrj"
#my_f.write_Qt_to_file(Qt[0], save_as)
my_f.write_Qt_Pt_to_file(Qt[0], Qt[1], save_as)


## compute total energy as a functin of time
Epot_NN = my_f.LJ3D_M(Qt[0])[0]
Ekin_NN = 0.5*(np.sum(Qt[1]**2, axis=1))
Etot_NN = Ekin_NN + Epot_NN    


print(f"Total run time: {round(time.time()-t0, 1)} seconds")
    
Etot_NN = np.asarray([ t_seq_NN, Etot_NN ]).T
Epot_NN = np.asarray([ t_seq_NN, Epot_NN ]).T
Ekin_NN = np.asarray([ t_seq_NN, Ekin_NN ]).T    

## save to files:
onp.save(output_dir+'Params.npy', onp.asarray(Params))
onp.save(output_dir+'loss_vs_epoch.npy', onp.asarray(loss_epoch))
onp.save(output_dir+'Etot_NN.npy', onp.asarray(Etot_NN))
onp.save(output_dir+'Epot_NN.npy', onp.asarray(Epot_NN))
onp.save(output_dir+'Ekin_NN.npy', onp.asarray(Ekin_NN))




"""
################ plot total energy:
#plt.style.use("classic")
#from matplotlib.ticker import MultipleLocator
plt.rcdefaults()
#plt.rc('font', family='serif')
plt.rc('font',family='Times New Roman')
#plt.rc('font', weight='bold')
plt.rc('text', usetex=False)
from matplotlib.ticker import MultipleLocator

fig0, ax0 = plt.subplots(figsize=(7, 6))
ax0.plot(Etot_NN[:,0][::5], Etot_NN[:,1][::5]/Np, 'ro-', lw=5, ms=10, label="Total")
ax0.plot(Epot_NN[:,0][::5], Epot_NN[:,1][::5]/Np, 'bo-', lw=5, ms=10, label="Potential")
#ax0.plot(Ekin_NN[:,0][::5], Ekin_NN[:,1][::5]/Np, 'go-', lw=3, ms=8, label="Kinetic")

ax0.set_xlim([-0.1, t_seq_NN[-1]+0.1])
ax0.set_xticks(np.arange(0, t_seq_NN[-1]+0.00001, 0.2))
ax0.xaxis.set_minor_locator(MultipleLocator(0.1))
#ax0.set_xlabel(r"\textbf{Time ($\tau$)}", fontsize=25)
ax0.set_xlabel(r"Time ($\tau$)", fontsize=30)

ax0.set_ylim([-4.7, -3.8])
ax0.set_yticks(np.arange(-4.6, -4.0+0.0001, 0.2))
ax0.yaxis.set_minor_locator(MultipleLocator(0.1))
ax0.set_ylabel(r"Per-atom energy ($\epsilon$)", fontsize=30)
plt.tick_params(which='both', top=True, right=True, direction='in')
plt.tick_params(axis='both', which='major', pad=10, direction='in', length=8, width=5, colors='k',labelsize=30)
plt.tick_params(axis='both', which='minor', pad=10, direction='in', length=6, width=5, colors='k')
for axis in ['top','bottom','left','right']:
    ax0.spines[axis].set_linewidth(5.0)
fig0.tight_layout()
ax0.legend(prop={'size': 30}, loc='upper right', frameon=False)
#fig0.savefig(fig_dir+'./E_vs_t.png', transparent=True, dpi=400)
fig0.savefig(fig_dir+'E_vs_t_normal.svg')

"""
