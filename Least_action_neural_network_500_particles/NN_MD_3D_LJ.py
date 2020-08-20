import time
start = time.time()
import jax.numpy as np
import matplotlib.pyplot as plt
import my_function_3D as my_f # load from my_functions.py
import numpy as onp
onp.random.seed(16807)

## Running Parameters
n = 125 ## number of neurons in hidden layer
d = 3 ## dimensionality of the system: 2 for 2D
Np = 500 ## number of particles
sigma = 1.0 #LJ sigma parameter
## box length in each dimension:
box_dim = [8.5945161556157537, 8.5945161556157537, 8.5945161556157537]
MDsteps = 1 ## number of time segments
Nreport = 1000 ## print loss every this many epochs
num_epochs = [200000, 200000] ## Loss0 training, Real training
## input and output directories:
input_file = f"../../3D_{Np}_particle/{Np}_particle_last_frame.lammpstrj"
output_dir = "./Output_files/"
epoch_dir = "./E_Q_vs_epoch/"

### time parameters:
t_incr, del_t_MD, del_t = 0.5, 0.001, 0.02
## time domain for ground-truth MD:
t0_md, t1_md = 0.0, t_incr
t_seq_MD = np.arange(t0_md, t1_md+0.5*del_t_MD, del_t_MD)
## time domain for NN MD:
t0_nn, t1_nn = 0.0, t_incr
t_seq_NN = np.arange(t0_nn, t1_nn+0.5*del_t, del_t)

## initilize parameters 
my_f.init_sys_params(n, d, sigma, del_t, Np, box_dim, input_file, 9)
#### initial state from lammps data:
Q0, P0 = my_f.Q0, my_f.P0 

#### ~~~~~~~~~~~~~~~~~~~ MD part: ~~~~~~~~~~~~~~~
steps = len(t_seq_MD)-1 ## number of MD steps
print(f" ###### MD is running, {Np} particles & {steps} steps ... ######")
XTN, PTN, ATN, Etot_MD, Ekin_MD, Epot_MD = my_f.run_MD(Q0, P0, steps, del_t_MD)
Etot_MD = np.array([t_seq_MD, Etot_MD]).T
Ekin_MD = np.array([t_seq_MD, Ekin_MD]).T
Epot_MD = np.array([t_seq_MD, Epot_MD]).T
## save XTN, PTN, ATN
onp.save(output_dir+'XTN.npy', XTN)
onp.save(output_dir+'PTN.npy', PTN)
onp.save(output_dir+'ATN.npy', ATN)
onp.save(output_dir+'Etot_MD.npy', Etot_MD)
onp.save(output_dir+'Epot_MD.npy', Epot_MD)
onp.save(output_dir+'Ekin_MD.npy', Ekin_MD)
### write MD trajectory to file
save_to = output_dir+"MD_trj.lammptrj"
my_f.write_trj(XTN[:, 0, :], XTN[:, 1, :], XTN[:, 2, :], save_to, True)
print(" ###### MD part is done ###### ")
#### ~~~~~~~~~~~~~~~~~~~ end of MD part: ~~~~~~~~~~~~~~~

## initialize parameters and system state: 
Params = my_f.init_nn_params(my_f.n, my_f.Np)
q0, qf = XTN[0].flatten(),  XTN[-1].flatten()
p0, pf = PTN[0].flatten(),  PTN[-1].flatten()
ifcs = np.asarray([q0, qf, p0, pf])
## start the pre-training:
Params = my_f.train_N0(Params, t_seq_MD, num_epochs[0], ifcs)

## epoch 0 energies and Qt:
Qt_0 = my_f.QT(Params, t_seq_NN)
Epot_0 = my_f.LJ3D_M(Qt_0[0])[0]
Ekin_0 = 0.5*(np.sum(Qt_0[1]**2,axis=1))
Etot_0 = Ekin_0 + Epot_0
## save to files:
onp.save(epoch_dir+'Qt_0.npy', Qt_0)
Epot_0 = onp.array([t_seq_NN, Epot_0]).T
onp.save(epoch_dir+'Epot_0.npy', Epot_0)
Ekin_0 = onp.array([t_seq_NN, Ekin_0]).T
onp.save(epoch_dir+'Ekin_0.npy', Ekin_0)
Etot_0 = onp.array([t_seq_NN, Etot_0]).T
onp.save(epoch_dir+'Etot_0.npy', Etot_0)

## ~~~~~~~~~~~~~~~ start the main training:
### store trajectory, energy vs t for all time segments:
Etot_NN, Ekin_NN, Epot_NN = [], [], []
### record time:
t_md = time.time()
### start training:
for step in range(MDsteps):
    print('\n=== MDstep {:6d}  {:8.3f} ==='.format(step, time.time()-t_md))
    Params, loss_epoch = my_f.run_training(Params, t_seq_NN, ifcs, Nreport, num_epochs[1], epoch_dir)
    print("Finished Training")
    print(f"Total time: {time.time()-t_md}")
    ## compute trajectory for all particles
    Qt = my_f.QT(Params, t_seq_NN)
    #ifcs = ?? #Update initial conditions for next MDStep 
    save_as = output_dir+f"NN_MD_trj_{step+1}.lammpstrj"
    my_f.write_trj(Qt[0][:,0:Np], Qt[0][:,Np:2*Np],Qt[0][:,2*Np:3*Np], save_as)
    ## compute total energy as a functin of time
    Epot = my_f.LJ3D_M(Qt[0])[0]
    Ekin = 0.5*(np.sum(Qt[1]**2, axis=1))
    Etot = Ekin+Epot    
    ## store energy info for every segment
    ## note: there is overlapping between time segments
    if step != 0:
        del Etot_NN[-1], Epot_NN[-1], Ekin_NN[-1]
    Etot_NN.extend(list(Etot))
    Epot_NN.extend(list(Epot))
    Ekin_NN.extend(list(Ekin))    

Etot_NN = onp.asarray([t_seq_NN, Etot_NN]).T
Epot_NN = onp.asarray([t_seq_NN, Epot_NN]).T
Ekin_NN = onp.asarray([t_seq_NN, Ekin_NN]).T

onp.save(output_dir+'Params.npy', onp.asarray(Params))
onp.save(output_dir+'loss_vs_epoch.npy', onp.asarray(loss_epoch))
onp.save(output_dir+'Etot_NN.npy', Etot_NN)
onp.save(output_dir+'Epot_NN.npy', Epot_NN)
onp.save(output_dir+'Ekin_NN.npy', Ekin_NN)

print("compute MSE to quantify if hyper-parameters are good:")
mse_E  = ((Etot_MD[:,1][::int(del_t/del_t_MD)] - Etot_NN[:,1])**2).mean()
mse_KE = ((Ekin_MD[:,1][::int(del_t/del_t_MD)] - Ekin_NN[:,1])**2).mean()
mse_PE = ((Epot_MD[:,1][::int(del_t/del_t_MD)] - Epot_NN[:,1])**2).mean()
print(f"MSE for Etot: {mse_E}\nMSE for Ekin: {mse_KE}\nMSE for Epot: {mse_PE}")
tmp = XTN.reshape((len(XTN), -1))[::int(del_t/del_t_MD)]
mse_Qt = ((Qt[0] - tmp)**2).mean()
tmp = PTN.reshape((len(PTN), -1))[::int(del_t/del_t_MD)]
mse_Pt = ((Qt[1] - tmp)**2).mean()
print(f"MSE for Qt: {mse_Qt}\nMSE for Pt: {mse_Pt}")


