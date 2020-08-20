#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from jax import jit
from jax import grad
from jax import random
from jax import lax
import time
from itertools import islice
import jax.numpy as np

import numpy as onp
onp.random.seed(16807)

def sigmoid(x):
    return 1./(1. + np.exp(-x))

## Initial conditions from lammps file:
## Format: line 1 = Item Atoms id mol type x y z vx vy vz, 
##         line 2-Np+1: values
def load_Pos(Np, filename, start):
    iQx0 = []
    iQy0 = []
    iQz0 = []
    iPx0 = []
    iPy0 = []
    iPz0 = [] 
    with open(filename,'r') as fin:
        for line in islice(fin, start, Np+start):
            iQx0.append(float(line.split(" ")[3]))
            iQy0.append(float(line.split(" ")[4]))
            iQz0.append(float(line.split(" ")[5]))
            iPx0.append(float(line.split(" ")[6]))
            iPy0.append(float(line.split(" ")[7]))
            iPz0.append(float(line.split(" ")[8]))
    fin.close()
    qx0 = np.asarray(iQx0)
    qy0 = np.asarray(iQy0)
    qz0 = np.asarray(iQz0)
    px0 = np.asarray(iPx0)
    py0 = np.asarray(iPy0)
    pz0 = np.asarray(iPz0)
    Q0 = np.asarray([qx0, qy0, qz0])
    P0 = np.asarray([px0, py0, pz0])
    return (Q0, P0)

def init_sys_params(n_i, d_i, sigma_i, del_t_i, Np_i, box_dim_i, filename, start):
    global Np, n, d # Neural network parameters/precalculated terms
    global box_dim, box_dim_half,  xlo, ylo, xhi, yhi, zlo, zhi #Simulation box parameters
    global px0, py0, qx0, qy0, H0, dt, Q0, P0        # Initial conditions for system
    global sigma, sigma12, sigma6 #LJ potential  
    sigma = sigma_i
    sigma12 = 4.0*sigma**12
    sigma6 = 4.0*sigma**6
    dt = del_t_i
    n  = n_i # Number of neurons
    d  = d_i # Dimension of system
    Np = Np_i # Number of particles
    Q0,P0 = load_Pos(Np,filename, start) # Load initial conditions from file
    
    box_dim = np.asarray(box_dim_i)
    box_dim_half = box_dim/2.0
    xlo, xhi = -box_dim[0]/2.0, box_dim[0]/2.0
    ylo, yhi = -box_dim[1]/2.0, box_dim[1]/2.0
    zlo, zhi = -box_dim[2]/2.0, box_dim[2]/2.0
    H0 = H(Q0, P0)
    return

def init_nn_params(n,Np):
    key = random.PRNGKey(0)
    Params = 0.01*random.uniform(key, (2*n+(n*d*Np)+d*Np,), minval=-1.0, maxval=1.0)
    return Params
@jit
def QT(Params, t_seq):
    ## output of the network is qx, qy for each particle
    w0 = Params[:n]
    b0 = Params[n:2*n]
    w1 = Params[2*n:2*n+(n*d*Np)].reshape((d*Np,n))
    b1 = Params[2*n+(n*d*Np):2*n+(n*d*Np)+d*Np]
    tmp = np.outer(w0, t_seq) + b0[:,None]
    tmp1 = sigmoid(tmp)
    q = np.dot(w1,tmp1) + b1[:,None]
    tmp2 = (tmp1 - tmp1**2)*w0[:,None]
    dq = np.dot(w1, tmp2)
    tmp3 = tmp2*(1.0-2.0*tmp1)*w0[:,None]
    ddq = np.dot(w1, tmp3)
    return q.T, dq.T, ddq.T

## minimum image convention + pbcs 
def MIC(qs, Lq, Lqh):
    qs -= (qs//Lq)*Lq
    qs -= Lq*(qs//Lqh)
    return qs
 
## Functions to calculate initial energy
## energy for certain t only (scalar)
@jit
def LJ3D_Force(qx, qy, qz):
    energy = 0.0
    xs, ys, zs = qx, qy, qz
    fx, fy, fz = 0.0, 0.0, 0.0
    x2s = np.asarray([np.roll(xs,i)-xs for i in range(1,Np)])
    y2s = np.asarray([np.roll(ys,i)-ys for i in range(1,Np)])
    z2s = np.asarray([np.roll(zs,i)-zs for i in range(1,Np)])
    diff_x = MIC(x2s, box_dim[0], box_dim_half[0])
    diff_y = MIC(y2s, box_dim[1], box_dim_half[1])
    diff_z = MIC(z2s, box_dim[2], box_dim_half[2])
    r2 = diff_x**2 + diff_y**2 + diff_z**2
    tmp =  -12*sigma12/(r2**7) + 6*sigma6/(r2**4)
    fx = np.sum(tmp*diff_x, axis=0)
    fy = np.sum(tmp*diff_y, axis=0)
    fz = np.sum(tmp*diff_z, axis=0)
    energy = np.sum(sigma12/r2**6 - sigma6/r2**3)
    return 0.5*energy, fx, fy, fz

def H(Q0, P0):
    PE = LJ3D_Force(Q0[0], Q0[1], Q0[2])[0]
    return PE + 0.5*np.sum(P0**2)

## lennard-jones potential in matrix form
@jit
def LJ3D_M(Qt):
    energy = 0.0
    xs, ys, zs = Qt[:,0:Np], Qt[:,Np:2*Np], Qt[:,2*Np:]
    x2s = np.asarray([np.roll(xs, i, axis = 1)-xs for i in range(1,Np)])
    y2s = np.asarray([np.roll(ys, i, axis = 1)-ys for i in range(1,Np)])
    z2s = np.asarray([np.roll(zs, i, axis = 1)-zs for i in range(1,Np)])
    diff_x = MIC(x2s, box_dim[0], box_dim_half[0])
    diff_y = MIC(y2s, box_dim[1], box_dim_half[1])
    diff_z = MIC(z2s, box_dim[2], box_dim_half[2])
    r2 = diff_x**2 + diff_y**2 + diff_z**2
    ## compute energy:
    energy = np.sum(sigma12/r2**6 - sigma6/r2**3, axis=(0,2))
    ## compute forces:
    tmp = -12.0*sigma12/(r2**7) + 6.0*sigma6/(r2**4)
    fx = np.sum(tmp*diff_x, axis=0)
    fy = np.sum(tmp*diff_y, axis=0)
    fz = np.sum(tmp*diff_z, axis=0)
    return 0.5*energy, fx, fy, fz

@jit
def loss_0(Params, t_seq, ifcs):
    Qt = QT(Params, t_seq)
    l = len(t_seq)//2
    ic  = np.sum((Qt[0][0]  - ifcs[0])**2)
    ic += np.sum((Qt[0][-1] - ifcs[1])**2)
    ic += np.sum((Qt[1][0:l-2] - ifcs[2])**2)
    ic += np.sum((Qt[1][l+2:]  - ifcs[3])**2)
    return ic
dL0dQ = jit(grad(loss_0, 0))

#### Onsager-Machlup action and related:
@jit

def Action(ax, ay, az, fx, fy, fz):
    return dt*np.mean((ax-fx)**2 + (ay-fy)**2 + (az-fz)**2)

### the real loss function
@jit
def loss(Params, t_seq, ifcs, norm):
    ## Qt contains [x, y], [dxdt, dydt]
    Qt = QT(Params, t_seq)
    ## ifcs contains initial and final state:
    ic  = 20.0*np.mean((Qt[0][0]  - ifcs[0])**2)
    ic += 20.0*np.mean((Qt[0][-1] - ifcs[1])**2)
    ic += 20.0*np.mean((Qt[1][0]  - ifcs[2])**2)
    ic += 20.0*np.mean((Qt[1][-1] - ifcs[3])**2)
    ## conservation of momentum:
    #cm  = (np.sum(Qt[1][:, 0:Np], axis=1))**2 \
    #    + (np.sum(Qt[1][:, Np:2*Np], axis=1))**2 \
    #    + (np.sum(Qt[1][:, 2*Np:], axis=1))**2
    ## calculate energies and gradient of action:
    KE = 0.5*np.sum(Qt[1]**2, axis=1)
    PE, fx, fy, fz = LJ3D_M(Qt[0])
    E_diff = ((PE+KE-H0)/norm)**2
    A = Action(Qt[2][:,0:Np], Qt[2][:,Np:2*Np], Qt[2][:,2*Np:], fx, fy, fz)
    return ic + A/norm + np.mean(E_diff)# + np.mean(cm)
## gradient of loss w.r.t Params
dLdQ = jit(grad(loss, 0))

## define another loss function, just to return 3 components of the loss
@jit
def fake_loss(Params, t_seq, ifcs, norm):
    ## Qt contains [x, y], [dxdt, dydt]
    Qt = QT(Params, t_seq)
    ## ifcs contains initial and final state:
    ic  = 20.0*np.mean((Qt[0][0]  - ifcs[0])**2)
    ic += 20.0*np.mean((Qt[0][-1] - ifcs[1])**2)
    ic += 20.0*np.mean((Qt[1][0]  - ifcs[2])**2)
    ic += 20.0*np.mean((Qt[1][-1] - ifcs[3])**2)
    ## conservation of momentum:
    #cm  = (np.sum(Qt[1][:, 0:Np], axis=1))**2 \
    #    + (np.sum(Qt[1][:, Np:2*Np], axis=1))**2 \
    #    + (np.sum(Qt[1][:, 2*Np:], axis=1))**2
    ## calculate energies and gradient of action:
    KE = 0.5*np.sum(Qt[1]**2, axis=1)
    PE, fx, fy, fz = LJ3D_M(Qt[0])
    E_diff = ((PE+KE-H0)/norm)**2
    A = Action(Qt[2][:,0:Np], Qt[2][:,Np:2*Np], Qt[2][:,2*Np:], fx, fy, fz)
    return ic, A/norm, np.mean(E_diff)#, np.mean(cm)

## update all the parameters using NADAM
@jit
def update_params_NADAM_0(t_seq, Params, V, S, ics, epoch,
                        alpha=0.001, eps = 10.0**-7, beta = np.asarray([0.999,0.999])):
    b0t, b1t = beta[0]**(epoch+1), beta[1]**(epoch+1)
    grads = dL0dQ(Params, t_seq, ics)
    V = beta[0]*V + (1.0-beta[0])*grads
    V_t = V/(1.0-b0t)
    S = beta[1]*S + (1.0-beta[1])*(grads**2)
    S_t = alpha/(np.sqrt(S/(1.0-b1t))+eps)
    Params -= S_t*(beta[0]*V_t+(1.0-beta[0])/(1.0-b0t)*grads)
    return Params, V, S

#Initial training function 
def train_N0(Params, t_seq, num_epochs, ics):
    S = onp.zeros((Params.shape))
    V = onp.zeros((Params.shape))
    for epoch in range(num_epochs):
        if (epoch+1) % 5000 == 0:
            lss = loss_0(Params, t_seq, ics)
            print(f'epoch: {epoch+1} loss: {lss}')
        Params, V, S = update_params_NADAM_0(t_seq, Params, V, S, ics,epoch)     
    return Params

@jit
def update_params_NADAM(t_seq, Params, V, S, ics, norm, epoch,
                        alpha=0.001, eps = 10.0**-7, beta = np.asarray([0.999,0.999])):
    b0t, b1t = beta[0]**(epoch+1), beta[1]**(epoch+1)
    grads = dLdQ(Params, t_seq, ics, norm)
    V = beta[0]*V + (1.0-beta[0])*grads
    V_t = V/(1.0-b0t)
    S = beta[1]*S + (1.0-beta[1])*(grads**2)
    S_t = alpha/(np.sqrt(S/(1.0-b1t))+eps)
    Params -= S_t*(beta[0]*V_t+(1.0-beta[0])/(1.0-b0t)*grads)
    return Params, V, S

def run_training(Params, t_seq, ifcs, report_every, num_epochs, epoch_dir):
    t_md = time.time()
    loss_epoch = []
    ## compute first normalizer:
    Qt = QT(Params, t_seq)
    Epot = LJ3D_M(Qt[0])[0]
    max_PE = np.max(Epot)
    norm = np.max(np.abs(Epot))
    print(f"Initial normalizer: {onp.asarray(norm)}")
    print(f"max Epot: {max_PE}")
    ## compute first loss:
    lss = onp.asarray(fake_loss(Params, t_seq, ifcs, norm))
    loss_epoch.append([0] + list(lss) + [np.sum(lss)])
    print(f"Initial loss: {np.sum(lss)} = sum of {lss}")
    ## initialize V and S for optimizer:
    V, S = onp.zeros((Params.shape)), onp.zeros((Params.shape)) 
    for epoch in  range(num_epochs):
        Params, V, S = update_params_NADAM(t_seq, Params, V, S, ifcs, norm, epoch)
        if (epoch+1) % report_every == 0:
            ## print value of normalizer:
            print(f"normalizer: {onp.asarray(norm)}")
            print(f"max Epot: {max_PE}")
            lss = onp.asarray(fake_loss(Params, t_seq, ifcs, norm))
            loss_epoch.append([epoch+1] + list(lss) + [np.sum(lss)])
            print(f"Loss at epoch {epoch+1}: {np.sum(lss)} = sum of {lss}")
            print(f"Time:{float(-t_md + time.time())}")
            t_md = time.time()
            ## compute energy as a function of epoch:
            Qt = QT(Params, t_seq)
            Epot = LJ3D_M(Qt[0])[0]
            Ekin = 0.5*(np.sum(Qt[1]**2, axis=1))
            Etot = Ekin+Epot
            ## update norm:
            norm = np.max(np.abs(Epot))
            max_PE = np.max(Epot)
            ## save to files:
            onp.save(epoch_dir+'Qt_'+str(epoch+1)+'.npy', Qt)
            Epot_epoch = np.array([t_seq, Epot]).T
            onp.save(epoch_dir+'Epot_'+str(epoch+1)+'.npy', Epot_epoch)
            Ekin_epoch = np.array([t_seq, Ekin]).T
            onp.save(epoch_dir+'Ekin_'+str(epoch+1)+'.npy', Ekin_epoch)
            Etot_epoch = np.array([t_seq, Etot]).T
            onp.save(epoch_dir+'Etot_'+str(epoch+1)+'.npy', Etot_epoch)
    print("Finished Training")
    print(f"Total time: {time.time()-t_md}")
    return Params, onp.asarray(loss_epoch)

    
## write trajectory to files:
def write_trj(Xt, Yt, Zt, save_to, write_out=True):
    if not write_out:
        return
    with open(save_to,"w") as f:
        for t in range(len(Xt)):
            f.write(f"ITEM: TIMESTEP\n{t}\n")
            f.write(f"ITEM: NUMBER OF ATOMS\n{Np}\n")
            f.write(f"ITEM: BOX BOUNDS pp pp pp\n {xlo} {xhi} \n {ylo} {yhi} \n {zlo} {zhi}\n")
            f.write("ITEM: ATOMS id type x y z\n")
            ID = 0
            for x, y, z in zip(Xt[t], Yt[t], Zt[t]):
                ID += 1
                f.write(f"{ID} {1} {x} {y} {z}\n")
    return


##### ~~~~~~~~~~~~~~~~ MD part ~~~~~~~~~    
def velocity_verlet(r, v, a_old, dt):
    ## r, v, a are matrices: (dimension, Num. of particle)
    r += v*dt + a_old*dt*dt/2.0
    PE,ax,ay,az = LJ3D_Force(r[0], r[1], r[2])
    a = np.asarray([ax, ay, az])
    v += (a+a_old)*dt/2

    return r, v, a,PE

def calc_KE(v):
    return 0.5*onp.sum(v**2)

def run_MD(r0, v0, steps, dt):
    r, v = r0, v0
    ## compute 'old' acceleration
    PE, ax, ay, az = LJ3D_Force(r[0], r[1], r[2])
    a_old = np.asarray([ax, ay, az])
    ## compute energy
    KE  =  calc_KE(v)
    energy = KE + PE
    ## store trajectory and energy info:
    XTN = [onp.array(r)]
    PTN = [onp.array(v)]
    ATN = [onp.array(a_old)]
    E_vs_t, KE_vs_t, PE_vs_t = [energy], [KE], [PE]
    for i in range(steps):
        r, v, a_old,PE = velocity_verlet(r, v, a_old, dt)
        ## store trajectory:
        XTN.append(onp.array(r))
        PTN.append(onp.array(v))
        ATN.append(onp.array(a_old))
        ## compute and store energy:
        KE = calc_KE(v)
        E_vs_t.append(KE+PE)
        KE_vs_t.append(KE)
        PE_vs_t.append(PE)
        #if i % 100 == 0:
        #    print(f"MD Energy: {energy}")
    return onp.asarray(XTN), onp.asarray(PTN), onp.asarray(ATN), \
           onp.asarray(E_vs_t), onp.asarray(KE_vs_t), onp.asarray(PE_vs_t)
