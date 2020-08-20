#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from jax import jit
from jax import grad
from jax import random
from jax import lax
import time
import jax.numpy as np

import numpy as onp
onp.random.seed(16807)


def sigmoid(x):
    return 1./(1. + np.exp(-x))


def read_xyz_file(file_path):
    temp = open(file_path, "r").readlines()[2:]
    data = []
    for i, line in enumerate(temp):
        tmp = line.strip("\n").split()
        data.append([float(tmp[1]), float(tmp[2]), float(tmp[3]) ])
    data = onp.asarray(data)
    return data.T

def read_lammps_file(file_path):
    temp = open(file_path, "r").readlines()[9:]
    data = []
    for i, line in enumerate(temp):
        tmp = line.strip("\n").split()
        data.append([float(tmp[3]), float(tmp[4]), float(tmp[5]) ])
    data = onp.asarray(data)
    return data.T


def init_sys_params(input_n, input_d, input_sigma, Q0, P0,
                    input_del_t, input_Np, input_box_dim):
    global Np, n, d
    # Number of neurons, Dimension of system, Number of particles
    n, d, Np = input_n, input_d, input_Np
    # Simulation box:
    global box_dim, box_dim_half,  xlo, ylo, zlo, xhi, yhi, zhi
    box_dim = np.asarray(input_box_dim)
    box_dim_half = box_dim/2.0
    xlo, xhi = -box_dim[0]/2.0, box_dim[0]/2.0
    ylo, yhi = -box_dim[1]/2.0, box_dim[1]/2.0
    zlo, zhi = -box_dim[2]/2.0, box_dim[2]/2.0
    # Simulation parameters:
    global sigma, sigma12, sigma6, H0, dt
    dt = input_del_t
    sigma12 = 4.0*input_sigma**12
    sigma6 = 4.0*input_sigma**6
    H0 = 0.5*(np.sum(P0[0]**2 + P0[1]**2 + P0[2]**2)) + LJ3D_Force(Q0[0], Q0[1], Q0[2])[0]

    return

def init_nn_params(n, d, Np):
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

@jit
def H(Q, P):
    return LJ3D_Force(Q[0], Q[1], Q[2])[0] + 0.5*np.sum(P**2)

## 2D lennard-jones potential in matrix form
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
    return 0.5*energy, np.hstack((fx, fy, fz))


## initialize parameters for qx and qy
@jit
def loss_0(Params, t_seq, ifcs):
    Qt = QT(Params, t_seq)[0]
    ic  = np.sum((Qt[0]   - ifcs[0])**2)
    ic += np.sum((Qt[-1]  - ifcs[1])**2)
    return ic
dL0dQ = jit(grad(loss_0, 0))

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
    velocities = onp.zeros((Params.shape))
    lss = loss_0(Params, t_seq, ics)
    print(f'epoch: {0} loss: {lss}')
    for epoch in range(num_epochs):
        if (epoch+1) % 5000 == 0:
            lss = loss_0(Params, t_seq, ics)
            print(f'epoch: {epoch+1} loss: {lss}')
        Params, velocities, S = update_params_NADAM_0(t_seq, Params, velocities, S, ics,epoch)     
    return Params


#### Onsager-Machlup action and related:
@jit
def Action(A, F):
    return dt*np.mean((A-F)**2)


### the real loss function
@jit
def loss(Params, t_seq, ifcs, norm):
    ## Qt contains [x, y], [dxdt, dydt]
    Qt = QT(Params, t_seq)
    ## initial and final conditions:
    ic  = np.sum( (Qt[0][0][0:d*Np]  - ifcs[0])**2 )
    ic += np.sum( (Qt[0][-1][0:d*Np] - ifcs[1])**2 )
    ## calculate energies:
    KE = 0.5*np.sum(Qt[1]**2, axis=1)
    PE, F = LJ3D_M(Qt[0])
    return ic + Action(Qt[2], F)/norm + np.mean((PE+KE-H0)**2)/norm
## gradient of loss w.r.t Params
dLdQ = jit(grad(loss, 0))

### the FAKE loss function
@jit
def fake_loss(Params, t_seq, ifcs, norm):
    ## Qt contains [x, y], [dxdt, dydt]
    Qt = QT(Params, t_seq)
    ## initial and final conditions:
    ic  = np.sum( (Qt[0][0][0:d*Np]  - ifcs[0])**2 )
    ic += np.sum( (Qt[0][-1][0:d*Np] - ifcs[1])**2 )
    ## calculate energies:
    KE = 0.5*np.sum(Qt[1]**2, axis=1)
    PE, F = LJ3D_M(Qt[0])
    return ic, Action(Qt[2], F)/norm, np.mean((PE+KE-H0)**2)/norm

@jit
def update_params_NADAM(t_seq, Params, norm, V, S, ics, epoch,
                        alpha=0.001, eps = 10.0**-7, beta = np.asarray([0.999,0.999])):
    b0t, b1t = beta[0]**(epoch+1), beta[1]**(epoch+1)
    grads = dLdQ(Params, t_seq, ics, norm)
    V = beta[0]*V + (1.0-beta[0])*grads
    V_t = V/(1.0-b0t)
    S = beta[1]*S + (1.0-beta[1])*(grads**2)
    S_t = alpha/(np.sqrt(S/(1.0-b1t))+eps)
    Params -= S_t*(beta[0]*V_t+(1.0-beta[0])/(1.0-b0t)*grads)
    return Params, V, S



def run_training(Params, t_seq, ifcs, report_every, num_epochs, out_dir):
    t_md = time.time()
    loss_epoch = []
    ## define normalizer: norm
    Qt = QT(Params, t_seq)
    norm = np.max(abs(LJ3D_M(Qt[0])[0]))
    ## initial loss:
    lss = onp.asarray(fake_loss(Params, t_seq, ifcs, norm))
    loss_epoch.append([0] + list(lss) + [np.sum(lss)])
    print(f"Initial loss: {np.sum(lss)} = sum of {lss}")

    ## initialize V and S for optimizer:
    V, S = onp.zeros((Params.shape)), onp.zeros((Params.shape))
    for epoch in  range(num_epochs):
        Params, V, S = update_params_NADAM(t_seq, Params, norm, V, S, ifcs, epoch)
        if (epoch+1) % report_every == 0:
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
            norm = np.max(abs(LJ3D_M(Qt[0])[0]))
            print(f"Normalizer = {norm}")
    ## save to files:
    onp.save(out_dir+'Qt_NN.npy', Qt)
    Epot_epoch = np.array([t_seq, Epot]).T
    onp.save(out_dir+'Epot_NN.npy', Epot_epoch)
    Ekin_epoch = np.array([t_seq, Ekin]).T
    onp.save(out_dir+'Ekin_NN.npy', Ekin_epoch)
    Etot_epoch = np.array([t_seq, Etot]).T
    onp.save(out_dir+'Etot_NN.npy', Etot_epoch)
    print("Finished Training")
    return Params, onp.asarray(loss_epoch)


## write trajectory to files:
def write_Qt_to_file(Qt, file_path):
    with open(file_path, "w") as fout:
        for i in range(len(Qt)):
            q = Qt[i]
            tmp_q = onp.asarray([ q[0:Np], q[Np:2*Np], q[2*Np:] ]).T
            fout.write(f"ITEM: TIMESTEP\n{i}\n")
            fout.write(f"ITEM: NUMBER OF ATOMS\n{len(tmp_q)}\n")
            fout.write("ITEM: BOX BOUNDS pp pp pp\n")
            fout.write(f"{-5.0} {5.0}\n{-5.0} {5.0}\n{-5.0} {5.0}\n")
            fout.write("ITEM: ATOMS id type x y z\n")
            for i, qi in enumerate(tmp_q):
                fout.write(f"{i+1} {1} {qi[0]} {qi[1]} {qi[2]}\n")
    return 

## write trajectory to files:
def write_Qt_Pt_to_file(Qt, Pt, file_path):
    with open(file_path, "w") as fout:
        for i in range(len(Qt)):
            q, p = Qt[i], Pt[i]
            tmp_q = onp.asarray([ q[0:Np], q[Np:2*Np], q[2*Np:] ]).T
            tmp_p = onp.asarray([ p[0:Np], p[Np:2*Np], p[2*Np:] ]).T
            fout.write(f"ITEM: TIMESTEP\n{i}\n")
            fout.write(f"ITEM: NUMBER OF ATOMS\n{len(tmp_q)}\n")
            fout.write("ITEM: BOX BOUNDS pp pp pp\n")
            fout.write(f"{-5.0} {5.0}\n{-5.0} {5.0}\n{-5.0} {5.0}\n")
            fout.write("ITEM: ATOMS id type x y z vx vy vz\n")
            for i, (qi, pi) in enumerate(zip(tmp_q, tmp_p)):
                fout.write(f"{i+1} {1} {qi[0]} {qi[1]} {qi[2]} {pi[0]} {pi[1]} {pi[2]}\n")
    return 




##### ~~~~~~~~~~~~~~~~ MD-related code ~~~~~~~~~ 

def get_velocities(qx, qy, qz, px, py, pz, H0):
    ### remove center momentum:
    px, py, pz = px-np.mean(px), py-np.mean(py), pz-np.mean(pz)
    ### current temperature:
    T = np.sum(px**2 + py**2 + pz**2)/(3.0*Np)
    #print(f"current temperature = {T*125.7} K")
    ### correct temperature:
    Tcor = 2.0*(H0 - LJ3D_Force(qx, qy, qz)[0])/(3.0*Np)
    #print(f"Correct temperature = {Tcor*125.7} K")
    return px*onp.sqrt(Tcor/T), py*onp.sqrt(Tcor/T), pz*onp.sqrt(Tcor/T)
    
    

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
