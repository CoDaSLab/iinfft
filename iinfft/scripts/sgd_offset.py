import os
import numpy as np
import matplotlib.pyplot as plt
import multiprocessing

from iinfft.iinfft import *
#from batch_job import test_infft
import pandas as pd

plt.style.use('tableau-colorblind10')

def test_infft(dat, idx, Ln, N, w):
    stp = np.random.randint(0, Ln)
    t = np.linspace(-0.5, 0.5, Ln, endpoint=False)
    Mn = int(Ln / (t[stp] + 0.5) / (1 - 1 / Ln))
    t_temp = np.linspace(-0.5, 0.5, Mn, endpoint=False)
    h_k = -(N // 2 ) + np.arange(N)
    idxi = np.pad(idx, (0, len(t_temp) - len(t)), constant_values=0)
    AhA1 = compute_sym_matrix_optimized(t_temp[idxi], h_k)
    _, _, _, err = infft(t_temp[idxi], dat[idx] - np.mean(dat[idx]), N=N, AhA=AhA1, w=w, return_adjoint=True)
    return (err, stp)

if __name__ == '__main__':

    df = pd.read_csv('iinfft/data/T.Suelo.csv')
    Ln = df.shape[0]
    smplR = 1800
    data_raw = df.iloc[0:,1:].to_numpy() #keep the missing values
    inverse_mat = np.zeros_like(data_raw,dtype="complex128")
    residue_mat = np.zeros_like(data_raw,dtype="float64")
    rec_mat = np.zeros_like(data_raw,dtype="float64")
    mni = np.zeros((df.shape[1]-1,1))

    N = 1024
    t = np.linspace(-0.5,0.5,Ln,endpoint=False)
    inverse_mat = np.zeros((N,df.shape[1]-1),dtype="complex128")
    #w = fjr(N)
    w = sobk(N,1,2,1e-2)

    dat = data_raw[:,0]

    idx = dat != -9999
    if sum(idx) % 2 != 0:
        idx = change_last_true_to_false(idx)

    h_k = -(N // 2 ) + np.arange(N)
    AhA1 = compute_sym_matrix_optimized(t[idx],h_k)
    ftot, _, _, _ = infft(t[idx], dat[idx] - np.mean(dat[idx]),N=N,AhA=AhA1,w=w)
    ytot_mean = adjoint(t,ftot) + np.mean(dat[idx])

    num_cores = 6 #multiprocessing.cpu_count() - 1

    # 10 iterations, 5 batches each, eta = 0.25
    q = np.zeros(5)
    err_best = np.inf
    stp_best = 0
    stp_iter = Ln
    eta = 0.25
    itrs = []
    errs = []
    epochs = 7
    results = [[None, None] for _ in range(num_cores)]

    for ii in range(epochs):
        for jj in range(num_cores):
            results[jj] = test_infft(dat,idx,Ln,N,w)
            args = [(dat,idx,Ln,N,w) for _ in range(num_cores)]
            #results = pool.starmap(test_infft,args)

        for jj in range(num_cores):
            if results[jj][0] < err_best:
                err_best = results[jj][0]
                stp_best = results[jj][1]
                stp_iter = int(stp_iter - eta * (stp_iter - stp_best))
        
        itrs.append(stp_iter)
        errs.append(err_best)

        print(f"epoch {ii} complete, error {errs[-1]}, stp_iter {stp_iter}")

    print("")
    #Plotting the results
    t = np.linspace(-0.5,0.5,Ln,endpoint=False)
    Mn = int(Ln/(t[stp_iter]+0.5)/(1-1/Ln))
    tn = np.linspace(-0.5,0.5,Mn,endpoint=False)
    idxi = np.pad(idx,(0,len(tn)-len(t)),constant_values=0)

    AhA1 = compute_sym_matrix_optimized(tn[idxi],h_k)

    fshift, _, _, _ = infft(tn[idxi], dat[idx] - np.mean(dat[idx]),N=N,AhA=AhA1,w=w)
    ytot_shift = adjoint(tn,fshift) + np.mean(dat[idx])

    # Creating subplots
    fig, axs = plt.subplots(2, 1, figsize=(8, 6))

    # Plotting scatter and line on the unshifted data
    axs[0].scatter(t[idx], dat[idx], s=0.1,c='k',label='measurements')
    axs[0].plot(t, ytot_mean, color='C1', label='unshifted')
    axs[0].set_title('Naive time labels')
    axs[0].set_xlabel("Normalized time values $t \in [-0.5,0.5)$")
    axs[0].set_ylabel("Temperature ($^\circ$C)")
    axs[0].legend()

    # Plotting scatter and line on the second subplot
    axs[1].scatter(tn[idxi], dat[idx], s=0.1,c='k',label='measurements')
    axs[1].plot(tn, ytot_shift, color='C2', label='shifted')
    axs[1].set_title('SGD shifted time labels')
    axs[1].set_xlabel("Normalized time values $t \in [-0.5,0.5)$")
    axs[1].set_ylabel("Temperature ($^\circ$C)")
    axs[1].legend()

    # Adjusting layout and displaying the plots
    plt.tight_layout()
    plt.show()
    plt.savefig("sgd_comparison.png")



