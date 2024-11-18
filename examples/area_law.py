import numpy as np
from matplotlib import pyplot as plt
from drb.entropy import *

def area_law():
    L = 8
    m = 100
    start_idx = 0
    p = 1
    psi_0 = init_psi0(L)
    psi_final = evolve_psi(psi_0, p=p, m=m)
    S_vals = calculate_entropy(psi_final)
    plt.plot(np.arange(0, L + 1), S_vals[0, :][0:])
    plt.xlabel(rf'$x\in\left[{start_idx},L-1\right]$')
    plt.ylabel(rf'$S([{start_idx} x])$')
    plt.title(rf'Volume Law Scaling with Entanglement Entropy ($m$={m})')
    plt.savefig(f'../output/area_law_{m}.png')
    

if __name__ == '__main__':
    area_law()