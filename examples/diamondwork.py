import joblib
from matplotlib import pyplot as plt
import numpy as np

from drb.entropy import *
from drb.reconstruct import *

def main():
    S_vals_B1_0_to_1 = joblib.load('/Users/minerva/Lab/holographic_duality/DiscreteBulkReconstruction/data/S_vals_B1_0_to_1.pickle')

    p_vals = np.linspace(0, 1, 12) # This is fixed to correspond to saved files
    L_vals = [8, ] # Also fixed, like p_vals
    num_repeats = 100 # Also fixed

    diamond_dict_B1_L_8_p_0_to_1 = extract_diamond_dict(S_vals_B1_0_to_1, p_vals, L_vals, num_repeats)

    ## Plot for B1, L=8, p=0 to 1
    N = 7 # Number of inner layers
    for layer in range(N // 2):
        innermost_layer_average = np.zeros_like(p_vals)
        for p_idx, p in enumerate(p_vals):
            innermost_layer_average[p_idx] = np.mean(diamond_dict_B1_L_8_p_0_to_1[p][layer, :])
        plt.xlabel(r'Measurement Probability $p$')
        plt.ylabel(r'Average diamondwork weights (per layer)')
        plt.plot(p_vals, innermost_layer_average, 'o', label=f"k={layer}")
    plt.savefig(f'../output/diamondwork_B1_L_8_p_0_to_1.png')   

    ## Okay, now let's move from L=8 to L=16

    S_vals_B1_L16 = joblib.load('/Users/minerva/Lab/holographic_duality/DiscreteBulkReconstruction/data/S_vals_B1_L16.pickle')
    L = 16
    N = L - 1
    diamond_dict_S_vals_B1_L16 = extract_diamond_dict(S_vals_B1_L16, p_vals, [L, ], num_repeats)

    ## Plot for B1, L=16, p=0 to 1
    L = 16
    N = L - 1
    for layer in range(N // 2):
        innermost_layer_average = np.zeros_like(p_vals)
        for p_idx, p in enumerate(p_vals):
            innermost_layer_average[p_idx] = np.mean(diamond_dict_S_vals_B1_L16[(p, L)][layer, :])
        plt.xlabel(r'Measurement Probability $p$')
        plt.ylabel(r'Average diamondwork weights (per layer)')
        plt.plot(p_vals, innermost_layer_average, 'o', label=f"k={layer}")
    plt.legend()
    plt.savefig(f'../output/diamondwork_B1_L_16_p_0_to_1.png')

    # Okay, we have a clear distinction between interior weights and the boundary
    # weights, but it's hard to clearly distinguish the phase transition point
    # which occurs for p*~0.15.

    # The B2 system should have p*~0.68, so let's try that instead. We'll start
    # again with L=8.

    S_vals_B2_L8 = joblib.load('/Users/minerva/Lab/holographic_duality/DiscreteBulkReconstruction/data/S_vals_B2_L8.pickle')
    p_vals = [0.3, 0.4, 0.5, 0.56, 0.59, 0.62, 0.65, 0.68, 0.71, 0.74, 0.8, 0.9, 1.0]
    L = 8
    N = L - 1
    diamond_dict_S_vals_B2_L8 = extract_diamond_dict(S_vals_B2_L8, p_vals, [L, ], num_repeats)

    # Here's that plot. It looks like the fact that the B2 scheme projects onto
    # a rank-2 subspace means that as p->1, there is nonzero entanglement and 
    # we do not observe the interior layers fully converging to zero weights.

    L = 8
    N = L - 1
    for layer in range(N // 2):
        innermost_layer_average = np.zeros_like(p_vals)
        for p_idx, p in enumerate(p_vals):
            innermost_layer_average[p_idx] = np.mean(diamond_dict_S_vals_B2_L8[(p, L)][layer, :])
        plt.xlabel(r'Measurement Probability $p$')
        plt.ylabel(r'Average diamondwork weights (per layer)')
        plt.plot(p_vals, innermost_layer_average, 'o', label=f"k={layer}")
    plt.legend()
    plt.savefig(f'../output/diamondwork_B2_L_8.png')


if __name__ == '__main__':
    main()