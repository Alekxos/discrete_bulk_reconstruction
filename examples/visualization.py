from drb.geometry import *
from drb.entropy import *

# We would also like to observe the evolution of weights geometrically,
# as we start off with an unentangled initial state with p=0, leading to
# progressive volume-law entanglement.

def main():
    center = (0.5, 0.5)
    radius = 0.5
    layer_size = 0.05

    N = 8
    color = (0.1, 0.2, 0.5)
    # color = (0.7, 0.1, 0.3)
    alpha = 1.0

    # Prepare initial state and evolve, recording all states in time
    max_t = 20
    psi_0 = init_psi0(N)
    # psi = evolve_psi(psi_0, p=0.18, m=max_t, all_states = True)
    psi = evolve_psi(psi_0, p=0.01, m=max_t, all_states = True)

    save_prefix = '../output/visualize/volume_law'
    plot_weight_evolution(psi, N, max_t, color, file_prefix=save_prefix)
    convert_to_gif(file_prefix=save_prefix, max_t=max_t)

if __name__ == '__main__':
    main()