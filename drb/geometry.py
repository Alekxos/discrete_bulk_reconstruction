import numpy as np
from matplotlib import pyplot as plt
from drb.entropy import *
from drb.reconstruct import *
import imageio

# We would also like to observe the evolution of weights geometrically,
# as we start off with an unentangled initial state with p=0, leading to
# progressive volume-law entanglement.

center = (0.5, 0.5)
radius = 0.5
layer_size = 0.05

def init_axis():
  plt.axis('off')
  plt.xlim([-.1, 1.1])
  plt.ylim([-.1, 1.1])
  axis = plt.gcf().gca()
  axis.figure.set_size_inches(5, 5)
  circle = plt.Circle(center, radius, color='black', fill=False)
  axis.add_patch(circle)
  return axis

def point_coordinates(index, layer, N):
  index %= N
  index -= (1 - layer % 2) /2
  x = center[0] + (radius - layer * layer_size) * np.cos(index * 2 * np.pi / N)
  y = center[1] + (radius - layer * layer_size) * np.sin(index * 2 * np.pi / N)
  return (x, y)

def convert_to_gif(file_prefix, max_t):
    filenames = [f'{file_prefix}_{t}.png' for t in range(max_t)]
    with imageio.get_writer(f'{file_prefix}_animation.gif', mode='I', fps=2) as writer:
        for filename in filenames:
            image = imageio.imread(filename)
            writer.append_data(image)

def plot_weight_evolution(psi, N, max_t, color, file_prefix):
  for t in range(max_t):
    axis = init_axis()
    psi_t = psi[t]
    S_vals = calculate_entropy(psi_t)
    diamondwork_weights = compute_diamondwork_weights(compute_bulk_weights(S_vals))
    diamondwork_weights[np.abs(diamondwork_weights) < 1e-14] = 0
    for index in range(N):
        for layer in range(N // 2):
            x1, y1 = point_coordinates(index, layer, N)
            x2, y2 = point_coordinates(index, layer + 1, N)
            x3, y3 = point_coordinates(index + 1, layer + (layer % 2), N)
            alpha_1 = diamondwork_weights[layer, 2 * index]
            alpha_2 = diamondwork_weights[layer, 2 * index + 1]
            axis.plot([x1, x2], [y1, y2], color=color, alpha=alpha_1)
            if layer % 2 == 0:
                axis.plot([x2, x3], [y2, y3], color=color, alpha=alpha_2)
            else:
                axis.plot([x1, x3], [y1, y3], color=color, alpha=alpha_2)
    plt.savefig(f'{file_prefix}_{t}.png')
    plt.clf()


    