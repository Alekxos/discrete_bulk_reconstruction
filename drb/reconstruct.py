import numpy as np

alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

def compute_bulk_weights(S_vals):
  L = S_vals.shape[0] - 1
  w_vals = np.zeros((L, L))
  for delta_w in np.arange(1, L - 1):
    for i in np.arange(L - delta_w - 1):
      w_vals[i, i + delta_w] = w_vals[i + delta_w, i] = (1 / 2) * (S_vals[i, i + delta_w] + S_vals[i + delta_w, i + delta_w + 1] - \
                                                                   S_vals[i, i + delta_w + 1]) - np.sum(w_vals[i+1:i+delta_w,i+delta_w])
      # print("w_{"+f"{alphabet[i]},{alphabet[i+delta_w]}"+"}"+f"={w_vals[i, i + delta_w]}")
  return w_vals

def compute_diamondwork_weights(bulk_weights, verbose=False):
  N = bulk_weights.shape[0]
  diamondwork_weights = np.zeros((N // 2, 2 * N))
  for layer in range(N // 2):
    for region in range(N):
      for layer_idx in range(layer + 1):
        weight_start_idx, weight_end_idx = region % N, (region - (layer + 1)) % N
        if verbose:
            print(f"A1: {diamondwork_weights[layer_idx][region]}, B1: {diamondwork_weights[layer_idx + 1][(region - 1) % N]}")
        diamondwork_weights[layer_idx, 2 * region] += bulk_weights[weight_start_idx, weight_end_idx]
        if verbose:
            print("w_{"+f"{alphabet[weight_start_idx]},{alphabet[weight_end_idx]}"+"}"+f"={bulk_weights[weight_start_idx, weight_end_idx]}")
        weight_start_idx, weight_end_idx = region, (region + (layer + 1)) % N
        if verbose:
            print(f"A2: {diamondwork_weights[layer_idx][region]}, B2: {diamondwork_weights[layer_idx + 1][region]}")
        diamondwork_weights[layer_idx, 2 * region + 1] += bulk_weights[weight_start_idx, weight_end_idx]
        if verbose:
            print("w_{"+f"{alphabet[weight_start_idx]},{alphabet[weight_end_idx]}"+"}"+f"={bulk_weights[weight_start_idx, weight_end_idx]}")
  return diamondwork_weights

def extract_diamond_dict(S_vals, p_vals, L_vals, num_repeats):
    diamond_dict = {}
    for L in L_vals:
        N = L - 1
        for p in p_vals:
            diamondwork_weights_average = np.zeros((N // 2, 2 * N))
            for repeat_idx in range(num_repeats):
                S_matrix = S_vals[(p, L, repeat_idx)]
                diamondwork_weights_average += compute_diamondwork_weights(compute_bulk_weights(S_matrix))
            diamondwork_weights_average /= num_repeats
            diamond_dict[(p, L)] = diamondwork_weights_average
    return diamond_dict