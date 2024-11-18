import numpy as np
import scipy.sparse as sparse

'''Generate a Haar random unitary with dimension D'''
def gen_randU(D):
  X = (np.random.randn(D,D) + 1j*np.random.randn(D,D))/np.sqrt(2);
  [Q,R] = np.linalg.qr(X);
  R = np.diag(np.diag(R)/abs(np.diag(R)));
  return np.dot(Q,R)

'''Compute the entanglement entropy of state psi between contiguous boundary
regions demarcated by [i ,j]'''
def EntanglementEntropy(psi, i, j):
  L = int(np.log2(psi.shape[0]))
  Psi = np.reshape(psi, (int(2**i), int(2**(j - i)), int(2**(L - j))))
  Psi = np.swapaxes(Psi, 0, 1)
  Psi = np.reshape(Psi, (int(2**(j - i)), int(2**(L - j + i))))
  S = np.linalg.svd(Psi, compute_uv = False)
  S = S[S > 0]
  return -np.sum(S**2 * (2 * np.log2(S)))

def gen_projectors(site, L, rank):
  assert site < L - 1, "Site chosen must leave at least one additional qubit wire to the right."
  up_state = sparse.csr_matrix([[1.], [0.]])
  down_state = sparse.csr_matrix([[0.], [1.]])
  ### Define projective states on site and site+1
  state_00 = sparse.kron(up_state, up_state)
  state_01 = sparse.kron(up_state, down_state)
  state_10 = sparse.kron(down_state, up_state)
  state_11 = sparse.kron(down_state, down_state)

  proj_00 = sparse.kron(state_00.conj().T, state_00)
  proj_01 = sparse.kron(state_01.conj().T, state_01)
  proj_10 = sparse.kron(state_10.conj().T, state_10)
  proj_11 = sparse.kron(state_11.conj().T, state_11)

  start_idx = 0
  if rank == '1':
    if site == 0:
      proj_00_full = proj_00
      proj_01_full = proj_01
      proj_10_full = proj_10
      proj_11_full = proj_11
      start_idx += 2
    else:
      proj_00_full = sparse.eye(2)
      proj_01_full = sparse.eye(2)
      proj_10_full = sparse.eye(2)
      proj_11_full = sparse.eye(2)
      start_idx += 1
    L_idx = start_idx
    while L_idx < L:
      if L_idx == site:
        proj_00_full = sparse.kron(proj_00_full, proj_00)
        proj_01_full = sparse.kron(proj_01_full, proj_01)
        proj_10_full = sparse.kron(proj_10_full, proj_10)
        proj_11_full = sparse.kron(proj_11_full, proj_11)
        L_idx += 2
      else:
        proj_00_full = sparse.kron(proj_00_full, sparse.eye(2))
        proj_01_full = sparse.kron(proj_01_full, sparse.eye(2))
        proj_10_full = sparse.kron(proj_10_full, sparse.eye(2))
        proj_11_full = sparse.kron(proj_11_full, sparse.eye(2))
        L_idx += 1
    return proj_00_full, proj_01_full, proj_10_full, proj_11_full
  elif rank == '2':
    proj_A = proj_00 + proj_11
    proj_B = proj_01 + proj_10
    if site == 0:
      proj_A_full = proj_A
      proj_B_full = proj_B
      start_idx += 2
    else:
      proj_A_full = sparse.eye(2)
      proj_B_full = sparse.eye(2)
      start_idx += 1
    L_idx = start_idx
    while L_idx < L:
      if L_idx == site:
        proj_A_full = sparse.kron(proj_A_full, proj_A)
        proj_B_full = sparse.kron(proj_B_full, proj_B)
        L_idx += 2
      else:
        proj_A_full = sparse.kron(proj_A_full, sparse.eye(2))
        proj_B_full = sparse.kron(proj_B_full, sparse.eye(2))
        L_idx += 1
    return proj_A_full, proj_B_full
  else:
    raise Exception("Projection set must be rank-1 or rank-2.")
  
def init_psi0(L):
  psi0_dense = np.zeros((int(2**L), 1))
  psi0_dense[0, 0] = 1
  psi0 = sparse.csr_matrix(psi0_dense)
  return psi0

def print_entropies(S_vals, L):
  alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
  x = 0
  for i in range(L):
      for j in range(i + 1, L):
          x += 1
          print(x)
          print(f"S({i},{j})=S({alphabet[i]}, {alphabet[j - 1]}): {S_vals[i, j]}")

'''Apply a given 2x2 unitary to a given state at a given site'''
def ApplyGateToState(U2qubit, state, site, periodic=False):
  L = int(np.log2(state.shape[0]))
  assert(site + 2 <= L - 1,
         f"Gate site {site} doesn't have room for two additional qubits in system of size {L}.")
  U2_sparse = sparse.csr_matrix(U2qubit)
  start_idx = 0  

  ## Method 1
  if site == 0:
    # U_full = sparse.kron(np.eye(2), np.eye(2), 'csr')
    U_full = U2_sparse
    start_idx = 2
  else:
    U_full = sparse.eye(2, format = 'csr', dtype = 'complex')
    start_idx = 1

  idx = start_idx
  while idx < L:
    if idx == site:
      U_full = sparse.kron(U_full, U2_sparse)
      idx += 2
    else:
      U_full = sparse.kron(U_full, np.eye(2), 'csr')
      idx += 1
  return U_full * state

'''Accepts sparse csr_row_matrix (state), site number (to project sites
`site` and `site+1,` and rank of projection operators.'''
def project_state(state, proj_operators):
  L = int(np.log2(state.shape[0]))
  ## Assumes two sites
  ### Form full projector operators
  ### Determine projection probabilities
  prob_values = [state.conj().T * proj_op * state
                  for proj_op in proj_operators]
  assert(not np.abs(np.sum([prob_value.sum() for prob_value in prob_values]) - 1) >= 1e-12)
  sampled_prob = np.random.uniform()
  if prob_values[0] >= sampled_prob:
    projected_state = proj_operators[0] * state
  elif prob_values[0] + prob_values[1] >= sampled_prob:
    projected_state = proj_operators[1] * state
  elif prob_values[0] + prob_values[1] + prob_values[2] >= sampled_prob:
    projected_state = proj_operators[2] * state
  else:
    projected_state = proj_operators[3] * state
  return projected_state / sparse.linalg.norm(projected_state)

## First, let's make our evolution and entropy calculation into functions:
def evolve_psi(psi_0, p, m=100, all_states = False):
  L = int(np.log2(psi_0.shape[0]))
  psi_t = psi_0
  if all_states:
    psi = [psi_0, ]
  proj_operators_by_location = {site: gen_projectors(site, L, rank = '1') for site in range(L - 1)}
  for m_idx in range(m):
    for odd_idx in range(1, L - 1, 2):
      if np.random.uniform() < p:
        psi_t = project_state(psi_t, proj_operators_by_location[odd_idx])
      U_idx = gen_randU(4)
      psi_t = ApplyGateToState(U_idx, psi_t, site = odd_idx)

    for even_idx in range(0, L - 1, 2):
      if np.random.uniform() < p:
        psi_t = project_state(psi_t, proj_operators_by_location[even_idx])
      U_idx = gen_randU(4)
      psi_t = ApplyGateToState(U_idx, psi_t, site = even_idx)
    if all_states:
      psi.append(psi_t)
  if all_states:
    return psi
  return psi_t

def calculate_entropy(psi_t):
  L = int(np.log2(psi_t.shape[0]))
  S_vals = np.zeros((L + 1, L + 1))
  for i in range(L + 1):
    for j in range(i + 1, L + 1):
      S = EntanglementEntropy(psi_t.toarray(), i, j)
      S_vals[i, j] = S_vals[j, i] = S
  return S_vals