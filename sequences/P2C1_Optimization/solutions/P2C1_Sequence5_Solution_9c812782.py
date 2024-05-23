
def np_sigmoid(x, tau=1):
  # high tau more exploration, low tau very little exploration
  x_scaled = np.clip(x/tau, -500, 500) #prevent overflow, fine because sigmoid saturates
  return 1 / (1 + np.exp(-x/tau))

def eval_params(W, x, y, tau=1, rng=None):
  """
  Parameters:
  - W (ndarray, shape: (1, n_inputs)): Connective strength weights between inputs and the output.
  - x (ndarray, shape: (n_inputs, batch)): Input features, col is a sample, each row an input feature type.
  - y (ndarray, shape: (1, batch)): Binary indication of prey presence, used to determine the reward.
  - tau (float, optional): Temperature parameter for the sigmoid function, controlling its steepness.
    A higher tau value leads to a steeper function.
  - rng (np.random.Generator, optional): NumPy random number generator instance for reproducibility.
  """
  if rng is None:
    rng = np.random.default_rng()
  z = np.dot(W, x) # 1 x batch = 1 x n_inputs @ n_inputs x batch
  strike_prob = np_sigmoid(z, tau) # 1 x batch
  strike = np.array(rng.random(size=strike_prob.shape) < strike_prob, int) # 1->did, 0->didn't
  r = np.where(y == 1, strike, -strike) # sampled reward
  r_all = 2*y-1 # reward when striking in all cases
  r_exp = strike_prob * r_all # + (1-strike_prob) * 0 # expected reward
  return z, strike_prob, strike, r, r_all, r_exp

def reward_prediction_step(W, x, y,
                           cheat=False,
                           tau=1.0,
                           rng=None,
                           learning_rate=0.0001):
  z, strike_prob, strike, r, r_all, r_exp = eval_params(W, x, y, tau, rng)
  if cheat:
    # learn regardless of whether organism actually strikes or not
    update = (r_all - z) * x
  else: # properly episodic
    # learn only from actual recieved rewards
    update = (r-z) * strike * x
  # average the update over all elements in the batch
  update = np.mean(update, axis=1, keepdims=True).T
  W_new = W + update * learning_rate
  return W_new, r, r_exp

def action_prob_step(W, x, y,
                     cheat=False,
                     tau=1,
                     rng=None,
                     learning_rate=0.001):
  z, strike_prob, strike, r, r_all, r_exp = eval_params(W, x, y, tau, rng)
  if cheat:
    # learn using expected reward
    update = r_exp * (strike_prob) * (1 -  strike_prob) * x
  else: # properly episodic
    # learn only from actual recieved rewards
    # reward of zero recieved when not striking
    update = r * (strike_prob) * (1 - strike_prob) * x
  update = np.mean(update, axis=1, keepdims=True).T
  W_new = W + update * learning_rate
  return W_new, r, r_exp

def perturb_measure_step(W, x, y,
                         perturbation_scale=0.01,
                         cheat=False,
                         tau=1,
                         rng=None,
                         learning_rate=0.001):
  if rng is None:
    rng = np.random.default_rng()
  z, strike_prob, strike, r, r_all, r_exp = eval_params(W, x, y, tau, rng)
  raw_test_perturb = learn_rng.standard_normal(size=(W.shape))
  unit_test_perturb = raw_test_perturb / np.linalg.norm(raw_test_perturb.flatten())
  test_perturbation = unit_test_perturb * perturbation_scale
  perturbed_W = W + test_perturbation
  _, _, _, r_perturb, r_all_perturb, r_exp_perturb = eval_params(perturbed_W, x, y, tau, rng)
  if cheat:
    # evaluate perturbation using expected reward, avg over the mini-batch
    directional_grad_est = (np.mean(r_exp_perturb - r_exp)) / perturbation_scale
  else: # more episodic (still evaluate peturb and base on same experiences)
    # evaluate perturbation using sampled rewards, avg over the mini-batch
    directional_grad_est = (np.mean(r_perturb - r)) / perturbation_scale
  update = learning_rate * directional_grad_est * unit_test_perturb
  W_new = W + update
  return W_new, r, r_exp

################################################################################
# Exercise Complete, simulations and plotting logic follow
################################################################################
# simulation
learn_rng = np.random.default_rng(0)
num_epochs = 1
num_steps = 0
mini_batch_size = 1
cooling_rate = 0.04
W_init = np.zeros((1,65))
indices = np.arange(Xs_aug.shape[0])
batch_x = Xs_aug.T
batch_y = y1.T
alg_names = ['Reward Prediction', 'Action Probability', 'Perturb Measure']
alg_funcs = [reward_prediction_step, action_prob_step, perturb_measure_step]
# cheating simulation
cheat_alg_lrs = {'Reward Prediction': 0.0001,
                 'Action Probability': 0.008,
                 'Perturb Measure': 0.0016}
cheat_reward_results = {alg_name: [] for alg_name in alg_names}
cheat_exp_reward_results = {alg_name: [] for alg_name in alg_names}
W_s = {alg_name: W_init.copy() for alg_name in alg_names}
for epoch in range(num_epochs):
  learn_rng.shuffle(indices)
  for batch_step in range(0, Xs_aug.shape[0], mini_batch_size):
    batch_indices = indices[batch_step:batch_step+mini_batch_size]
    batch_x = Xs_aug[batch_indices].T
    batch_y = y1[batch_indices].T
    for alg_name, alg_func in zip(alg_names, alg_funcs):
      lr = cheat_alg_lrs[alg_name]
      W = W_s[alg_name]
      if alg_name == 'Reward Prediction':
        tau = 1/((num_steps+10.0) * cooling_rate)
      else:
        tau = 1.0
      new_W, r, r_exp = alg_func(W, batch_x, batch_y, tau=tau, cheat=True, rng=learn_rng, learning_rate=lr)
      _, _, _, _, _, r_exp_full = eval_params(W, Xs_aug.T, y1.T, tau=0.00001, rng=learn_rng)
      W_s[alg_name] = new_W
      cheat_reward_results[alg_name].append(np.mean(r))
      cheat_exp_reward_results[alg_name].append(np.mean(r_exp_full))
    num_steps += 1
    if num_steps > 1000:
      break

num_steps = 0
real_alg_lrs = {'Reward Prediction': 0.0001,
                'Action Probability': 0.003,
                'Perturb Measure': 0.0000002}
# realishtic simulation
W_s = {alg_name: W_init.copy() for alg_name in alg_names}
actual_reward_results = {alg_name: [] for alg_name in alg_names}
actual_exp_reward_results = {alg_name: [] for alg_name in alg_names}
for epoch in range(num_epochs):
  learn_rng.shuffle(indices)
  for batch_step in range(0, Xs_aug.shape[0], mini_batch_size):
    batch_indices = indices[batch_step:batch_step+mini_batch_size]
    batch_x = Xs_aug[batch_indices].T
    batch_y = y1[batch_indices].T
    for alg_name, alg_func in zip(alg_names, alg_funcs):
      lr = real_alg_lrs[alg_name]
      W = W_s[alg_name]
      if alg_name == 'Reward Prediction':
        tau = 1/((num_steps+10.0) * cooling_rate)
      else:
        tau = 1.0
      new_W, r, r_exp = alg_func(W, batch_x, batch_y, tau=tau, cheat=False, rng=learn_rng, learning_rate=lr)
      _, _, _, _, _, r_exp_full = eval_params(W, Xs_aug.T, y1.T, tau=0.00001, rng=learn_rng)
      W_s[alg_name] = new_W
      actual_reward_results[alg_name].append(np.mean(r))
      actual_exp_reward_results[alg_name].append(np.mean(r_exp_full))
    num_steps += 1
    if num_steps > 1000:
      break
# plotting
with plt.xkcd():
  # Create subplots with a shared x-axis
  #fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 8), sharex=True)
  fig, ax1 = plt.subplots(figsize=(8,6))
  theoretical_max = np.sum(y1 == 1) / len(y1)

  # Colors for algorithms
  colors = {'Reward Prediction': 'b', 'Action Probability': 'g', 'Perturb Measure': 'r'}

  # First subplot for expected rewards
  ax1.hlines(theoretical_max, 0, num_steps, linestyle='--', color='gray', label='Max Possible Avg. Reward per Episode')
  for alg_name in alg_names:
    eval_cheat = np.array(cheat_exp_reward_results[alg_name])
    eval_cheat = np.cumsum(eval_cheat)
    eval_cheat = eval_cheat / (np.arange(len(eval_cheat)) + 1)
    ax1.plot(eval_cheat, linestyle='--', color=colors[alg_name], label=f'{alg_name}-Cheating')

    eval_real = np.array(actual_exp_reward_results[alg_name])
    eval_real = np.cumsum(eval_real)
    eval_real = eval_real / (np.arange(len(eval_real)) + 1)
    ax1.plot(eval_real, linestyle='-', color=colors[alg_name], label=f'{alg_name}')

  ax1.set_title('Cumulative per Episode Average of\nExpected (Full Batch) Reward')
  ax1.set_ylabel('Cumulative Avg. Expected Reward')
  ax1.set_xlabel('Learning Episodes')
  ax1.legend()
  # Second subplot for actual rewards
  # ax2.hlines(theoretical_max, 0, num_steps, linestyle='--', color='gray', label='Max Possible Avg. Reward per Episode')
  # for alg_name in alg_names:
  #  eval_cheat = np.array(cheat_reward_results[alg_name])
  #  eval_cheat = np.cumsum(eval_cheat)
  #  eval_cheat = eval_cheat / (np.arange(len(eval_cheat)) + 1)
  #  ax2.plot(eval_cheat, linestyle='--', color=colors[alg_name], label=f'{alg_name}-Cheating')

  #  eval_real = np.array(actual_reward_results[alg_name])
  #  eval_real = np.cumsum(eval_real)
  #  eval_real = eval_real / (np.arange(len(eval_real)) + 1)
  #  ax2.plot(eval_real, linestyle='-', color=colors[alg_name], label=f'{alg_name}-Realishtic')

  #ax2.set_title('Cumulative Average of Actual Reward Per Learning Episode')
  #ax2.set_xlabel('Learning Episodes')
  #ax2.set_ylabel('Cumulative Avg. Actual Reward')
  #ax2.legend()
  plt.tight_layout()
  plt.show()