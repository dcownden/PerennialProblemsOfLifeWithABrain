
def np_softmax(x, tau=1):
  # high tau more exploration, low tau very little exploration
  x_scaled = x / tau
  # Shift x by subtracting the max value to prevent overflow in exp
  x_shifted = x_scaled - np.max(x_scaled, axis=0, keepdims=True)
  exps = np.exp(x_shifted)
  # Normalize the exponentials while maintaining batch structure
  softmax_output = exps / np.sum(exps, axis=0, keepdims=True)
  return softmax_output

def eval_params_multi(W, x, y, tau=1, rng=None):
  # W shape is num_action x input
  # x shape is input x batch
  # y shape is 1 x batch
  if rng is None:
    rng = np.random.default_rng()
  z = np.dot(W, x)  # num_action x batch
  pi = np_softmax(z, tau)  # num_action x batch
  cumulative_probs = np.cumsum(pi, axis=0)
  random_samples = rng.random(size=cumulative_probs.shape[1])
  #sampled actions
  a = (cumulative_probs > random_samples).argmax(axis=0)
  # calculate which actions were correct and compute reward
  correct = a == y
  r = np.zeros_like(y)
  r[correct] = 1
  r[~correct] = -1  # sampled reward
  # Create an outcomes matrix to calculate expected reward
  outcomes_matrix = -np.ones_like(pi)
  # Set reward to +1 at the position of the correct action for each sample
  outcomes_matrix[np.squeeze(y), np.arange(z.shape[1])] = 1
  r_exp = np.sum(pi * outcomes_matrix, axis=0)  # expected reward
  return z, pi, a, r, outcomes_matrix, r_exp

def reward_prediction_multi(W, x, y,
                            tau=1.0,
                            rng=None,
                            learning_rate=0.0001):
  z, pi, a, r, outcomes_matrix, r_exp = eval_params_multi(W, x, y, tau, rng)
  # learn only from actual received rewards
  errors = np.zeros_like(z)
  actual_errors = r - z[a, np.arange(len(a))]
  errors[a, np.arange(len(a))] = actual_errors
  # implicit sum over batch here in this matrix multiplication
  update = errors @ x.T  # errors is num_actions x batch, x.T is batch x num_features, update is num_actions x num_features
  update /= x.shape[1]  # Divide by the number of samples in batch to make the sum an average
  W_new = W + learning_rate * update  # Apply learning rate to update step
  return W_new, r, r_exp

def action_prob_multi(W, x, y,
                      tau=1,
                      rng=None,
                      learning_rate=0.001,
                      test=False):
  z, pi, a, r, outcomes_matrix, r_exp = eval_params_multi(W, x, y, tau, rng)
  num_actions, batch_size = pi.shape
  num_features = x.shape[0]
  pi_a = pi[a, np.arange(len(a))]  # [batch_size]
  broadcast_pi_a = np.broadcast_to(pi_a.reshape(1,batch_size), (num_actions, batch_size))
  # Compute updates for case when row (i) of W does not correspond to the sampled action
  delta_W = np.zeros((num_actions, num_features, batch_size))
  delta_W -= r[np.newaxis,:,:] * broadcast_pi_a[:,np.newaxis,:] * pi[:,np.newaxis,:] * x[np.newaxis,:,:]  # Shape [num_actions, num_features, batch_size]
  # now compute updates for case when row (i) of W does correspond to the sampled action
  mask = np.arange(num_actions)[:, None] == a[None, :]  # [num_actions, batch_size]
  mask = np.broadcast_to(mask[:,np.newaxis,:], delta_W.shape) # [num_actions, num_features, batch_size]
  positive_update = r[np.newaxis,:,:] * (pi * (1 - pi))[:,np.newaxis,:] * x[np.newaxis,:,:]  # [num_features, num_actions, batch_size]
  # Use positive update where appropriate
  delta_W[mask] = positive_update[mask]
  # average over the elements of the mini-batch
  delta_W = np.mean(delta_W, axis=2)
  W_new = W + learning_rate * delta_W
  if test:
    # as a sanity check on all the clever broadcasting and array operations
    # check against bog simple for loop implementation
    delta_W_test = np.zeros_like(delta_W)
    for b in range(batch_size):
      for i in range(num_actions):
        for j in range(num_features):
          if i == a[b]:
            delta_W_test[i,j,b] = r[0,b] * pi[i,b] * (1 - pi[i,b]) * x[j,b]
          else:
            delta_W_test[i,j,b] = -r[0,b] * pi_a[b] * pi[i,b] * x[j,b]
    assert np.allclose(delta_W, delta_W_test)
  return W_new, r, r_exp

def action_log_prob_multi(W, x, y,
                          tau=1,
                          rng=None,
                          learning_rate=0.001,
                          test=False):
  z, pi, a, r, outcomes_matrix, r_exp = eval_params_multi(W, x, y, tau, rng)
  num_actions, batch_size = pi.shape
  num_features = x.shape[0]
  # Compute updates for case when row (i) of W does not correspond to the sampled action
  a_1hot = np.zeros_like(pi)  # [num_actions, batch_size]
  a_1hot[a, np.arange(batch_size)] = 1
  pi_term = a_1hot - pi
  delta_W = r[np.newaxis,:,:] * pi_term[:,np.newaxis,:] * x[np.newaxis,:,:]  # Shape [num_actions, num_features, batch_size]
  # average over the elements of the mini-batch
  delta_W = np.mean(delta_W, axis=2)
  W_new = W + learning_rate * delta_W
  if test:
    # as a sanity check on all the clever broadcasting and array operations
    # check against bog simple for loop implementation
    delta_W_test = np.zeros_like(delta_W)
    for b in range(batch_size):
      for i in range(num_actions):
        for j in range(num_features):
          if i == a[b]:
            delta_W_test[i,j,b] = r[0,b] * (1 - pi[i,b]) * x[j,b]
          else:
            delta_W_test[i,j,b] = -r[0,b] * pi[i,b] * x[j,b]
    assert np.allclose(delta_W, delta_W_test)
  return W_new, r, r_exp

############### Exercise Complete ###############
##### Simulation and Plotting Logic Follows #####

def always_cheat_perturb_measure_multi(W, x, y,
                                      perturbation_scale=0.0001,
                                      tau=1,
                                      rng=None,
                                      learning_rate=0.001):
  if rng is None:
    rng = np.random.default_rng()
  z, pi, a, r, outcomes_matrix, r_exp = eval_params_multi(W, x, y, tau, rng)
  raw_test_perturb = learn_rng.standard_normal(size=(W.shape))
  unit_test_perturb = raw_test_perturb / np.linalg.norm(raw_test_perturb.flatten())
  test_perturbation = unit_test_perturb * perturbation_scale
  perturbed_W = W + test_perturbation
  _, _, _, r_perturb, _, r_exp_perturb = eval_params_multi(perturbed_W, x, y, tau, rng)
  directional_grad_est = (np.mean(r_exp_perturb - r_exp)) / perturbation_scale
  update = learning_rate * directional_grad_est * unit_test_perturb
  W_new = W + update
  return W_new, r, r_exp

# simulation
learn_rng = np.random.default_rng(0)
num_epochs = 10000
num_steps = 0
mini_batch_size = 281
cooling_rate = 0.01
W_init = np.zeros((10,65))
indices = np.arange(Xs_aug.shape[0])
batch_x = Xs_aug.T
batch_y = y1.T
alg_names = ['Reward Prediction', 'Action Probability', 'Action Log Probability', 'Perturb Measure']
alg_funcs = [reward_prediction_multi, action_prob_multi, action_log_prob_multi, always_cheat_perturb_measure_multi]
alg_lrs = {'Reward Prediction': 0.0001,
           'Action Probability': 0.008,
           'Action Log Probability': 0.004,
           'Perturb Measure': 0.01}
reward_results = {alg_name: [] for alg_name in alg_names}
exp_reward_results = {alg_name: [] for alg_name in alg_names}
W_s = {alg_name: W_init.copy() for alg_name in alg_names}
for epoch in range(num_epochs):
  learn_rng.shuffle(indices)
  for batch_step in range(0, Xs_aug.shape[0], mini_batch_size):
    batch_indices = indices[batch_step:batch_step+mini_batch_size]
    batch_x = Xs_aug[batch_indices].T
    batch_y = y1[batch_indices].T
    for alg_name, alg_func in zip(alg_names, alg_funcs):
      lr = alg_lrs[alg_name]
      W = W_s[alg_name]
      if alg_name == 'Reward Prediction':
        tau = 1/((num_steps+10.0) * cooling_rate)
      else:
        tau = 1.0
      # perturb-measure-step uses r_exp everything else just uses r
      new_W, r, r_exp = alg_func(W, batch_x, batch_y, tau=tau, rng=learn_rng, learning_rate=lr)
      _, _, _, _, _, r_exp_full = eval_params_multi(W, Xs_aug.T, y1.T, tau=0.00001, rng=learn_rng)
      W_s[alg_name] = new_W
      reward_results[alg_name].append(np.mean(r))
      exp_reward_results[alg_name].append(np.mean(r_exp_full))
    num_steps += 1
  if num_steps > 2000:
    break

with plt.xkcd():
  fig, ax1 = plt.subplots()
  theoretical_max = 1.0
  # Colors for algorithms
  colors = {'Reward Prediction': 'blue', 'Action Probability': 'green',
            'Action Log Probability': 'orange', 'Perturb Measure': 'red'}

  # First subplot for expected rewards
  ax1.hlines(theoretical_max, 0, num_steps, linestyle='--', color='gray', label='Max Possible Avg. Reward per Episode')
  for alg_name in alg_names:
    eval = np.array(exp_reward_results[alg_name])
    eval = np.cumsum(eval)
    eval = eval / (np.arange(len(eval)) + 1)
    ax1.plot(eval, linestyle='-', color=colors[alg_name], label=f'{alg_name}')

    ax1.set_title('Cumulative per Episode Average of\nExpected (Full Batch) Reward')
    ax1.set_ylabel('Cumulative Avg. Expected Reward')
    ax1.legend()
  plt.show()