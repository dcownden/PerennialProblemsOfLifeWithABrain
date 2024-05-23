
# As a little trick to keep our code cleaner we 'hide' our bias term.
# We to do this by augmenting the features to include a feature that always has the value '1'.
# Then, the 'weight' associated with this feature, which always has a value of '1', effectively serves as the bias term.
# After augmentation there is one extra column of features
Xs_aug = np.hstack([Xs, np.ones((Xs.shape[0],1))])

def np_sigmoid(x):
  x = np.clip(x, -500, 500) #prevent overflow, fine because sigmoid saturates
  return 1 / (1 + np.exp(-x))

def eval_params_stochastic_single(W, x, y, verbose=False, rng=None):
  """
  evaluates parameters of simple behaviour circuit given inputs and target
  outputs, use numpy broadcasting to be fast and concise
  Args:
    W: (outputs(1) x inputs(65) np.array)
       weights between sensory neurons and output neuron
    x: (input(65) np.array) sensory input
    y: (outputs(1) np.array) target behavioural output

  Returns:
    R: the reward obtained given the parameters, inputs and targets
  """
  if rng is None:
    rng = np.random.default_rng()
  # activaation
  z = np.dot(W,x)
  # strike probability
  strike_prob = np_sigmoid(z)
  # what the organism actually does
  # rng.random is a sample from the uniform distribution on [0,1)
  did_strike = rng.random() < strike_prob
  if did_strike == True: #organism strikes
    if y == 1: #prey is present
      R = 1
    else: # prey is not present
      R = -1
  else: # organism does not strike
    R = 0
  if verbose:
    print(f'Probability of striking: {strike_prob}')
    action_string = 'Strike' if did_strike == True else 'No Strike'
    print(f'Action taken: {action_string}')
    target_string = 'Strike' if y == 1 else 'No Strike'
    print(f'Correct Action: {target_string}')
    print(f'Reward recieved: {R}')
  else:
    return R

eval_rng = np.random.default_rng(0)
W_test = np.zeros((1,65))
eval_params_stochastic_single(W_test, Xs_aug[0], y1[0], verbose=True, rng=eval_rng)