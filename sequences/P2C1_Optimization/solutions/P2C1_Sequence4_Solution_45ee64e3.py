
# As a little trick to keep our code cleaner we 'hide' our bias term.
# We to do this by augmenting the features to include a feature that always has the value '1'.
# Then, the 'weight' associated with this feature, which always has a value of '1', effectively serves as the bias term.
# After augmentation there is one extra column of features
Xs_aug = np.hstack([Xs, np.ones((Xs.shape[0],1))])

def np_sigmoid(x):
  x = np.clip(x, -500, 500) #prevent overflow, fine because sigmoid saturates
  return 1 / (1 + np.exp(-x))

def eval_params_stochastic_batch(W, x, y, verbose=False, rng=None):
  """
  evaluates parameters of simple behaviour circuit given inputs and target
  outputs, use numpy broadcasting to be fast and concise
  Args:
    W: (outputs(1) x inputs(65) np.array)
       weights between sensory neurons and output neuron
    x: (input(65) x batch np.array) sensory input
    y: (outputs(1) x batch np.array) target behavioural output

  Returns:
    R: the reward obtained given the parameters, inputs and targets
  """
  if rng is None:
    rng = np.random.default_rng()
  if rng is None:
    rng = np.random.default_rng()
  # activation
  z = np.dot(W,x) # 1 x batch
  # strike probability
  strike_probs = np_sigmoid(z) # 1 x batch
  # what the organism actually does
  # rng.random is a sample from the uniform distribution on [0,1)
  did_strike = rng.random(size=strike_probs.shape) < strike_probs  # 1 x batch
  R = np.zeros(did_strike.shape)
  did_not_strike = np.logical_not(did_strike)
  should_strike = y == 1
  should_not_strike = y == 0
  TP = np.logical_and(did_strike, should_strike) # True Positive
  FP = np.logical_and(did_strike, should_not_strike) # False Positive
  FN = np.logical_and(did_not_strike, should_strike) # False Negative
  TN = np.logical_and(did_not_strike, should_not_strike) # True Negative
  R[TP] = 1
  R[FP] = -1
  R[FN] = 0
  R[TN] = 0
  TPs = np.sum(TP)
  FPs = np.sum(FP)
  FNs = np.sum(FN)
  TNs = np.sum(TN)
  confusion_matrix = np.array([[TPs, FNs], [FPs, TNs]])
  if verbose:
    table = [["Should Strike", TPs, FNs],
                 ["Shouldn't Strike", FPs, TNs]]
    headers = ["", "Did Strike", "Didn't Strike"]
    print("Confusion_matrix: ")
    print(tabulate(table, headers=headers, tablefmt="grid"))
    print(f'Total Reward: {np.sum(R)}')
    return None
  else:
    return np.sum(R), confusion_matrix

eval_rng = np.random.default_rng(0)
W_test = np.zeros((1,65))
# Xs_aug and y1 are batch x 65 and batch x 1, function wants transpose of this shape
# for broadcasting to work
print('Evaluation 1')
eval_params_stochastic_batch(W_test, Xs_aug.T, y1.T, verbose=True, rng=eval_rng)
print('\nEvaluation 2')
eval_params_stochastic_batch(W_test, Xs_aug.T, y1.T, verbose=True, rng=eval_rng)