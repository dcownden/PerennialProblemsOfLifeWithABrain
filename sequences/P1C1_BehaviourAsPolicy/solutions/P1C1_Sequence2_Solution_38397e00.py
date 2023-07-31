

def parameterized_policy(percept, rng=None, W=W, softmax_temp=None):
  """
  Determine an action based on perception.

  Args:
    percept: A 1D len 12 array representing the perception of the organism.
      Indices correspond to spaces around the organism. The values in the array
      can be -2 (out-of-bounds), 0 (empty space), or -1 (food).
    W: a 4 x 12 weight matrix parameter representing the connection strenghts
      between the 12 perceptions inputs and the 4 possible output actions.

  Returns:
    action: a str, one of 'up', 'down', 'left', 'right'. If food in one or more
    of the spaces immediately beside the organism, the function will return a
    random choice among these directions. If there is no food nearby, the
    function will return a random direction.
  """
  if rng is None:
    rng = np.random.default_rng()
  if softmax_temp is None:
    # very low temp, basically deterministic for this range of values
    softmax_temp = 0.01
  # a human interpretable overview of the percept structure
  percept_struct = [
  'far up', 'left up', 'near up', 'right up',
  'far left', 'near left', 'near right', 'far right',
  'left down', 'near down', 'right down', 'far down']
  # a human iterpretable overview of the out structure
  output_struct = ['up', 'down', 'left', 'right']
  # boolean representation of percept, no edges, just 1's where food is
  # zero otherwise
  x = np.asarray(percept == -1, int)
  # hint: Look at the equations above, the matrix-vector product (and it's
  # higher dimension generalizations) are implemented by the @ operator for
  # numpy arrays.
  output_activations = W @ x
  if np.sum(output_activations > 0):
    # softmax shift by max, scale by temp
    shift_scale_ex = np.exp((output_activations -
                             np.max(output_activations))/softmax_temp)
    sm = shift_scale_ex / shift_scale_ex.sum() #normalized
    probs_sm = sm / sm.sum(axis=0) #re-normalized again for fp precision issues
    # probs below is a naive way to get a discrete probability distribution
    # from a real valued vector, why did we use softmax normalization instead?
    # probs = output_activations / np.sum(output_activations)
    action = rng.choice(output_struct, p=probs_sm)
  else:
    action = rng.choice(output_struct)
  return action

test_action_from_perception(parameterized_policy)