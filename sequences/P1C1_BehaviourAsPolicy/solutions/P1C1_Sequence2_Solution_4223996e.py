def simple_action_from_percept(percept, rng=None):
  """
  Determine an action based on perception.

  Args:
    percept: A 1D len 12 array representing the perception of the organism.
      Indices correspond to spaces around the organism. The values in the array
      can be -2 (out-of-bounds), 0 (empty space), or -1 (food).

  Returns:
    action: a str, one of 'up', 'down', 'left', 'right'. If food in one or more
    of the spaces immediately beside the organism, the function will return a
    random choice among these directions. If there is no food nearby, the
    function will return a random direction.
  """
  if rng is None:
    rng = np.random.default_rng()
  # a human interpretable overview of the percept structure
  percept_struct = [
  'far up', 'left up', 'near up', 'right up',
  'far left', 'near left', 'near right', 'far right',
  'left down', 'near down', 'right down', 'far down']
  # Defines directions corresponding to different perception indices
  direction_struct = [
    'None', 'None', 'up', 'None',
    'None', 'left', 'right', 'None',
    'None', 'down', 'None', 'None']
  # these are what count as nearby in the percept
  nearby_directions = ['near up', 'near down', 'near left', 'near right']
  # Get the corresponding indices in the percept array
  nearby_indices = [percept_struct.index(dir_) for dir_ in nearby_directions]
  # Identify the directions where food is located
  food_indices = [index for index in nearby_indices if percept[index] == -1]
  food_directions = [direction_struct[index] for index in food_indices]
  if len(food_directions) > 0:  # If there is any food nearby
    # If there is any food nearby randomly choose a direction with food
    return rng.choice(food_directions)  # Move towards a random one
  else:
    # If there is no food nearby, move randomly
    return rng.choice(['up', 'down', 'left', 'right'])

test_action_from_perception(simple_action_from_percept)