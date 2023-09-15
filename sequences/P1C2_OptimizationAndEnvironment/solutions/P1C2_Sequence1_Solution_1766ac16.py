game = GridworldGame(batch_size=9, n_rows=7, n_cols=7,
                     num_critters=1, num_food=10, lifetime=30,
                     rng=np.random.default_rng(48))
best_avg_score = float('-inf')
best_params = None

def convert_symmetry_to_weights(symmetry_params):
  # Initialize the weight matrix with zeros
  weights = np.zeros((4,12))
  symmetry_indices = {
    'Up':    [0,  1,  2,  1,  3,  4,  4,  3,  5,  6,  5,  7],
    'Down':  [7,  5,  6,  5,  3,  4,  4,  3,  1,  2,  1,  0],
    'Left':  [3,  1,  4,  5,  0,  2,  6,  7,  1,  4,  5,  3],
    'Right': [3,  5,  4,  1,  7,  6,  2,  0,  5,  4,  1,  3]}
  # Use the symmetry indices to populate the 48-dimensional weight vector
  for i, direction in enumerate(['Up', 'Down', 'Left', 'Right']):
    for j, idx in enumerate(symmetry_indices[direction]):
      weights[i, j] = symmetry_params[idx]
  return weights

# Loop through each combination
for params in tqdm(param_combinations):
  # Convert symmetry parameters to the actual weights
  weights = convert_symmetry_to_weights(params)

  # Run the game with the weights
  boppp = BatchOptPerceptParamPlayer(game, weights=weights, deterministic=True)
  final_board = game.play_game(players=[boppp], visualize=False)

  # Evaluate the score
  scores = final_board['scores'].flatten()
  avg_score = np.mean(scores)

  # Update best parameters if needed
  if avg_score > best_avg_score:
    best_avg_score = avg_score
    best_params = params

print(best_params)
print(best_avg_score)