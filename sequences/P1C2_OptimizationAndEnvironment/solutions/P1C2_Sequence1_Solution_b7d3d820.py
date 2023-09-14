
def propose_and_test(batch_size=25, high_batch_size=400,
                     max_rejected=100,
                     step_scale=1.0,
                     verbose=True):

  game = GridworldGame(batch_size=batch_size, n_rows=7, n_cols=7,
                       num_critters=1, num_food=10, lifetime=30,
                       rng=np.random.default_rng(48))
  high_batch_game = GridworldGame(batch_size=high_batch_size, n_rows=7, n_cols=7,
                                  num_critters=1, num_food=10, lifetime=30,
                                  rng=np.random.default_rng(48))
  # Initialize parameters
  initial_params = np.zeros(8)
  best_params = initial_params
  best_avg_score = float('-inf')
  rejected_count = 0
  total_tests = 0  # Number of iterations
  std_dev = step_scale  # Standard deviation for Gaussian proposal

  # Propose-and-test loop
  start_time = time.time()
  while rejected_count < max_rejected:
    # Propose new parameters: sample from Gaussian centered at best_params
    delta_params = np.random.normal(0, std_dev, best_params.shape)
    proposal_params = best_params + delta_params
    # Convert symmetry parameters to actual weights
    weights = convert_symmetry_to_weights(proposal_params)
    # Run the game with the proposed weights
    boppp = BatchOptPerceptParamPlayer(game, weights=weights, deterministic=False)
    final_board = game.play_game(players=[boppp], visualize=False)
    # Evaluate the score
    scores = final_board['scores'].flatten()
    avg_score = np.mean(scores)

    # If a promising candidate is found, validate it with a high batch size evaluation
    if avg_score > best_avg_score:
      boppp_high_batch = BatchOptPerceptParamPlayer(high_batch_game, weights=weights, deterministic=False)
      final_board_high_batch = high_batch_game.play_game(players=[boppp_high_batch], visualize=False)
      scores_high_batch = final_board_high_batch['scores'].flatten()
      avg_score_high_batch = np.mean(scores_high_batch)
      # Only update best parameters if the candidate also performs well in the
      # high batch size evaluation to avoid choosing parameters based on 'luck'
      # i.e. from a really exceptional batch of simulations
      if avg_score_high_batch > best_avg_score:
        best_avg_score = avg_score_high_batch
        best_params = proposal_params
        if verbose:
          #print('best params so far:')
          #display(best_params)
          print(f"Best score so far: {best_avg_score}")
          print(f"Found after {rejected_count} tests")
        rejected_count = 0
      else:
        rejected_count += 1
    else:
      rejected_count += 1
    total_tests +=1
  end_time = time.time()
  elapsed_time = end_time - start_time

  if verbose:
    # Print the best found parameters and score
    print("Best Parameters:", best_params)
    print("Best Average Score:", best_avg_score)
    print("Parameter combinations tested:", total_tests)
    print(f"Time taken for the optimization loop: {elapsed_time:.2f} seconds")
  return best_params, best_avg_score

best_params, best_avg_score = propose_and_test()