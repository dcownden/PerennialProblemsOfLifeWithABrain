def drift_propose_and_test(batch_size=25, high_batch_size=900,
                           initial_params=None,
                           max_rejected=500,
                           step_scale=10.0,
                           verbose=True,
                           drift_probs = [0, 0, 1.0, 0, 0]):

  game = GridworldGame(batch_size=batch_size, n_rows=7, n_cols=7,
                       num_critters=1, num_food=10, lifetime=30,
                       rng=np.random.default_rng(48),
                       drift_probs=drift_probs,
                       wrapping=True, drift_after_move=False)
  high_batch_game = GridworldGame(batch_size=high_batch_size, n_rows=7, n_cols=7,
                                  num_critters=1, num_food=10, lifetime=30,
                                  rng=np.random.default_rng(48),
                                  drift_probs=drift_probs,
                                  wrapping=True, drift_after_move=False)
  # Initialize parameters
  if initial_params is None:
    initial_params = np.zeros(48)
  best_params = initial_params
  best_avg_score = evaluate(best_params, high_batch_game)
  print(f"Initial score: {best_avg_score}")
  rejected_count = 0
  total_tests = 0  # Number of iterations
  std_dev = step_scale  # Standard deviation for Gaussian proposal
  if verbose:
    intermediate_params = []
    intermediate_params.append(best_params)
    intermediate_values = [best_avg_score]
    iterations = [0]

  # Propose-and-test loop
  start_time = time.time()
  while rejected_count < max_rejected:
    total_tests +=1
    # Propose new parameters: sample from Gaussian centered at best_params
    delta_params = np.random.normal(0, std_dev, best_params.shape)
    proposal_params = best_params + delta_params
    avg_score = evaluate(proposal_params, game)
    # If a promising candidate is found, validate it with a high batch size evaluation
    if avg_score > best_avg_score:
      avg_score_high_batch = evaluate(proposal_params, high_batch_game)
      # Only update best parameters if the candidate also performs well in the
      # high batch size evaluation to avoid choosing parameters based on 'luck'
      # i.e. from a really exceptional batch of simulations
      if avg_score_high_batch > best_avg_score:
        best_avg_score = avg_score
        best_params = proposal_params
        if verbose:
          #print('best params so far:')
          #display(best_params)
          print(f"Best score so far: {best_avg_score}")
          print(f"Found after a total of {time.time() - start_time:.2f} seconds and an additional {rejected_count} tests")
          intermediate_params.append(best_params)
          intermediate_values.append(best_avg_score)
          iterations.append(total_tests)
        rejected_count = 0
      else:
        rejected_count += 1
    else:
      rejected_count += 1
  end_time = time.time()
  elapsed_time = end_time - start_time
  iterations.append(total_tests)

  if verbose:
    # Print the best found parameters and score
    print("Best Parameters:", best_params)
    print("Best Average Score:", best_avg_score)
    print("Parameter combinations tested:", total_tests)
    print(f"Time taken for the optimization loop: {elapsed_time:.2f} seconds")
    return (best_params, best_avg_score, intermediate_params,
            intermediate_values, iterations)
  else:
    return best_params, best_avg_score

drift_result = drift_propose_and_test(initial_params=None,
                                      drift_probs=[0, 0, 1.0, 0, 0])
best_drift_params = drift_result[0]