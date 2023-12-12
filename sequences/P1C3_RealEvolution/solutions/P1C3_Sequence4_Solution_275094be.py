

def evo_learning_simulation(
  pop_size=1000, n_gen=200, num_blocks=20, block_size=1,
  has_changing_environment=True, change_env_prop = 0.5, env_change_rate=0.0,
  has_learning=True, n_learning_trials=100,
  mutation_rate=0.025, has_mutation=True,
  has_recombination=True, has_assortative_pairing=True, recombination_type='crossover',
  selection_type='softmax', #softmax, deterministic truncation, proportional truncation
  softmax_temp=1, truncation_threshold=0.5,
  proportion_shift=0.0, proportion_scale=1.0,
  seed=None, init_pop='simple',
  compute_IQR=False):
  """
  Simulates the evolutionary process in a population of genotypes. This function
  models the evolution of genotypes through various mechanisms like learning,
  selection, mutation, recombination, and environmental changes. It is intended
  to explore how populations adapt to dynamic environments.

  Args:
    pop_size (int): Size of the population.
    n_gen (int): Number of generations to simulate.
    num_blocks (int): Number of receptors/genetic traits in each individual.
    has_changing_environment (bool): If True, the environment changes over generations.
    change_env_prop (float): Proportion of the environment that is subject to change.
    env_change_rate (float): Rate at which the environment changes.
    has_learning (bool): If True, individuals can learn during their lifetime.
    n_learning_trials (int): Number of learning trials per individual.
    mutation_rate (float): Rate of mutation in the population.
    has_mutation (bool): If True, mutations occur in the population.
    has_recombination (bool): If True, recombination occurs during reproduction.
    has_assortative_pairing (bool): If True, assortative mating is used in recombination.
    selection_type (str): Type of selection mechanism used.
    softmax_temp (float): Temperature parameter for softmax selection.
    truncation_threshold (float): Threshold for truncation selection.
    proportion_shift (float): Shift parameter for proportional selection.
    proportion_scale (float): Scale parameter for proportional selection.
    seed (int): Seed for random number generator.
    init_pop (str): Type of initial population ('simple' or 'adapted').
    compute_IQR (bool): If True, compute the interquartile range of the population scores

  Returns:
    dict: A dictionary containing historical data of various statistics over generations.
  """
  # Set seed
  rng = np.random.default_rng(seed)

  if not has_learning:
    # with only one trial there is no learning
    n_learning_trials = 1

  # intialize the target connection pattern and how it changes each generation
  base_genome_shape = (num_blocks, block_size)
  if has_changing_environment:
    env_target = rng.integers(0, 2, size=base_genome_shape, dtype=np.bool_)
    num_elements = np.prod(base_genome_shape)
    env_change_mask = np.zeros(base_genome_shape, dtype=np.bool_)
    num_changes = int(change_env_prop * num_elements)
    change_indices_1d = rng.choice(num_elements, size=num_changes, replace=False)
    change_indices = np.unravel_index(change_indices_1d, base_genome_shape)
    env_change_mask[change_indices] = True
  else:
    env_target = np.ones(base_genome_shape, dtype=np.bool_)
    env_change_mask = np.zeros(base_genome_shape, dtype=np.bool_)

  genome_shape = (pop_size,) + base_genome_shape + (2,)
  # the two in the last dim is for preset and flexible

  # intialize the population genotypes
  if init_pop == 'simple':
    g_presets = rng.integers(0, 2, size=genome_shape[:-1], dtype=np.bool_)
    g_flexible = rng.integers(0, 2, size=genome_shape[:-1], dtype=np.bool_)
  elif init_pop == 'adapted':
    # For an adapted population, presets match the current environment
    reps = [pop_size] + [1] * len(env_target.shape)
    g_presets = np.tile(env_target, reps)  # Replicate env_target for each individual
    g_flexible = np.zeros(genome_shape[:-1], dtype=np.bool_)
  population_genome = np.stack((g_presets, g_flexible), axis=len(genome_shape)-1) # pop_size x receptors x x block_size x 2

  # ensure even divisors/multiples for deterministic truncation
  if selection_type == 'deterministic truncation':
    num_parents = pop_size * (1 - truncation_threshold)
    offspring_per_parent = pop_size / num_parents
    # Check if the numbers are close to integers, and even number of parents
    if not (
        np.isclose(num_parents, np.round(num_parents)) and
        np.isclose(offspring_per_parent, np.round(offspring_per_parent)) and
        num_parents % 2 == 0):
      print(truncation_threshold)
      print(pop_size)
      print(num_parents)
      print(offspring_per_parent)
      raise ValueError("For deterministic truncation, num_parents must be an even integer and divisor of pop_size, and offspring_per_parent must be an integer")
    num_parents = int(num_parents)
    offspring_per_parent = int(offspring_per_parent)

  # Track statistics genotypes and scores over generations.
  # Note, depending on simulation type not all of these will be
  # updated and returned
  mean_good_bits_hist = np.zeros(n_gen)
  mean_score_hist = np.zeros(n_gen)
  mean_flexible_change_hist = np.zeros(n_gen)
  mean_flexible_no_change_hist = np.zeros(n_gen)
  var_score_hist = np.zeros(n_gen)
  skew_score_hist = np.zeros(n_gen)
  lower_score_quartile_hist = np.zeros(n_gen)
  upper_score_quartile_hist = np.zeros(n_gen)

  # Helper function calculate score of params against the environmental target
  def score_calc(params, target):
    # params has shape pop_size x num_blocks x block_size
    scores = np.sum(np.prod(np.array(params == target, dtype=float), axis=2), axis=1)
    return scores

  # Run the simulation
  for generation in range(n_gen):
    # Calculate scores for each genotype
    g_presets = population_genome[..., 0]
    g_flexible = population_genome[..., 1]
    is_fixed = np.zeros(pop_size, dtype=np.bool_)
    if has_learning:
      best_scores = np.zeros(pop_size)
      cumulative_scores = np.zeros(pop_size)
      best_params = np.copy(g_presets)  # Initially, best parameters are the presets
      composite_params = np.copy(g_presets)  # Initially, composite parameters are the presets
      for t in range(n_learning_trials):
        # Explore only for individuals still learning
        still_learning = ~is_fixed
        expanded_shape = (pop_size,) + (1,) * (g_flexible.ndim - 1)
        still_learning_expanded = still_learning.reshape(expanded_shape)
        to_update = np.logical_and(still_learning_expanded, g_flexible)
        # Generate new samples only for the part of composite_params to be updated
        update_param_samples = rng.integers(0, 2, size=to_update.sum(), dtype=np.bool_)
        composite_params[to_update] = update_param_samples
        # trial scores are computed for everyhone
        trial_scores = score_calc(composite_params, env_target)
        # but only learners should improve of decline
        improved = trial_scores > best_scores
        declined = trial_scores < best_scores
        best_scores[improved] = trial_scores[improved]
        best_params[improved] = composite_params[improved]  # Update best parameters
        # our learning rule is stop after any improvement
        is_fixed[improved] = True
        # or if things get worse stop exploring and use the known best
        is_fixed[declined] = True
        composite_params[declined] = best_params[declined]
        cumulative_scores += trial_scores
      scores = cumulative_scores / n_learning_trials
    else:
      scores = score_calc(g_presets, env_target)

    # Track statistics of genotype scores over generations.
    mean_ = np.mean(scores)
    var_ = np.var(scores)
    if np.std(scores)>0:
      skew_ = ((pop_size / (pop_size - 1) / (pop_size-2)) *
        (np.sum((scores-np.mean(scores))**3) / np.std(scores)))
    else:
      skew_ = 0

    flexible_env_and_g = g_flexible[:, env_change_mask]
    if any(dim == 0 for dim in flexible_env_and_g.shape):
      mean_flexible_change_ = 0
    else:
      mean_flexible_change_ = np.mean(np.mean(flexible_env_and_g, axis=1))

    fixed_env_flex_g = g_flexible[:, ~env_change_mask]
    if any(dim == 0 for dim in fixed_env_flex_g.shape):
      mean_flexible_no_change_ = 0
    else:
      mean_flexible_no_change_ = np.mean(np.mean(fixed_env_flex_g, axis=1))

    mean_good_bits_ = np.mean(np.sum(g_presets == env_target, axis=1))
    mean_score_hist[generation] = mean_
    var_score_hist[generation] = var_
    skew_score_hist[generation] = skew_
    lower_score_quartile_hist[generation] = np.percentile(scores, 25)
    upper_score_quartile_hist[generation] = np.percentile(scores, 75)
    mean_flexible_change_hist[generation] = mean_flexible_change_
    mean_flexible_no_change_hist[generation] = mean_flexible_no_change_
    mean_good_bits_hist[generation] = mean_good_bits_

    # Selective Reproduction of different types
    if has_recombination and has_assortative_pairing:
      # Shuffle and then sort the population by scores
      shuffled_indices = np.arange(pop_size)
      rng.shuffle(shuffled_indices)
      population_genome = population_genome[shuffled_indices]
      scores = scores[shuffled_indices]
      # Sort the population by scores
      sorted_indices = np.argsort(scores)[::-1]  # Higher scores are better
      population_genome = population_genome[sorted_indices]
      scores = scores[sorted_indices]

    if selection_type == 'proportional':
      pos_scores = np.where(scores < 0, 0, scores)
      scale_shift_scores = (pos_scores * proportion_scale) + proportion_shift
      if np.sum(scores) > 0:
        prob_scores = scale_shift_scores / np.sum(scale_shift_scores)
      else:
        prob_scores = np.ones_like(scores) / len(scores)
    elif selection_type == 'softmax':
      stabilized_scores = scores - np.max(scores)
      exp_scaled_scores = np.exp(stabilized_scores / softmax_temp)
      prob_scores = exp_scaled_scores / np.sum(exp_scaled_scores)
    elif selection_type == 'proportional truncation':
      pos_scores = np.where(scores < 0, 0, scores)
      trunc_scores = np.zeros_like(scores)
      selected = scores >= np.quantile(scores, truncation_threshold)
      trunc_scores[selected] = pos_scores[selected]
      if np.sum(trunc_scores) > 0:
        prob_scores = trunc_scores / np.sum(trunc_scores)
      else:
        prob_scores = np.ones_like(scores) / len(scores)
    elif selection_type == 'deterministic truncation':
      threshold_score = np.quantile(scores, truncation_threshold)
      # Indices of individuals who meet or exceed the threshold score
      eligible_indices = np.where(scores >= threshold_score)[0]
      # If there are more eligible individuals than needed, take only the top
      # num_parents individuals
      selected_indices = sorted(eligible_indices, key=lambda x: scores[x], reverse=True)[:num_parents]
      # split these into parent groups
      parent_set_1 = selected_indices[::2]
      parent_set_2 = selected_indices[1::2]
      # Replicate each parent in their respective sets
      parent_set_1 = np.repeat(parent_set_1, offspring_per_parent)
      parent_set_2 = np.repeat(parent_set_2, offspring_per_parent)
      # Shuffle the parent sets based on mating strategy
      if not has_assortative_pairing:
        rng.shuffle(parent_set_2)
      # Combine the two parent sets to form genome pairs
      parent_pairs_shape = (pop_size // 2, 2,) + env_target.shape + (2,)
      parent_pairs = np.zeros(parent_pairs_shape, dtype=np.bool_)
      parent_pairs[:, 0, ...] = population_genome[parent_set_1]
      parent_pairs[:, 1, ...] = population_genome[parent_set_2]
    else:
      raise ValueError("Invalid selection_type string, use one of proportional, softmax, proportional truncation, or deterministic truncation")

    if selection_type != 'deterministic truncation':
      # form parent pairs
      parent_pairs_shape = (pop_size // 2, 2,) + env_target.shape + (2,)
      parent_pairs = np.zeros(parent_pairs_shape, dtype=np.bool_)

      if has_recombination:
        selected_parent_indices_1 = rng.choice(pop_size, size=pop_size // 2,
                                               p=prob_scores, replace=True)
        pair_directions = rng.integers(0, 2, size=pop_size // 2) * 2 - 1  # Results in either -1 or 1
        if has_assortative_pairing:
          # Use pair direction for assortative mating
          selected_parent_indices_2 = selected_parent_indices_1 + pair_directions
        else:
          # For non-assortative mating, select a second set of parents
          selected_parent_indices_2 = rng.choice(pop_size, size=pop_size // 2,
                                                 p=prob_scores, replace=True)
          # Correct self-pairing
          selected_parent_indices_2 = np.where( #ternary use of where
            selected_parent_indices_1 == selected_parent_indices_2, # if this condition
            (selected_parent_indices_2 + pair_directions) % pop_size, # do this
            selected_parent_indices_2) # other wise do this
        # Adjust any out-of-bounds indices
        selected_parent_indices_2[selected_parent_indices_2 < 0] = 1
        selected_parent_indices_2[selected_parent_indices_2 >= pop_size] = pop_size - 1
      else: # no recombination
        selected_parent_indices_1 = rng.choice(pop_size, size=pop_size // 2,
                                               p=prob_scores, replace=True)
        selected_parent_indices_2 = rng.choice(pop_size, size=pop_size // 2,
                                               p=prob_scores, replace=True)
      # use the selected indices to form the parent genome pairs
      parent_pairs[:, 0, ...] = population_genome[selected_parent_indices_1]
      parent_pairs[:, 1, ...] = population_genome[selected_parent_indices_2]

    # recombination
    if has_recombination:
      if recombination_type == 'random':
        mask_shape = (pop_size // 2,) + env_target.shape + (2,)
        mask1 = rng.integers(0, 2, size=mask_shape).astype(np.bool_)
        mask2 = rng.integers(0, 2, size=mask_shape).astype(np.bool_)
        children = np.empty_like(parent_pairs)
        children[:, 0, ...] = np.where(mask1, parent_pairs[:, 0, ...], parent_pairs[:, 1, ...])
        children[:, 1, ...] = np.where(mask2, parent_pairs[:, 0, ...], parent_pairs[:, 1, ...])
      elif recombination_type == 'crossover':
        # Flatten the genomes for crossover operations
        # Shape before flattening: (pop_size // 2, 2, num_blocks, block_size, 2)
        flat_parent_pairs = parent_pairs.reshape((pop_size // 2, 2, -1))
        # Shape after flattening: (pop_size // 2, 2, num_blocks * block_size * 2)
        # Generate two sets of unique crossover points for each parent pair
        crossover_points_1 = rng.integers(1, flat_parent_pairs.shape[-1], size=(pop_size // 2,))
        crossover_points_2 = rng.integers(1, flat_parent_pairs.shape[-1], size=(pop_size // 2,))
        # Create a range array that matches the last dimension of flat_parent_pairs
        # to help make the mask for crossover operations
        range_array = np.arange(flat_parent_pairs.shape[-1])
        # Use broadcasting to create masks: True if index is less than the crossover point
        mask1 = range_array < crossover_points_1[:, np.newaxis]
        mask2 = range_array < crossover_points_2[:, np.newaxis]
        mask2 = ~mask2
        children_flat = np.empty_like(flat_parent_pairs)
        children_flat[:, 0, :] = np.where(mask1, flat_parent_pairs[:, 0, :], flat_parent_pairs[:, 1, :])
        children_flat[:, 1, :] = np.where(mask2, flat_parent_pairs[:, 0, :], flat_parent_pairs[:, 1, :])
        children = children_flat.reshape(parent_pairs.shape)
      else:
        raise ValueError("Invalid recombination_type string, use one of random or crossover")
      recombined_population_genome = children.reshape(genome_shape)
      population_genome = recombined_population_genome
    else:
      # the population is just the selected parents
      population_genome = parent_pairs.reshape(genome_shape)

    # mutation
    if has_mutation:
      mutation_mask = rng.random(population_genome.shape) < mutation_rate
      population_genome ^= mutation_mask

    # dynamic environment
    if has_changing_environment:
      # see which aspect of the environment change and update them
      did_change = env_change_mask & (rng.random(size = env_change_mask.shape) < env_change_rate)
      # flip the bits where change happened
      env_target[did_change] = ~env_target[did_change]

    # repeat the loop for n_gen iterations

  results = {
    'mean_score_hist': mean_score_hist,
    'var_score_hist': var_score_hist,
    'skew_score_hist': skew_score_hist,
    'mean_good_bits_hist': mean_good_bits_hist
  }
  if compute_IQR:
    results['lower_score_quartile_hist'] = lower_score_quartile_hist
    results['upper_score_quartile_hist'] = upper_score_quartile_hist,
  if has_changing_environment:
    results['mean_flexible_change_hist'] = mean_flexible_change_hist
    results['mean_flexible_no_change_hist'] = mean_flexible_no_change_hist
  return results

# Parameters
common_params = {
    'n_gen': 200,
    'pop_size': 1000,
    'init_pop': 'simple',
    'seed': 123,
    'mutation_rate': 0.01,
    'has_mutation': True,
    'has_recombination': True,
    'has_assortative_pairing': True,
    'recombination_type': 'crossover',
    'has_learning': False,
    'n_learning_trials': 1,
    'proportion_shift': 1.0,
    'proportion_scale': 19.0,
    'has_changing_environment':False,
    'change_env_prop': 0.0,
    'env_change_rate': 0.0,
    'compute_IQR': False,
}

# Create 3x1 subplots
fig, axs = plt.subplots(3, 1, figsize=(10, 15))

selection_types = ['proportional', 'deterministic truncation', 'softmax']
base_colors = {'proportional': 'Blues', 'deterministic truncation': 'Reds', 'softmax': 'Greens'}
labels = {'proportional': 'Proportional', 'deterministic truncation': 'Deterministic Truncation', 'softmax': 'Softmax'}

# Define parameter dictionaries for each block size
#vary num_blocks with block_size to keep constant genotype length = num_blocks * block_size
block_sizes =       [ 2, 5, 10, 20]
numbers_of_blocks = [10, 4,  2,  1]
block_params = []
for block_size, num_blocks in zip(block_sizes, numbers_of_blocks):
  params = {'block_size': block_size, 'num_blocks': num_blocks}
  block_params.append(params)

# Plot for each selection type
for ax, selection_type in zip(axs, selection_types):
  base_color = base_colors[selection_type]
  base = matplotlib.colormaps[base_color]
  num_shades = len(block_sizes)
  colors = [base((i+2)/(num_shades+2)) for i in range(num_shades)]
  label = labels[selection_type]

  for block_param, color in zip(block_params, colors):
    # Combine parameters
    combined_params = {**common_params,
                       **block_param,
                       'selection_type': selection_type}
    r = evo_learning_simulation(**combined_params)
    generations = np.arange(combined_params['n_gen'])

    # normalized mean score and quartiles
    mean_hist = r['mean_score_hist']/combined_params['num_blocks']
    ax.plot(generations, mean_hist, color=color, label=label + " - Block Size: {block_param['block_size']}, Block Number: {block_param['num_blocks']}")
  norm = matplotlib.colors.Normalize(vmin=min(block_sizes), vmax=max(block_sizes))
  scalar_mappable = matplotlib.cm.ScalarMappable(norm=norm, cmap=base)

  ax.set_title(f'Selection Type: {selection_type}')
  ax.set_xlabel('Generation')
  ax.set_ylabel('Normalized Mean Score')
  #ax.legend()
  cbar = fig.colorbar(scalar_mappable, ax=ax, orientation='vertical')
  cbar.set_label('Bits per Block')

plt.tight_layout()
remove_ip_clutter(fig)
plt.show()