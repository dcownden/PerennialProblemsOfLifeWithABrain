
def selection_variation_type_simulation(
    pop_size=1000, n_gen=100, n_rcpt=100,
    mutation_rate=0.005, has_mutation=True,
    has_distinct_types=False,
    has_recombination=False, has_assortative_pairing=True,
    selection_type='proportional', #softmax, deterministic truncation, proportional truncation
    softmax_temp=1, truncation_threshold=0.5,
    seed=None, init_pop='simple'): # mutation_eq, recomb_eq
  """
  Simulates the evolutionary process in a population of genotypes playing the
  Strike-No-Strike game using different types of selection.
  """
  # Set seed
  rng = np.random.default_rng(seed)

  # Initialize population
  if init_pop == 'simple':
    population = rng.integers(0, 2, size=(pop_size, n_rcpt),
                              dtype=np.bool_)
  elif init_pop == 'adapted':
    population = np.ones((pop_size, n_rcpt), dtype=np.bool_)
  elif init_pop != 'simple':
    # for agreement with theoretical models we initialize population so that
    # variance starts at 'dynamic equilibrium' level
    if init_pop == 'mutation_eq':
      # Calculate n' and b
      n_prime = int(2 * mutation_rate * n_rcpt * (np.pi - 2))
      b = int(n_rcpt * (-2 * np.pi * mutation_rate + 4 * mutation_rate + 1) / 2)
    elif init_pop == 'recomb_eq':
      n_prime = int(n_rcpt*(np.pi - 2) / (2 + np.pi))
      b = int(2 * n_rcpt / (2 + np.pi))
    # Generate shifted binomial distribution
    num_ones = rng.binomial(n_prime, 0.5, size=pop_size) + b
    population = np.zeros((pop_size, n_rcpt), dtype=np.bool_)
    # Assign '1's randomly for each population member
    for i in range(pop_size):
      one_positions = rng.choice(n_rcpt, num_ones[i], replace=False)
      population[i, one_positions] = True

  # ensure even divisors/multiples for deterministic truncation
  if selection_type == 'deterministic truncation':
    num_parents = pop_size * (1 - truncation_threshold)
    offspring_per_parent = pop_size / num_parents
    # Check if the numbers are close to integers
    if not (np.isclose(num_parents, np.round(num_parents)) and np.isclose(offspring_per_parent, np.round(offspring_per_parent))):
      print(truncation_threshold)
      print(num_parents)
      print(offspring_per_parent)
      raise ValueError("For deterministic truncation, both pop_size * (1-truncation_threshold) and 1/(1-truncation_threshold) must result in integers.")
    num_parents = int(num_parents)
    offspring_per_parent = int(offspring_per_parent)

  # Track statistics genotype scores over generations.
  mean_hist = np.zeros(n_gen)
  mean_after_sel_hist = np.zeros(n_gen)
  var_hist = np.zeros(n_gen)
  skew_hist = np.zeros(n_gen)
  lower_quartile_hist = np.zeros(n_gen)
  upper_quartile_hist = np.zeros(n_gen)

  # label individuals as recombinors and/or mutators or not
  if has_recombination:
    if has_distinct_types:
      recombination_labels = np.zeros(pop_size, dtype=bool)
      recombination_labels[:pop_size // 2] = True
      rng.shuffle(recombination_labels)
    else:
      recombination_labels = np.ones(pop_size, dtype=bool)
  else:
    recombination_labels = np.zeros(pop_size, dtype=bool)

  if has_mutation:
    if has_distinct_types:
      mutation_labels = np.zeros(pop_size, dtype=bool)
      mutation_labels[:pop_size // 2] = True
      rng.shuffle(mutation_labels)
    else:
      mutation_labels = np.ones(pop_size, dtype=bool)
  else:
    mutation_labels = np.zeros(pop_size, dtype=bool)
  # Track statistics of distinct types if we have them
  if has_distinct_types:
    mutator_non_recombinator = mutation_labels & ~recombination_labels
    recombinator_non_mutator = ~mutation_labels & recombination_labels
    neither = ~mutation_labels & ~recombination_labels
    both = mutation_labels & recombination_labels
    present_types = {
            'mutator_non_recombinator': np.sum(mutator_non_recombinator) > 0,
            'recombinator_non_mutator': np.sum(recombinator_non_mutator) > 0,
            'neither': np.sum(neither) > 0,
            'both': np.sum(both) > 0}
    type_count_hist = {key: np.zeros(n_gen) for key, value in present_types.items() if value}
    type_mean_fitness_hist = {key: np.zeros(n_gen) for key, value in present_types.items() if value}

  # Run the simulation
  for generation in range(n_gen):
    # Calculate scores for each genotype
    scores = np.sum(np.array(population, dtype=float), axis=1)

    mean_ = np.mean(scores)
    var_ = np.var(scores)
    if np.std(scores)>0:
      skew_ = ((pop_size / (pop_size - 1) / (pop_size-2)) *
        (np.sum((scores-np.mean(scores))**3) / np.std(scores)))
    else:
      skew_ = 0
    # Track statistics genotype scores over generations.
    mean_hist[generation] = mean_
    var_hist[generation] = var_
    skew_hist[generation] = skew_
    lower_quartile_hist[generation] = np.percentile(scores, 25)
    upper_quartile_hist[generation] = np.percentile(scores, 75)

    ###########################################################################
    # Student exercises in this code block on selection types
    ###########################################################################
    # Selective Reproduction
    if selection_type == 'proportional':
      pos_scores = np.where(scores < 0, 0, scores)
      if np.sum(scores) > 0:
        prob_scores = pos_scores / np.sum(scores) # this line as exercise
      else:
        prob_scores = np.ones_like(scores) / len(scores)
    elif selection_type == 'softmax':
      stabilized_scores = scores - np.max(scores)
      exp_scaled_scores = np.exp(stabilized_scores / softmax_temp) # this line
      prob_scores = exp_scaled_scores / np.sum(exp_scaled_scores)  # and this line as exercise
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
      # Deterministic truncation selection
      threshold_score = np.quantile(scores, truncation_threshold)
      # Indices of individuals who meet or exceed the threshold score
      eligible_indices = np.where(scores >= threshold_score)[0]
      # If there are more eligible individuals than needed, select randomly
      if len(eligible_indices) > num_parents:
        selected_indices = rng.choice(eligible_indices, size=num_parents, replace=False)
      else:
        selected_indices = eligible_indices
      # Create the new population
      new_population = []
      new_mutation_labels = []
      new_recombination_labels = []
      for idx in selected_indices:
        new_population.extend([population[idx]] * offspring_per_parent)
        if has_mutation:
          new_mutation_labels.extend([mutation_labels[idx]] * offspring_per_parent)
        if has_recombination:
          new_recombination_labels.extend([recombination_labels[idx]] * offspring_per_parent)
      population = np.array(new_population)
      if has_mutation:
        mutation_labels = np.array(new_mutation_labels, dtype=bool)
      if has_recombination:
        recombination_labels = np.array(new_recombination_labels, dtype=bool)
    else:
      raise ValueError("Invalid selection_type string, use one of proportional, softmax, proportional truncation, or deterministic truncation")
    if selection_type != 'deterministic truncation':
      selected_indices = rng.choice(pop_size, size=pop_size, p=prob_scores, replace=True)
      population = population[selected_indices]
      if has_distinct_types:
        if has_mutation:
          mutation_labels = mutation_labels[selected_indices]
        if has_recombination:
          recombination_labels = recombination_labels[selected_indices]
    ###########################################################################
    # No more student exercises outside this code block on selection types
    ###########################################################################

    post_selection_score = np.sum(np.array(population, dtype=float), axis=1)
    mean_after_sel_hist[generation] = np.mean(post_selection_score)

    # track counts and mean fitness score of different types
    if has_distinct_types:
      # Identify each type based on mutation and recombination labels
      mutator_non_recombinator = mutation_labels & ~recombination_labels
      recombinator_non_mutator = ~mutation_labels & recombination_labels
      neither = ~mutation_labels & ~recombination_labels
      both = mutation_labels & recombination_labels
      for type_label, present in present_types.items():
        if present:
          if type_label == 'mutator_non_recombinator':
            type_indices = mutator_non_recombinator
          elif type_label == 'recombinator_non_mutator':
            type_indices = recombinator_non_mutator
          elif type_label == 'neither':
            type_indices = neither
          elif type_label == 'both':
            type_indices = both

          # Count individuals of this type
          type_count_hist[type_label][generation] = np.sum(type_indices)
          # Calculate mean fitness for this type
          if np.any(type_indices):  # To handle division by zero
            type_mean_fitness = np.mean(post_selection_score[type_indices])
          else:
            type_mean_fitness = 0
          type_mean_fitness_hist[type_label][generation] = type_mean_fitness

    # mutation
    if has_mutation:
      if has_distinct_types:
        # Apply mutation only to mutators, identified by mutation_labels
        mutators = population[mutation_labels]
        mutation_mask = rng.random(mutators.shape) < mutation_rate
        mutators ^= mutation_mask
        population[mutation_labels] = mutators
      else:
        # Apply mutation to the entire population
        mutation_mask = rng.random(population.shape) < mutation_rate
        population ^= mutation_mask

    # recombination
    if has_recombination:
      if has_distinct_types:
        recombiners = population[recombination_labels]
        non_recombiners = population[~recombination_labels]
        #make sure even number of recombiners
        if len(recombiners) % 2 != 0:
          # Move the last recombiner to the non-recombiners
          non_recombiners = np.append(non_recombiners, [recombiners[-1]], axis=0)
          recombiners = recombiners[:-1]
        num_recombiners = len(recombiners)
        recombiner_mutation_labels = mutation_labels[recombination_labels]
        non_recombiner_mutation_labels = mutation_labels[~recombination_labels]
      else:
        recombiners = population

      # Shuffle the indices of the recombing population
      shuffled_indices = rng.permutation(num_recombiners)
      shuffled_recombiners = recombiners[shuffled_indices]
      #mating can be assortative or totally random
      if has_assortative_pairing:
        shuffled_scores = np.sum(np.array(shuffled_recombiners, dtype=float), axis=1)
        # Sort the shuffled population based on scores for assortative mating
        sorted_indices = np.argsort(shuffled_scores)
        sorted_recombiners = shuffled_recombiners[sorted_indices]
        # Reshape sorted population to group similar parents into pairs
        parent_pairs = sorted_recombiners.reshape(num_recombiners // 2, 2, n_rcpt)
      else:
        # Reshape shuffled population to group similar parents into pairs
        parent_pairs = shuffled_recombiners.reshape(num_recombiners // 2, 2, n_rcpt)
      mask1 = rng.integers(0, 2, size=(num_recombiners // 2, n_rcpt)).astype(np.bool_)
      mask2 = rng.integers(0, 2, size=(num_recombiners // 2, n_rcpt)).astype(np.bool_)
      children = np.empty_like(parent_pairs)
      children[:, 0, :] = np.where(mask1, parent_pairs[:, 0, :], parent_pairs[:, 1, :])
      children[:, 1, :] = np.where(mask2, parent_pairs[:, 0, :], parent_pairs[:, 1, :])
      recombined_population = children.reshape(num_recombiners, n_rcpt)
      if has_distinct_types:
        population = np.concatenate([recombined_population, non_recombiners])
        recombiner_labels_bool = np.full(len(recombiners), True, dtype=bool)
        non_recombiner_labels_bool = np.full(len(non_recombiners), False, dtype=bool)
        recombination_labels = np.concatenate([recombiner_labels_bool, non_recombiner_labels_bool])
        if has_mutation:
          mutation_labels = np.concatenate([recombiner_mutation_labels,
                                            non_recombiner_mutation_labels])
      else:
        population = recombined_population



  results = {
    'mean_hist': mean_hist,
    'mean_after_sel_hist': mean_after_sel_hist,
    'var_hist': var_hist,
    'skew_hist': skew_hist,
    'upper_quartile_hist': upper_quartile_hist,
    'lower_quartile_hist': lower_quartile_hist}
  if has_distinct_types:
    results['type_count_hist'] = type_count_hist
    results['type_mean_fitness_hist'] = type_mean_fitness_hist
  return results


def plot_selection_results(ax, selection_type='deterministic_truncation',
                           n_gen=200, color='red', label='', seed=123,
                           softmax_temp=1, truncation_threshold=0.5,
                           mutation_rate=0.005, pop_size=1000,
                           has_distinct_types=False, has_mutation=False,
                           has_recombination=False, has_assortative_pairing=False,
                           plot_IQR=True, init_pop='simple'):
    r = selection_variation_type_simulation(
        selection_type=selection_type,
        n_gen=n_gen,
        seed=seed,
        softmax_temp=softmax_temp,
        truncation_threshold=truncation_threshold,
        mutation_rate=mutation_rate,
        has_distinct_types=has_distinct_types,
        has_mutation=has_mutation,
        has_recombination=has_recombination,
        has_assortative_pairing=has_assortative_pairing,
        pop_size=pop_size,
        init_pop=init_pop)
    generations = np.arange(n_gen)
    if plot_IQR:
      ax.fill_between(generations, r['lower_quartile_hist'],
                      r['upper_quartile_hist'],
                      color=color, alpha=0.2)
    ax.plot(generations, r['mean_hist'], color=color, label=label)

# Parameters
common_params = {
    'n_gen': 200,
    'pop_size': 1000,
    'seed': 123,
    'mutation_rate': 0.005,
    'has_mutation': True,
    'has_recombination': False,
    'has_assortative_pairing': True,
    'has_distinct_types': False
}

softmax_params = {
    'selection_type': 'softmax',
    'color': 'green',
    'label': 'Softmax',
    'softmax_temp': 1,
}

truncation_params = {
    'selection_type': 'deterministic truncation',
    'color': 'red',
    'label': 'Truncation - 50%',
    'truncation_threshold': 0.5,
}

proportional_params = {
    'selection_type': 'proportional',
    'color': 'blue',
    'label': 'Proportional',
    }


fig, ax = plt.subplots(figsize=(10, 4))

ig, ax = plt.subplots(figsize=(10, 4))

plot_selection_results(ax, **softmax_params, **common_params)
plot_selection_results(ax, **truncation_params, **common_params)
plot_selection_results(ax, **proportional_params, **common_params)

ax.set_xlabel('Generation')
ax.set_ylabel('Population Score\nMean+IQR')
ax.set_title('Comparison of Selection Types over Generations')
ax.legend()

remove_ip_clutter(fig)
plt.show()