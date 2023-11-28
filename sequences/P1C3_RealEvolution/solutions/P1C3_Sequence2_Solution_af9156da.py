

def recombination_truncation_model(n_rcpt, n_gen):
  mu = n_rcpt*0.5
  mean_hist = np.zeros(n_gen)
  for generation in range(n_gen):
    mean_hist[generation] = mu
    f = mu/n_rcpt
    if f >= 1.0:
      delta_mu = 0
    else:
      delta_mu = np.sqrt(n_rcpt*f*(1-f)/2)
    mu += delta_mu
  return mean_hist


def compare_recomb_theory_sim(n_gen=500, n_rcpt=100, pop_size=1000, mutation_rate=None):
  if mutation_rate==None:
    mutation_rate = 1 / n_rcpt

  fig, ax = plt.subplots(figsize=(12, 8))

  # Theoretical recombination model
  theory_recomb1 = recombination_truncation_model(n_rcpt, n_gen)

  # Simulated recombination model
  simul_recomb_results = selection_type_simulation(
        pop_size=pop_size, n_gen=n_gen, n_rcpt=n_rcpt,
        has_mutation=False, has_recombination=True,
        selection_type='deterministic truncation', truncation_threshold=0.5,
        init_pop='recomb_eq')
  simul_recomb_mean_hist = simul_recomb_results[1]

  # Plotting
  ax.plot(theory_recomb1, color='blue', linestyle='--', label='Theory Recombination')
  ax.plot(simul_recomb_mean_hist, color='blue', label='Simulated Recombination')

  # Set titles and labels
  ax.set_title("Comparison of Theoretical and Simulated Recombination and Mutation")
  ax.set_xlabel("Generation")
  ax.set_ylabel("Mean Score")

  # Adjust legend
  ax.legend()
  plt.tight_layout()
  remove_ip_clutter(fig)
  plt.show()

compare_recomb_theory_sim(n_gen=30)