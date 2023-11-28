

def mutation_truncation_model(n_rcpt, n_gen, mutation_rate, mu_0 = 0):
  mu = mu_0
  mean_hist = np.zeros(n_gen)
  for generation in range(n_gen):
    mean_hist[generation] = mu + n_rcpt/2 #convert delta to z for plotting
    delta_mu = np.sqrt(n_rcpt*mutation_rate) - 2*mutation_rate*mu
    mu += delta_mu
  return mean_hist


def compare_mutation_theory_sim(per_geno_mutation_rates, n_gen=200, n_rcpt=100, pop_size=1000):
  mutation_rates = np.array(per_geno_mutation_rates) / n_rcpt
  fig, ax = plt.subplots(figsize=(10, 6))

  num_shades = len(mutation_rates)
  base_color_map = matplotlib.colormaps['Reds']
  colors = [base_color_map((i+1)/(num_shades+1)) for i in range(num_shades)]
  custom_lines = [matplotlib.lines.Line2D([0], [0], color='black', linestyle='-'),
                  matplotlib.lines.Line2D([0], [0], color='black', linestyle='--')]

  for mutation_rate, color in zip(mutation_rates, colors):
    # Run the theoretical model
    theoretical_mean_hist = mutation_truncation_model(n_rcpt, n_gen, mutation_rate)
    # Run the simulation model
    simulation_results = selection_type_simulation(pop_size=pop_size, n_gen=n_gen,
                                                   n_rcpt=n_rcpt, mutation_rate=mutation_rate,
                                                   selection_type='deterministic truncation',
                                                   truncation_threshold=0.5,
                                                   init_pop='mutation_eq')
    simulated_mean_hist = simulation_results[1]
    # Plotting
    ax.plot(theoretical_mean_hist, color=color, linestyle='--', label=f"Theory (Mutation rate: {mutation_rate})")
    ax.plot(simulated_mean_hist, color=color, label=f"Simulation (Mutation rate: {mutation_rate})")

  # Set titles and labels
  ax.set_title("Comparison of Theoretical and Simulated Models Across Mutation Rates")
  ax.set_xlabel("Generation")
  ax.set_ylabel("Mean Score")
  ax.legend()
  remove_ip_clutter(fig)
  plt.tight_layout()
  plt.show()

per_geno_mutation_rates = [2.0, 0.5, 0.05]
compare_mutation_theory_sim(per_geno_mutation_rates, n_gen=500)