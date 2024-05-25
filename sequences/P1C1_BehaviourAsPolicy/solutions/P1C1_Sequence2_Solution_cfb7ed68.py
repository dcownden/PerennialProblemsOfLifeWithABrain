
# Calculate the probability of eating given the amount of food perceived
# for food amounts with non-zero observations
non_zero_cols = np.where(data.sum(axis=0) > 0)[0]
plot_data = data[:, non_zero_cols]
prob_eating = plot_data[1,:] / plot_data.sum(axis=0)

# Create a figure and axis for the plot
with plt.xkcd():
  fig_prop, ax_prop = plt.subplots()
  # Plot the probabilities
  ax_prop.plot(np.arange(len(data[0]))[non_zero_cols], prob_eating, marker='o')
  # Set the title and labels
  ax_prop.set_title('Proportion of Eating Events by\nAmount of Food Perceived Prior')
  ax_prop.set_xlabel('Number of Food Items Perceived')
  ax_prop.set_ylabel('Proportion where Eating Follows')
  remove_ip_clutter(fig_prop)
  fig_prop.tight_layout()
  # Display the plot
  plt.show()