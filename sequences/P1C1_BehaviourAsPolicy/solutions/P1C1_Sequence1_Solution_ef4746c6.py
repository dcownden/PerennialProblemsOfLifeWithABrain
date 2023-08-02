

def init_loc(n_rows, n_cols, num, rng=None):
  """
  Samples random 2d grid locations without replacement

  Args:
    n_rows: int, number of rows in the grid
    n_cols: int, number of columns in the grid
    num:    int, number of samples to generate. Should throw an error if num > n_rows x n_cols
    rng:    instance of numpy.random's default rng. Used for reproducibility.

  Returns:
    int_loc: ndarray(int) of shape (num,), flat indices for a 2D grid flattened into 1D
    rc_index: tuple(ndarray(int), ndarray(int)), a pair of arrays with the first giving
      the row indices and the second giving the col indices. Useful for indexing into
      an n_rows by n_cols numpy array.
    rc_plotting: ndarray(int) of shape (num, 2), 2D coordinates suitable for matplotlib plotting
  """
  # If no random number generator given, make one using predefined global SEED
  if rng is None:
    rng = np.random.default_rng(seed=SEED)
  # Choose 'num' unique random indices from a flat
  # 1D array of size n_rows*n_cols
  int_loc = rng.choice(n_rows * n_cols, num, replace=False)
  # Convert flat indices to 2D indices based on shape (n_rows, n_cols)
  rc_index = np.unravel_index(int_loc, (n_rows, n_cols))
  # Transpose indices to get num x 2 array for easy plotting with matplotlib
  rc_plotting = np.array(rc_index).T
  # Return 1D flat indices, 2D indices for numpy array indexing
  # and 2D indices for plotting
  return int_loc, rc_index, rc_plotting

# Set the drawing style to 'xkcd'
with plt.xkcd():
  # Create a grid for the plot
  fig, ax = make_grid(7, 7)
  # Generate 11 unique locations on the grid
  int_locs, rc_index, rc_plotting = init_loc(7, 7, 11)
  # The first location is for the "critter"
  rc_critter = rc_plotting[0]
  plot_critter(fig, ax, rc_critter)
  # Remaining locations are for "food"
  rc_food = rc_plotting[1:]
  plot_food(fig, ax, rc_food)
  # Add legend outside the upper right corner
  fig.legend(loc='outside right upper')
  plt.show()