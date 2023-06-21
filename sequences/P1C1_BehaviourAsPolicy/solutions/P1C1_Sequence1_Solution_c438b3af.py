

def init_loc(n_rows, n_cols, num):
  """
  Samples random 2d grid locations without replacement

  Args:
    n_rows: int
    n_cols: int
    num:    int, wnumber of samples to generate, should
            throw an error ifnum <= n_rows x n_cols

  Returns:
    int_loc: ndarray(int) of flat indices for a grid
    rc_index: (ndarray(int), ndarray(int)) a pair of arrays the first
      giving the row indices, the second giving the col indices, useful
      for indexing an n_rows by n_cols numpy array
    rc_plotting: ndarray(int) num x 2, same rc coordinates but structured
      in the way that matplotlib likes
  """
  int_loc = np.random.choice(n_rows * n_cols, num, replace=False)
  rc_index = np.unravel_index(int_loc, (n_rows, n_cols))
  rc_plotting = np.array(rc_index).T
  return int_loc, rc_index, rc_plotting


with plt.xkcd():
  fig, ax = make_grid(7, 7)
  int_locs, rc_index, rc_plotting = init_loc(7, 7, 11)

  rc_critter = (rc_plotting[0])
  plot_critter(fig, ax, rc_critter)

  rc_food = rc_plotting[1:]
  plot_food(fig, ax, rc_food)

  fig.legend(loc='outside right upper')