def init_loc(x_size, y_size, num):
  """Returns random 2d grid locations, without replacement,
  in both unravled/flat coordinates and as xy pairs"""
  int_loc = np.random.choice(x_size * y_size, num, replace=False)
  xy_loc = np.vstack(np.unravel_index(int_loc, (x_size, y_size))).T
  return int_loc, xy_loc

with plt.xkcd():
  fig, ax = make_grid(7, 7)
  int_locs, xy_locs = init_loc(7, 7, 11)

  xy_critter_loc = xy_locs[0]
  plot_critter(fig, ax, xy_critter_loc)

  xy_food_loc = xy_locs[1:]
  plot_food(fig, ax, xy_food_loc)

  fig.legend(loc='outside right upper')
  plt.show()