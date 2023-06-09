def init_loc(x_size, y_size, num):
  """Returns random 2d grid locations, without replacement,
  in both unravled/flat coordinates and as xy pairs"""
  int_loc = np.random.choice(x_size * y_size, num, replace=False)
  xy_loc = np.vstack(np.unravel_index(int_loc, (x_size, y_size))).T
  return int_loc, xy_loc

fig, ax = make_grid(7, 7)
int_food_loc, xy_food_loc = init_loc(7,7,10)
plot_food(fig, ax, xy_food_loc)

xy_critter_loc = (5,5)
plot_critter(fig, ax, xy_critter_loc)

fig.legend(loc='outside right upper')
plt.show()