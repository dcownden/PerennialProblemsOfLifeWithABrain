
display(output)

# 20 random moves, one every ~0.1 seconds
state = init_state(x_size=3, y_size=2, num_food=5)
for ii in range(30):
  random_click()
  time.sleep(0.1)