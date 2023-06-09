
display(output)

# 20 random button presses, one every ~0.1 seconds
state = init_state(x_size=7, y_size=7, num_food=0)
for ii in range(30):
  random_click()
  time.sleep(0.1)