gwg0 = GridworldGame(batch_size=1, n_cols=2, n_rows=2, num_food=0,
                     lifetime=30)
random_igwg_0 = InteractiveGridworld(gwg0, player=None, figsize=(5,4))
display(random_igwg_0.b_fig.canvas)
clear_output()
display(random_igwg_0.final_display)