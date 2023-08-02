gwg100 = GridworldGame(batch_size=1, n_cols=2, n_rows=2, num_food=3,
                       lifetime=30)
random_igwg_100 = InteractiveGridworld(gwg100, player=None, figsize=(5,4))
display(random_igwg_100.b_fig.canvas)
clear_output()
display(random_igwg_100.final_display)