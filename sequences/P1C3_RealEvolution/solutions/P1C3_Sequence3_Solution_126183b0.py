gwg = GridworldGame(batch_size=1, num_prey=1, num_pred=1)
prey_player_params = {'critter_index': 1, 'fov_radius': 2, 'speed': 2,
                      'has_food_percept': True, 'has_edge_percept': True,
                      'has_prey_percept': True, 'has_pred_percept': True}
prey_player = GeneralLinearPlayer(gwg, **prey_player_params)

pred_player_params = {'critter_index': 2, 'fov_radius': 3, 'speed': 1,
                      'has_food_percept': True, 'has_edge_percept': True,
                      'has_prey_percept': True, 'has_pred_percept': True}
pred_player = GeneralLinearPlayer(gwg, **pred_player_params)