

def evaluate(flat_W, game):
  # Run the game with the proposed weights
  W = flat_W.reshape((4,12))
  boppp = BatchOptPerceptParamPlayer(game, weights=W, deterministic=False)
  final_board = game.play_game(players=[boppp], visualize=False)
  # Evaluate the score
  scores = final_board['scores'].flatten()
  avg_score = np.mean(scores)
  return(avg_score)


no_drift_game = GridworldGame(batch_size=900, n_rows=7, n_cols=7,
                              num_critters=1, num_food=10, lifetime=30,
                              rng=np.random.default_rng(48),
                              drift_probs=None)
drift_game = GridworldGame(batch_size=900, n_rows=7, n_cols=7,
                           num_critters=1, num_food=10, lifetime=30,
                           rng=np.random.default_rng(48),
                           drift_probs=[0, 0, 1.0, 0, 0.0],
                           wrapping=True, drift_after_move=False)
try:
  # try use the params we found earlier
  good_no_drift_params = best_params
except NameError:
  # Use some 'canned' parameters if skipping ahead
  good_no_drift_params = np.array([
    72.44502999, -10.1196398,   75.58460857,   6.30015161,  22.72267252,
   -21.87763734, -25.29124192, -37.70261395, -32.06002976, -37.55333989,
     3.05150248, -10.64373529,  -6.70861488, -25.30559016, -56.576501,
    16.30638079,  10.80652839,  -0.85537565, -61.01324566, -37.03868886,
    21.13524191,  18.40083665, -25.36843063,  43.14481738,  29.39306285,
   -24.54420009,  -2.0335233,   -0.31976222,  40.61494124,  57.30685941,
   -34.73713881, -30.63781385,   8.45863857, -32.02285974,  -9.81967995,
    19.31799882,  -5.87473403, -10.68178413, -54.75878878,  21.34389051,
     6.64222051, -71.77728684,  11.06174306,  12.94640783, -13.26093112,
   -23.2402532,    9.96075751,  25.32552931])
try:
  # try use the params we found earlier
  good_drift_params = best_drift_params
except NameError:
  # Use some 'canned' parameters if skipping ahead
  good_drift_params = np.array([
         37.89629402,  -48.2903797 ,  104.57227396,   82.96216298,
        -33.72075405,  -11.60406537,  -13.81356909,  -51.3297726 ,
        -88.41182722,  -40.9836382 ,  -11.69845196,   26.95174635,
         14.99113154,  -42.84500512,  -31.22642526,    0.8173807 ,
         14.84325766,    5.03038252, -103.05269519,  -28.56967293,
         51.6487887 ,   73.90697663,   18.44399746,   76.73473352,
         40.74093128,  -42.35290311,   45.31994088,  -31.84604343,
          8.83368266,   81.59865548,   -5.40502478,  -72.75935279,
         46.0222178 ,  -64.32899446,   -1.39618574,   45.16903944,
          2.57398304,  -65.45567314,  -24.56553091,    0.86333044,
         45.63555023,  -56.93741837,   10.23266167,  -13.28302046,
         29.150387  ,  -39.25495485,  -31.43520672,   31.85720807])
data = {
  "no_drift_game": [evaluate(good_no_drift_params, no_drift_game),
                    evaluate(good_drift_params, no_drift_game)],
  "drift_game": [evaluate(good_no_drift_params, drift_game),
                 evaluate(good_drift_params, drift_game)]
}
df = pd.DataFrame.from_dict(data, orient='index',
                            columns=["good_no_drift_params", "good_drift_params"])
display(df)