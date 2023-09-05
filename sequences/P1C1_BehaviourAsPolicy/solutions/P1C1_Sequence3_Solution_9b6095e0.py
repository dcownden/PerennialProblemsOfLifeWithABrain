
class RandomValidMemoryPlayer():
  """
  Instantiate random player that uses memory for GridWorld
  """


  def __init__(self, game, critter_index=1):
    self.game = game
    self.critter_index = critter_index
    assert (isinstance(critter_index, int) and
        0 < critter_index <= game.num_critters), "Value is not a positive integer or exceeds the upper limit."
    self.last_locs = None


  def play(self, board):
    """
    Simulates a batch of random game plays based on the given board state.

    This function computes the probability of each valid move being played
    (uniform for valid moves, 0 for others), then selects a move randomly for
    each game in the batch based on these probabilities.

    Args:
      board (dict): A dictionary representing the state of the game. It
          contains:
          - 'pieces': A (batch_size, x_size, y_size) numpy array indicating
                      the pieces on the board.
          - 'scores' (not used directly in this function, but expected in dict)
          - 'rounds_left' (not used directly in this function, but expected in dict)

    Returns:
      tuple:
      - a (numpy array): An array of shape (batch_size,) containing randomly
                         chosen actions for each game in the batch.
      - a_1hots (numpy array): An array of shape (batch_size, action_size)
                               with one-hot encoded actions.
      - probs (numpy array): An array of the same shape as 'valids' containing
                             the probability of each move being played.
    """
    batch_size, n_rows, n_cols = board['pieces'].shape
    valids = self.game.get_valid_actions(board, self.critter_index)
    #invalidate old moves using memory of where critter last was
    if self.last_locs is not None:
      valids[(np.arange(batch_size), self.last_locs)] = 0
    probs = valids / np.sum(valids, axis=1).reshape(batch_size,1)
    action_size = n_rows * n_cols
    a = [self.game.rng.choice(action_size, p=probs[ii])
                                for ii in range(batch_size)]
    a_1hots = np.zeros((batch_size, action_size))
    a_1hots[(range(batch_size), a)] = 1.0
    # update memory
    rounds_left = board['rounds_left'][0]
    if rounds_left > 1:
      self.last_locs = self.game.moves_to_actions(
          np.where(board['pieces'] == self.critter_index ))
    else:
      # last move of the game reset memory for next game
      self.last_locs = None
    return np.array(a), a_1hots, probs

run_all_tests()