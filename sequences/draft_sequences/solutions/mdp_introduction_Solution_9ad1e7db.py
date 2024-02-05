
class SimpleThresholdPlayer():
  """
  Player that moves in a sweeping pattern and decides whether to forages
  or move to a new patch based on threshold number of failed foraging attempts.
  """

  def __init__(self, game, critter_index=1, threshold_new=1, threshold_known=2,
               return_direction=True):
    self.game = game
    self.critter_index = critter_index
    self.threshold_new = threshold_new
    self.threshold_known = threshold_known
    self.last_direction = ['right'] * self.game.batch_size
    self.return_direction = return_direction

  def play(self, board):
    batch_size = board['pieces'].shape[0]
    chosen_directions = []

    valid_directions_batch = self.game.get_valid_directions(board, self.critter_index)

    for i in range(batch_size):
      valid_directions_for_this_board = valid_directions_batch[i]
      is_at_new_patch = board['at_new_patch'][i, self.critter_index - 1]
      # Decide to forage or move
      if is_at_new_patch:
        # print('at new patch')
        misses = board['misses_new_patch'][i, self.critter_index - 1]
        # print('misses:', misses)
        threshold = self.threshold_new
      else:
        # print('at known patch')
        misses = board['misses_known_patch'][i, self.critter_index - 1]
        # print('misses:', misses)
        threshold = self.threshold_known

      if misses >= threshold:
        # print('move to new patch')
        chosen_directions.append(self._get_next_direction(i, valid_directions_for_this_board))
      else:
        # print('forage at current patch')
        chosen_directions.append('still')

    if self.return_direction:
      return chosen_directions
    else:
      # Convert chosen directions to actions
      actions = self.game.critter_directions_to_actions(board, chosen_directions, self.critter_index)
      action_size = self.game.get_action_size()
      a_1hots = np.zeros((batch_size, action_size))
      a_1hots[np.arange(batch_size), actions] = 1.0
      return actions, a_1hots, a_1hots

  def _get_next_direction(self, board_idx, valid_directions_for_this_board):
    """
    Get the next direction based on left-right-down sweeping pattern.
    """
    if self.last_direction[board_idx] == 'right':
      if 'right' in valid_directions_for_this_board:
        return 'right'
      elif 'down' in valid_directions_for_this_board:
        self.last_direction[board_idx] = 'left'
        return 'down'

    if self.last_direction[board_idx] == 'left':
      if 'left' in valid_directions_for_this_board:
        return 'left'
      elif 'down' in valid_directions_for_this_board:
        self.last_direction[board_idx] = 'right'
        return 'down'

    return 'still'  # Default to 'still' if none of the conditions are met