class GridworldBoard():
  """
  A collection methods and parameters of a gridworld game board that
  define the logic of the game.
  Board state is tracked as a triple (pieces, scores, rounds_left)
  pieces: batch_size x n_rows x n_cols np.array
  scores: batch_size np.array
  rounds_left: batch_size np.array

  Pieces are interpreted as:
  1=critter, -1=food, 0=empty

  First dim is batch, second dim row , third is col, so pieces[0][1][7]
  is the square in row 2, in column 8 of the first board in the batch of boards

  Note:
    In 2d np.array first dim is row (vertical), second dim is col (horizontal),
    i.e. top left corner is (0,0), so take care when visualizing/plotting
    as np.array visualization inline with typical tensor notation but at odds
    with conventional plotting where (0,0) is bottom left, first dim, x, is
    horizontal, second dim, y, is vertical
  """


  def __init__(self, batch_size=1,
               n_rows=7, n_cols=7,
               num_food=10, lifetime=30,
               rng = None):
    """Set the parameters of the game."""
    self.n_rows = n_rows
    self.n_cols = n_cols
    self.batch_size = batch_size
    self.num_food = num_food
    self.lifetime = lifetime
    if rng is None:
      self.rng = np.random.default_rng(seed=SEED)
    else:
      self.rng = rng


  def init_loc(self, n_rows, n_cols, num, rng=None):
    """
    Samples random 2d grid locations without replacement

    Args:
      n_rows: int, number of rows in the grid
      n_cols: int, number of columns in the grid
      num:    int, number of samples to generate. Should throw an error if num > n_rows x n_cols
      rng:    instance of numpy.random's default rng. Used for reproducibility.

    Returns:
      int_loc: ndarray(int) of shape (num,), flat indices for a 2D grid flattened into 1D
      rc_index: tuple(ndarray(int), ndarray(int)), a pair of arrays with the first giving 
        the row indices and the second giving the col indices. Useful for indexing into 
        an n_rows by n_cols numpy array.
      rc_plotting: ndarray(int) of shape (num, 2), 2D coordinates suitable for matplotlib plotting
    """

    # Set up default random generator, use the boards default if none explicitly given
    if rng is None:
      rng = self.rng
    # Choose 'num' unique random indices from a flat 1D array of size n_rows*n_cols
    int_loc = rng.choice(n_rows * n_cols, num, replace=False)
    # Convert the flat indices to 2D indices based on the original shape (n_rows, n_cols)
    rc_index = np.unravel_index(int_loc, (n_rows, n_cols))
    # Transpose indices to get num x 2 array for easy plotting with matplotlib
    rc_plotting = np.array(rc_index).T
    # Return 1D flat indices, 2D indices for numpy array indexing and 2D indices for plotting
    return int_loc, rc_index, rc_plotting


  def get_init_board_state(self):
    """Set up starting board using game parameters"""
    #set rounds_left and score
    self.rounds_left = np.ones(self.batch_size) * self.lifetime
    self.scores = np.zeros(self.batch_size)
    # create an empty board array.
    self.pieces = np.zeros((self.batch_size, self.n_rows, self.n_cols))
    # Place critter and initial food items on the board randomly
    for ii in np.arange(self.batch_size):
      # num_food+1 because we want critter and food locations
      int_loc, rc_idx, rc_plot = self.init_loc(
        self.n_rows, self.n_cols, self.num_food+1) 
      # critter random start location
      self.pieces[(ii, rc_idx[0][0], rc_idx[1][0])] = 1
      # food random start locations
      self.pieces[(ii, rc_idx[0][1:], rc_idx[1][1:])] = -1
    state = {'pieces': self.pieces.copy(),
             'scores': self.scores.copy(),
             'rounds_left': self.rounds_left.copy()}
    return state


  def set_state(self, board):
    """ board is a triple of np arrays
    pieces,       - batch_size x n_rows x n_cols
    scores,       - batch_size
    rounds_left   - batch_size
    """
    self.pieces = board['pieces'].copy()
    self.scores = board['scores'].copy()
    self.rounds_left = board['rounds_left'].copy()


  def get_state(self):
    """ returns a board state, which is a triple of np arrays
    pieces,       - batch_size x n_rows x n_cols
    scores,       - batch_size
    rounds_left   - batch_size
    """
    state = {'pieces': self.pieces.copy(),
             'scores': self.scores.copy(),
             'rounds_left': self.rounds_left.copy()}
    return state


  def __getitem__(self, index):
    return self.pieces[index]


  def execute_moves(self, moves):
    """
    Updates the state of the board given the moves made.

    Args:
      moves: a 3-tuple of 1-d arrays each of length batch_size,
        the first array gives the specific board within the batch,
        the second array in the tuple gives the new row coord for each critter
        on each board and the third gives the new col coord.

    Note:
      Assumes that there is exactly one valid move for each board in the
      batch of boards. i.e. it does't check for bounce/reflection on edges,
      or for multiple move made on the same board. It only checks for eating
      food and adds new food when appropriate. Invalid moves could lead to
      illegal teleporting behavior, critter dublication, or index out of range
      errors.
    This assumes the move is valid, i.e. doesn't check for
    bounce/reflection on edges, it only checks eating and adds new food,
    so invalid moves could lead to illegal teleporting behaviour or index out
    of range errors
    """
    #critters leave their spots
    self.pieces[self.pieces==1] = 0
    #which critters have food in their new spots
    eats_food = self.pieces[moves] == -1
    # some critters eat and their scores go up
    self.scores = self.scores + eats_food

    num_empty_after_eat = self.n_rows*self.n_cols - self.num_food
    # -1 for the critter +1 for food eaten
    # which boards in the batch had eating happen
    g_eating = np.where(eats_food)[0]
    if np.any(eats_food):
      # add random food to replace what is eaten
      possible_new_locs = np.where(np.logical_and(
          self.pieces == 0, #the spot is empty
          eats_food.reshape(self.batch_size, 1, 1))) #food eaten on that board
      food_sample_ = self.rng.choice(num_empty_after_eat,
                                     size=np.sum(eats_food))
      food_sample = food_sample_ + np.arange(len(g_eating))*num_empty_after_eat
      assert np.all(self.pieces[(possible_new_locs[0][food_sample],
                                 possible_new_locs[1][food_sample],
                                 possible_new_locs[2][food_sample])] == 0)
      #put new food on the board
      self.pieces[(possible_new_locs[0][food_sample],
                   possible_new_locs[1][food_sample],
                   possible_new_locs[2][food_sample])] = -1
    # put critters in new positions
    self.pieces[moves] = 1.0
    self.rounds_left = self.rounds_left - 1
    assert np.all(self.pieces.sum(axis=(1,2)) == ((self.num_food * -1) + 1))


  def get_legal_moves(self):
    """
    Identifies all legal moves for the critter, taking into acount
    bouncing/reflection at edges,

    Returns:
      A numpy int array of size batch x 3(g,x,y) x 4(possible moves)

    Note:
      moves[0,1,3] is the x coordinate of the move corresponding to the
      fourth offset on the first board.
      moves[1,:,1] will give the g,x,y triple corresponding to the
      move on the second board and the second offset, actions are integers
    """

    #apply all possible offsets to each game
    moves = np.stack([
      np.array(np.where(self.pieces == 1)) +
        np.array([np.array([0,  1, 0])]*self.batch_size).T,
      np.array(np.where(self.pieces == 1)) +
        np.array([np.array([0, -1, 0])]*self.batch_size).T,
      np.array(np.where(self.pieces == 1)) +
        np.array([np.array([0, 0,  1])]*self.batch_size).T,
      np.array(np.where(self.pieces == 1)) +
        np.array([np.array([0, 0, -1])]*self.batch_size).T]).swapaxes(0,2)

    #check bounces at boundaries
    moves[:,1,:] = np.where(moves[:,1,:] >=
                            self.n_rows, self.n_rows-2, moves[:,1,:])
    moves[:,2,:] = np.where(moves[:,2,:] >=
                            self.n_cols, self.n_cols-2, moves[:,2,:])
    moves[:,1,:] = np.where(moves[:,1,:] < 0, 1, moves[:,1,:])
    moves[:,2,:] = np.where(moves[:,2,:] < 0, 1, moves[:,2,:])
    return moves


  def get_perceptions(self, radius):
    """
    Generates a vector representation of of the critter perceptions, oriented
    around the critter. get_precept_filter is used to get a canonical version
    of the board with unknonw positions ocluded

    Args:
      radius: int, how many grid squared the critter can see around it
        using L1  (Manhattan/cityblock) distance

    Returns:
      A batch_size x 2*radius*(radius+1) + 1, giving the values
      of the percept reading left to right, top to bottom over the board,
      for each board in the batch
    """
    # define the L1 ball mask
    diameter = radius*2+1
    mask = np.zeros((diameter, diameter), dtype=bool)
    mask_coords = np.array([(i-radius, j-radius)
      for i in range(diameter)
        for j in range(diameter)])
    mask_distances = cdist(mask_coords, [[0, 0]],
                           'cityblock').reshape(mask.shape)
    mask[mask_distances <= radius] = True
    mask[radius,radius] = False  # exclude the center

    # pad the array
    padded_arr = np.pad(self.pieces, ((0, 0), (radius, radius),
     (radius, radius)), constant_values=-2)

    # get locations of critters
    critter_locs = np.argwhere(padded_arr == 1)

    percepts = []
    for critter_loc in critter_locs:
      b, r, c = critter_loc
      surrounding = padded_arr[b, r-radius:r+radius+1, c-radius:c+radius+1]
      percept = surrounding[mask]
      percepts.append(percept)
    return(np.array(percepts))