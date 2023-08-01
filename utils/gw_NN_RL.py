class dotdict(dict):
  def __getattr__(self, name):
    return self[name]


args = dotdict({
  'numIters': 1,            # In training, number of iterations = 1000 and num of episodes = 100
  'numEps': 1,              # Number of complete self-play games to simulate during a new iteration.
  'tempThreshold': 15,      # To control exploration and exploitation
  'updateThreshold': 0.6,   # During arena playoff, new neural net will be accepted if threshold or more of games are won.
  'maxlenOfQueue': 200,     # Number of game examples to train the neural networks.
  'numMCTSSims': 15,        # Number of games moves for MCTS to simulate.
  'arenaCompare': 10,       # Number of games to play during arena play to determine if new net will be accepted.
  'cpuct': 1,
  'maxDepth':5,             # Maximum number of rollouts
  'numMCsims': 5,           # Number of monte carlo simulations
  'mc_topk': 3,             # Top k actions for monte carlo rollout

  'checkpoint': './temp/',
  'load_model': False,
  'load_folder_file': ('/dev/models/8x100x50','best.pth.tar'),
  'numItersForTrainExamplesHistory': 20,

  # Define neural network arguments
  'lr': 0.001,               # lr: Learning Rate
  'dropout': 0.3,
  'epochs': 10,
  'batch_size': 64,
  'device': DEVICE,
  'num_channels': 512,
})




class GridWorldNNet(nn.Module):
  """
  Instantiate GridWorld Neural Net with following configuration
  nn.Conv2d(1, args.num_channels, 3, stride=1, padding=1) # Convolutional Layer 1
  nn.Conv2d(args.num_channels, args.num_channels, 3, stride=1, padding=1) # Convolutional Layer 2
  nn.Conv2d(args.num_channels, args.num_channels, 3, stride=1) # Convolutional Layer 3
  nn.Conv2d(args.num_channels, args.num_channels, 3, stride=1) # Convolutional Layer 4
  nn.BatchNorm2d(args.num_channels) X 4
  nn.Linear(args.num_channels * (self.board_x - 4) * (self.board_y - 4), 1024) # Fully-connected Layer 1
  nn.Linear(1024, 512) # Fully-connected Layer 2
  nn.Linear(512, self.action_size) # Fully-connected Layer 3
  nn.Linear(512, 1) # Fully-connected Layer 4
  """


  def __init__(self, game, args):
    """
    Initialise game parameters

    Args:
      game: GridWorld Game instance
        Instance of the GridWorldGame class above;
      args: dictionary
        Instantiates number of iterations and episodes, controls temperature threshold, queue length,
        arena, checkpointing, and neural network parameters:
        learning-rate: 0.001, dropout: 0.3, epochs: 10, batch_size: 64,
        num_channels: 512

    Returns:
      Nothing
    """
    self.board_x, self.board_y = game.get_board_size()
    self.action_size = game.get_action_size()
    self.args = args

    super(GridWorldNNet, self).__init__()
    self.conv1 = nn.Conv2d(1, args.num_channels, 3, stride=1, padding=1)
    self.conv2 = nn.Conv2d(args.num_channels, args.num_channels, 3, stride=1,
                           padding=1)
    self.conv3 = nn.Conv2d(args.num_channels, args.num_channels, 3, stride=1)
    self.conv4 = nn.Conv2d(args.num_channels, args.num_channels, 3, stride=1)

    self.bn1 = nn.BatchNorm2d(args.num_channels)
    self.bn2 = nn.BatchNorm2d(args.num_channels)
    self.bn3 = nn.BatchNorm2d(args.num_channels)
    self.bn4 = nn.BatchNorm2d(args.num_channels)

    self.fc1 = nn.Linear(args.num_channels * (self.board_x - 4) * (self.board_y - 4), 1024)
    self.fc_bn1 = nn.BatchNorm1d(1024)

    #figure out how to connect score and rounds left in here somewhere

    self.fc2 = nn.Linear(1024, 512)
    self.fc_bn2 = nn.BatchNorm1d(512)

    self.fc3 = nn.Linear(512, self.action_size)

    self.fc4 = nn.Linear(512, 1)


  def forward(self, s, currentScore, rounds_left):
    """
    Controls forward pass of GridWorldNNet

    Args:
      s: np.ndarray
        Array of size (batch_size x board_x x board_y)
      scoreRoundsContext: np.ndarray
        Array of size (batch_size x 2)
    Returns:
      Probability distribution over actions at the current state and the value of the current state.
    """
    s = s.view(-1, 1, self.board_x, self.board_y)                # batch_size x 1 x board_x x board_y
    rounds_left = rounds_left.view(-1, 1)                          # batch_siez x 1
    currentScore = currentScore.view(-1, 1)                          # batch_siez x 1

    s = F.relu(self.bn1(self.conv1(s)))                          # batch_size x num_channels x board_x x board_y
    s = F.relu(self.bn2(self.conv2(s)))                          # batch_size x num_channels x board_x x board_y
    s = F.relu(self.bn3(self.conv3(s)))                          # batch_size x num_channels x (board_x-2) x (board_y-2)
    s = F.relu(self.bn4(self.conv4(s)))                          # batch_size x num_channels x (board_x-4) x (board_y-4)
    s = s.view(-1, self.args.num_channels * (self.board_x - 4) * (self.board_y - 4))

    #need figure out how to put currentScore and rounds_left into the network here instead of and/or in addition to
    #finessing the value function at the end

    s = F.dropout(F.relu(self.fc_bn1(self.fc1(s))), p=self.args.dropout, training=self.training)  # batch_size x 1024
    s = F.dropout(F.relu(self.fc_bn2(self.fc2(s))), p=self.args.dropout, training=self.training)  # batch_size x 512

    pi = self.fc3(s)  # batch_size x action_size
    v = self.fc4(s)   # batch_size x 1 # the way this is structured now this is
                      # the average return per move, takes no account of rounds
                      # left though so kind of rough, but let's start with
                      # this for now
    #softmax_pi = F.softmax(pi, dim=1) # batch_size x action_size
    #v_pi_approx = torch.tensordot(softmax_pi, pi, dims=1) # batch_size x 1
    log_softmax_pi = F.log_softmax(pi, dim=1) # batch_size x action_size
    #corrected_pi = pi + (v - v_q_approx) # batch_size x num
    #scaled_pi = ...
    scaled_v = torch.add(torch.multiply(torch.add(torch.tanh(v), 1),
                                        rounds_left), currentScore)
    # Returns probability distribution over actions at the current state
    # and the value of the current state.
    return log_softmax_pi, scaled_v#, scaled_q




class PolicyValueNetwork():
  """
  Initiates the Policy-Value Network
  """


  def __init__(self, game, random_seed=None):
    """
    Initialise network parameters

    Args:
      game: GridWorld Game instance
        Instance of the GridWorldGame class above;

    Returns:
      Nothing
    """
    self.nnet = GridWorldNNet(game, args)
    self.board_x, self.board_y = game.get_board_size()
    self.action_size = game.get_action_size()
    self.nnet.to(args.device)
    self.rng = np.random.default_rng(seed=random_seed)


  def train(self, games, targetType='total',
            verbose=True, num_epochs=args.epochs):
    """
    Function to train network using just Value prediction loss

    Args:
      games: list
        List of examples with each example is of form (board, pi, v)
      targetType = 'total', 'value', 'policy'

    Returns:
      Nothing
    """
    optimizer = optim.Adam(self.nnet.parameters())
    print('training on a set of examples')
    for examples in games:
      for epoch in range(num_epochs):
        if verbose:
          print('EPOCH ::: ' + str(epoch + 1))
        self.nnet.train()
        v_losses = []   # To store the value losses per epoch
        pi_losses = []  # To store the policy losses per epoch
        t_losses = [] # To store the total losses per epoch
        batch_count = int(len(examples) / args.batch_size)  # e.g. len(examples)=200, batch_size=64, batch_count=3
        if verbose:
          t = tqdm(range(batch_count), desc='Training Value Network')
        else:
          t = range(batch_count)
        for _ in t:
          sample_ids = self.rng.integers(len(examples), size=args.batch_size)  # Read the ground truth information from MCTS examples
          boards, currentScores, rounds_lefts, pis, vs = list(zip(*[examples[i] for i in sample_ids]))  # Length of boards, pis, vis = 64
          boards = torch.FloatTensor(np.array(boards).astype(np.float64))
          currentScores = torch.FloatTensor(np.array(currentScores).astype(np.float64))
          rounds_lefts = torch.FloatTensor(np.array(rounds_lefts).astype(np.float64))
          target_pis = torch.FloatTensor(np.array(pis).astype(np.float64))
          target_vs = torch.FloatTensor(np.array(vs).reshape((-1, 1)).astype(np.float64)) # reshape to batch_size x 1 (not just batch_size) so can be treated the same as target pis

          # Predict
          # To run on GPU if available
          boards = boards.contiguous().to(args.device)
          currentScores = currentScores.contiguous().to(args.device)
          rounds_lefts = rounds_lefts.contiguous().to(args.device)
          target_pis = target_pis.contiguous().to(args.device)
          target_vs = target_vs.contiguous().to(args.device)

          # Compute output
          out_pi, out_v = self.nnet(boards, currentScores, rounds_lefts)
          #print(out_v.shape)
          #print(target_vs.shape)
          #print(out_pi.shape)
          #print(target_pis.shape)

          l_pi = self.loss_pi(target_pis, out_pi) # policy loss
          l_v = self.loss_v(target_vs, out_v)    # value loss
          l_total = torch.add(l_pi, l_v)        # total loss (no regularization term?!? or is that built in somewhere)

          # Record loss
          pi_losses.append(l_pi.item())
          v_losses.append(l_v.item())
          t_losses.append(l_total.item())
          if verbose:
            t.set_postfix(Loss_v=l_v.item(), Loss_pi=l_pi.item(), Loss_total=l_total.item())

          # Compute gradient and do SGD step
          optimizer.zero_grad()
          if targetType == 'total':
            l_total.backward()
          elif targetType == 'value':
            l_v.backward()
          elif targetType == 'policy':
            l_pi.backward()
          else:
            print('Invalid trainType chosen')
          optimizer.step()
        if verbose:
          print('v loss: ' + str(np.mean(v_losses)) +
                ' ::: pi loss: ' + str(np.mean(pi_losses)) +
                ' ::: total loss: ' + str(np.mean(t_losses)))
        else:
          if (epoch + 1) == args.epochs:
            print('Last Epoch Losses:')
            print('v loss: ' + str(np.mean(v_losses)) +
                  ' ::: pi loss: ' + str(np.mean(pi_losses)) +
                  ' ::: total loss: ' + str(np.mean(t_losses)))


  def predict(self, board, score, rounds_left):
    """
    Function to perform prediction of both policy and value, note
    policy is exponentiated on the way out so these should be directly
    interpretable as probabilities

    Args:
      board: batch x 7 x 7 np.ndarray giving board positions
      score: batch np.ndarray the current scores
      rounds_left: batch np.ndarray of the turns left

    Returns:
      pi: probabilities over actions
      v: predicted score at game end;
    """
    # Timing
    # start = time.time()

    # Preparing input
    board = torch.FloatTensor(board.astype(np.float64))
    board = board.contiguous().to(args.device)
    board = board.view(-1, self.board_x, self.board_y)

    score = torch.FloatTensor(np.array(score, dtype=np.float64))
    score = score.contiguous().to(args.device)
    score = score.view(-1, 1)

    rounds_left = torch.FloatTensor(np.array(rounds_left, dtype=np.float64))
    rounds_left = rounds_left.contiguous().to(args.device)
    rounds_left = rounds_left.view(-1, 1)

    self.nnet.eval()
    with torch.no_grad():
        pi, v = self.nnet(board, score, rounds_left)
    return torch.exp(pi).data.cpu().numpy(), v.data.cpu().numpy().flatten()


  def loss_v(self, targets, outputs):
    """
    Calculates Mean squared error
    Args:
      targets: np.ndarray
        Ground Truth end game scores corresponding to input board state
      outputs: np.ndarray
        value prediction of network as raw score

    Returns:
      MSE Loss calculated as: square of the difference between model predictions
      and the ground truth and averaged across the whole batch
    """
    # Mean squared error (MSE)
    return torch.sum((targets - outputs)**2) / targets.size()[0]


  def loss_pi(self, targets, outputs):
    """
    Calculates Negative Log Likelihood(NLL) of Targets
    Args:
      targets: np.ndarray
        Ground Truth action played during recording of "expert" player
      outputs: np.ndarray
        log-softmax action probability predictions of network

    Returns:
      Negative Log Likelihood calculated as:
    """
    return -torch.sum(targets * outputs) / targets.size()[0]


  def save_checkpoint(self, folder='checkpoint', filename='checkpoint.pth.tar'):
    """
    Code Checkpointing
    Args:
      folder: string
        Path specifying training examples
      filename: string
        File name of training examples
    Returns:
      Nothing
    """
    filepath = os.path.join(folder, filename)
    if not os.path.exists(folder):
      print("Checkpoint Directory does not exist! Making directory {}".format(folder))
      os.mkdir(folder)
    else:
      print("Checkpoint Directory exists! ")
    torch.save({'state_dict': self.nnet.state_dict(),}, filepath)
    print("Model saved! ")


  def load_checkpoint(self, folder='checkpoint', filename='checkpoint.pth.tar'):
    """
    Load code checkpoint
    Args:
      folder: string
        Path specifying training examples
      filename: string
        File name of training examples
    Returns:
      Nothing
    """
    # https://github.com/pytorch/examples/blob/master/imagenet/main.py#L98
    filepath = os.path.join(folder, filename)
    if not os.path.exists(filepath):
      raise ("No model in path {}".format(filepath))

    checkpoint = torch.load(filepath, map_location=args.device)
    self.nnet.load_state_dict(checkpoint['state_dict'])




class MonteCarlo():
  """
  Implementation of Monte Carlo Algorithm
  """


  def __init__(self, game, nnet, default_depth=5, random_seed=None):
    """
    Initialize Monte Carlo Parameters

    Args:
      game: Gridworld Game instance
        Instance of the gridworldGame class above;
      nnet: gridworldNet instance
        Instance of the gridworldNNet class above;
      args: dictionary
        Instantiates number of iterations and episodes, controls temperature threshold, queue length,
        arena, checkpointing, and neural network parameters:
        learning-rate: 0.001, dropout: 0.3, epochs: 10, batch_size: 64,
        num_channels: 512

    Returns:
      Nothing
    """
    self.game = game
    self.nnet = nnet
    self.default_depth = default_depth
    self.rng = np.random.default_rng(seed=random_seed)


  def simulate(self, board, actions, action_indexes, depth=None):
    """
    Helper function to simulate one Monte Carlo rollout

    Args:
      board: triple (batch_size x x_size x y_size np.array of board position,
                     scalar of current score,
                     scalar of rounds left
      actions: batch size list/array of integer indexes for moves on each board
      these are assumed to be legal, no check for validity of moves
    Returns:
      temp_v:
        Terminal State
    """
    batch_size, x_size, y_size = board['pieces'].shape
    next_board = self.game.get_next_state(board, actions, action_indexes)
    if depth is None:
      depth = self.default_depth
    # potentially expand the game tree here,
    # but just do straight rollouts after this
    # doesn't expand to deal with all random food generation possibilities
    # just expands based on the actions given
    expand_bs, _, _ = next_board['pieces'].shape

    for i in range(depth):  # maxDepth
      if next_board['rounds_left'][0] <= 0:
        # check that game isn't over
        # assumes all boards have the same rounds left
        # no rounds left return scores as true values
        terminal_vs = next_board['scores'].copy()
        return terminal_vs
      else:
        pis, vs = self.nnet.predict(next_board['pieces'], next_board['scores'], next_board['rounds_left'])
        valids = self.game.get_valid_actions(next_board)
        masked_pis = pis * valids
        sum_pis = np.sum(masked_pis, axis=1)
        probs = np.array(
            [masked_pi / masked_pi.sum() if masked_pi.sum() > 0
             else valid / valid.sum()
             for valid, masked_pi in zip(valids, masked_pis)])
        samp = self.rng.uniform(size = expand_bs).reshape((expand_bs,1))
        sampled_actions = np.argmax(probs.cumsum(axis=1) > samp, axis=1)
      next_board = self.game.get_next_state(next_board, sampled_actions)

    pis, vs = self.nnet.predict(next_board['pieces'], next_board['scores'], next_board['rounds_left'])
    return vs




class MonteCarloBasedPlayer():
  """
  Simulate Player based on Monte Carlo Algorithm
  """

  def __init__(self, game, nnet,
               default_depth = 1,
               default_rollouts = 1,
               default_K = 4,
               default_temp = 1.0,
               random_seed=None):
    """
    Initialize Monte Carlo Parameters

    Args:
      game: Gridworld Game instance
        Instance of the gridworldGame class above;
      nnet: gridworldNet instance
        Instance of the gridworldNNet class above;
      args: dictionary
        Instantiates number of iterations and episodes, controls temperature threshold, queue length,
        arena, checkpointing, and neural network parameters:
        learning-rate: 0.001, dropout: 0.3, epochs: 10, batch_size: 64,
        num_channels: 512

    Returns:
      Nothing
    """
    self.game = game
    self.nnet = nnet
    self.default_depth = default_depth
    self.default_rollouts = default_rollouts
    self.mc = MonteCarlo(self.game, self.nnet, self.default_depth)
    self.default_K = default_K
    self.default_temp = default_temp
    self.rng = np.random.default_rng(seed=random_seed)


  def play(self, board,
           num_rollouts=None,
           rollout_depth=None,
           K=None,
           softmax_temp=None):
    """
    Simulate Play on a Board

    Args:
      board: triple (batch x num_rows x num_cols np.ndarray of board position,
                     batch x a of current score,
                     batch x 1 of rounds left

    Returns:
      best_action: tuple
        (avg_value, action) i.e., Average value associated with corresponding action
        i.e., Action with the highest topK probability
    """
    batch_size, n_rows, n_cols = board['pieces'].shape
    if num_rollouts is None:
      num_rollouts = self.default_rollouts
    if rollout_depth is None:
      rollout_depth = self.default_depth
    if K is None:
      K = self.default_K
    if softmax_temp is None:
      softmax_temp = self.default_temp

    # figure out top k actions according to normalize action probability
    # given by our policy network prediction
    pis, vs = self.nnet.predict(board['pieces'], board['scores'], board['rounds_left'])
    valids = self.game.get_valid_actions(board)
    masked_pis = pis * valids  # Masking invalid moves
    sum_pis = np.sum(masked_pis, axis=1)
    num_valid_actions = np.sum(valids, axis=1)
    effective_topk = np.array(np.minimum(num_valid_actions, K), dtype= int)
    probs = np.array([masked_pi / masked_pi.sum() if masked_pi.sum() > 0
                      else valid / valid.sum()
                      for valid, masked_pi in zip(valids, masked_pis)])
    partioned = np.argpartition(probs,-effective_topk)
    topk_actions = [partioned[g,-(ii+1)]
                      for g in range(batch_size)
                        for ii in range(effective_topk[g])]
    topk_actions_index = [ii
                            for ii, etk in enumerate(effective_topk)
                              for _ in range(etk)]
    values = np.zeros(len(topk_actions))
    # Do some rollouts
    for _ in range(num_rollouts):
      values = values + self.mc.simulate(board, topk_actions,
                                         topk_actions_index,
                                         depth=rollout_depth)
    values = values / num_rollouts

    value_expand = np.zeros((batch_size, n_rows*n_cols))
    value_expand[(topk_actions_index, topk_actions)] = values
    value_expand_shift = value_expand - np.max(value_expand, axis=1, keepdims=True)
    value_expand_scale = value_expand_shift/softmax_temp
    v_probs = np.exp(value_expand_scale) / np.sum(
        np.exp(value_expand_scale), axis=1, keepdims=True)
    v_probs = v_probs * valids
    v_probs = v_probs / np.sum(v_probs, axis=1, keepdims=True)
    samp = self.rng.uniform(size = batch_size).reshape((batch_size,1))
    sampled_actions = np.argmax(v_probs.cumsum(axis=1) > samp, axis=1)
    a_1Hots = np.zeros((batch_size, n_rows*n_cols))
    a_1Hots[(range(batch_size), sampled_actions)] = 1.0
    return sampled_actions, a_1Hots, v_probs