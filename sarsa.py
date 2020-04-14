import numpy as np

class StateActionFeatureVectorWithTile():
    def __init__(self,
                 state_low:np.array,
                 state_high:np.array,
                 num_actions:int,
                 num_tilings:int,
                 tile_width:np.array):
        """
        state_low: possible minimum value for each dimension in state
        state_high: possible maximum value for each dimension in state
        num_actions: the number of possible actions
        num_tilings: # tilings
        tile_width: tile width for each dimension
        """
        self.low = state_low
        self.high = state_high
        self.num_tilings = num_tilings
        self.num_actions = num_actions
        self.tile_width = tile_width

        # num tiles in each dimension
        self.num_tiles = (np.ceil((state_high - state_low) / tile_width) + 1).astype(int)

        # number of tiles in state space
        self.ttl_tiles = self.num_tiles.prod()

        # map tiling to combined tiles in dimensions
        self.tiles = np.zeros(shape=(num_actions, num_tilings, self.ttl_tiles), dtype=float)

        # offset of each tiling
        self.starts = [state_low - i / num_tilings * tile_width for i in range(num_tilings)]


    def feature_vector_len(self) -> int:
        """
        return dimension of feature_vector: d = num_actions * num_tilings * num_tiles
        """
        return self.num_actions * self.num_tilings * self.ttl_tiles

    def featureExtractor(self, s, a):
        """
        Creates binary mask for activated tiles given the state
        :param s: state
        :return: binary mask of active tiles
        """

        features = np.zeros(shape=(self.num_actions, self.num_tilings, self.ttl_tiles), dtype=int)

        for i in range(self.num_tilings):
            offset = (s - self.starts[i]) // self.tile_width
            tile = int(self.num_tiles[0] * offset[0] + offset[1])
            features[a, i, tile] = 1
        return features


    def update(self, w):
        """
        Update tile weights with weight vector
        :param w: weight vector
        """
        w = w.reshape((self.num_actions, self.num_tilings, self.ttl_tiles))
        self.tiles += w.reshape((self.num_actions, self.num_tilings, self.ttl_tiles))

    def __call__(self, s, done, a) -> np.array:
        """
        implement function x: S+ x A -> [0,1]^d
        if done is True, then return 0^d
        """

        if not done:
            fi = self.featureExtractor(s, a).flatten()

        else:
            fi = np.zeros(self.feature_vector_len())

        return fi

def SarsaLambda(
    env, # openai gym environment
    gamma:float, # discount factor
    lam:float, # decay rate
    alpha:float, # step size
    X:StateActionFeatureVectorWithTile,
    num_episode:int,
) -> np.array:
    """
    Implement True online Sarsa(\lambda)
    """

    def epsilon_greedy_policy(s, done, w, epsilon=0.05):
        nA = env.action_space.n

        Q = [np.dot(w, X(s, done, a)) for a in range(nA)]

        if np.random.rand() < epsilon:
            return np.random.randint(nA)
        else:
            return np.argmax(Q)

    w = np.zeros((X.feature_vector_len()))

    # for each episode
    for ep in range(num_episode):

        # initialize S
        obs = env.reset()

        done = False

        # choose an action A ~ pi(|S) or near greadily from S using w
        action = epsilon_greedy_policy(obs, done, w)

        # retrieve feature vector
        x = X(obs, done, action)

        # init eligibility trace
        z = np.zeros(len(x))

        qOld = 0

        # until S is terminal
        while not done:

            # take action and observe R and St+1
            s1, reward, done, info = env.step(action)

            # choose an action A' ~ pi(|S') or near greadily from S using w
            a1 = epsilon_greedy_policy(s1, done, w)
            x1 = X(s1, done, a1)

            q = (w * x).sum()
            qPrime = (w * x1).sum()

            # compute TD error for state-action value
            tderr = reward + (gamma * qPrime - q)

            # compute eligibility trace
            z = gamma * lam * z + (1 - alpha * gamma * lam * z * x) * x

            # update weights
            w = w + alpha * (tderr + q - qOld) * z - alpha * (q - qOld) * x

            qOld = qPrime
            x = x1
            action = a1

            # update tile weights
            X.update(w)

    return w