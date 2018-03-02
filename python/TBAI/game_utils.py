import random
import string
import numpy as np


class Player(object):
    def __init__(self, id_length=8, data=None):
        self._id = ''.join(random.SystemRandom().choice(string.ascii_uppercase + string.ascii_lowercase + string.digits)
                           for _ in range(id_length))
        
        self._data = data
        if self._data is None:
            self._data = {}

    def getMove(self, game_state):
        moves = game_state.moves
        return random.choice(moves)

    @property
    def id(self):
        return self._id

    @property
    def data(self):
        return self._data


class GameState(object):
    def __init__(self):
        self._turn_order = [0, 1]
        self._turn_idx = 0

    @property
    def player_turn(self):
        return self._turn_order[self._turn_idx % len(self._turn_order)]

    @property
    def moves(self):
        return []

    def enactMove(self, move):
        '''
        Returns a new state.
        '''
        return None

    def checkVictory(self):
        return -1
    
    def features(self):
        return np.array([])

    @property
    def compressed(self):
        return self

    def __hash__(self):
        return 0


class Game(object):
    def __init__(self):
        self._player_keys = []
        self._players = []
        self._player_dict = dict()

    def registerPlayer(self, player, key_length=12):
        self._players += [player]
        key = ''.join(random.SystemRandom().choice(string.ascii_uppercase + string.ascii_lowercase + string.digits)
                           for _ in range(key_length))
        self._player_dict[key] = player
        return key

    def getState(self, player_key):
        pass

    @property
    def num_players(self):
        return len(self._players)