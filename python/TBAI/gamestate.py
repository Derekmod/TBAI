import numpy as np

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