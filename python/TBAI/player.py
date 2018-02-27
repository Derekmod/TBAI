import random
import string

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