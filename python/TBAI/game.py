import random
import string


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