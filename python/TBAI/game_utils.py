import random
import string
import copy
import torch


class Player(object):
    '''Plays turn-base games.'''
    def __init__(self, data=None):        
        self._data = data
        if self._data is None:
            self._data = {}

    def getMove(self, game_state):
        '''Primary method - makes a move.
        Args:
            game_state <GameState>: current state
        Returns:
            <Move>: move the player would like to enact
        '''
        moves = game_state.moves
        return random.choice(moves)

    @property
    def data(self):
        return self._data


class GameState(object):
    '''Encodes the current state of a game. '''
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
        '''Returns a new state.'''
        new_state = copy.deepcopy(self)
        new_state._turn_idx += 1
        return new_state

    def checkVictory(self):
        '''Checks if game is over.
        Returns:
            winner (0 or 1) if there is one
            0.5 if tie
            -1 if game ongoing
        '''
        return -1
    
    def features(self):
        ''' PyTorch interpretable features of the game state.
        Returns:
            [FloatTensor]: suggested form:
                [public state,] simple globals,] hidden_info]
        '''
        return [torch.FloatTensor([]), torch.FloatTensor([self.player_turn, self._turn_idx])]

    @property
    def compressed(self):
        return self

    def __hash__(self):
        return 0


class Game(object):
    '''Responsible for faithfully running a Game.'''
    def __init__(self):
        self._player_keys = []
        self._players = []
        self._player_dict = dict()

        self._state = GameState()

    def registerPlayer(self, player, key_length=12):
        '''Adds 'player' to game, in order
        Args:
            player <Player>: player to add
            key_length <int>: access key for the player (for limited information games)
        '''
        self._players += [player]
        key = ''.join(random.SystemRandom().choice(string.ascii_uppercase + string.ascii_lowercase + string.digits)
                           for _ in range(key_length))
        self._player_dict[key] = player
        return key

    def getState(self, player_key=None):
        return self._state

    @property
    def num_players(self):
        return len(self._players)

    def start(self, display=True):
        '''Play the game until completion.'''
        if self.num_players < 2:
            print('ERROR: not enough players')
            return

        while self._state.checkVictory() < 0:
            self.turn(display)

        victor = self._state.checkVictory()
        if display:
            if victor == 0.5:
                print('Tie!\n\n')
            else:
                print('Player %d won!\n\n' % victor)
            print(self._state.toString())

    def turn(self, display=True):
        '''Play one turn (without safety checks).'''
        if display:
            print('\n\n\n')
            print(self._state.toString())
            print('Player %d turn:' % self._state.player_turn)
        player = self._players[self._state.player_turn]
        move = player.getMove(self._state)
        self._state = self._state.enactMove(move)