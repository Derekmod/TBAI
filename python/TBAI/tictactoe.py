from game_utils import *
import copy
import numpy as np
import sys

class TicTacToeGameState(GameState):
    def __init__(self):
        GameState.__init__(self)
        self._turn_order = [0, 1]

        self._position = [[-1]*3 for _ in range(3)]
        self.recalcCompressed()

    @property
    def moves(self):
        moves = []
        for row in range(3):
            for col in range(3):
                if self._position[row][col] < 0:
                    moves.append(TicTacToeMove((row, col), self.player_turn))

        return moves

    def enactMove(self, move):
        row, col = move.coords
        if self._position[row][col] >= 0:
            print('ERROR: location already filled')
            exit()

        new_state = copy.deepcopy(self)
        new_state._position[row][col] = self.player_turn
        new_state._turn_idx += 1
        new_state.recalcCompressed()
        return new_state

    def checkVictory(self):
        for col in range(3):
            items = []
            for row in range(3):
                items += [self._position[row][col]]
            victor = self._checkVictor(items)
            if victor >= 0:
                return victor

        for row in range(3):
            items = []
            for col in range(3):
                items += [self._position[row][col]]
            victor = self._checkVictor(items)
            if victor >= 0:
                return victor

        items = [self._position[0][0], self._position[1][1], self._position[2][2]]
        victor = self._checkVictor(items)
        if victor >= 0:
            return victor

        items = [self._position[2][0], self._position[1][1], self._position[0][2]]
        victor = self._checkVictor(items)
        if victor >= 0:
            return victor

        for row in range(3):
            for col in range(3):
                if self._position[row][col] < 0:
                    return -1

        return 0.5

    def _checkVictor(self, items):
        p = items[0]
        for v in items:
            if v != p:
                return -1
                break
            if v < 0:
                return -1
        return p

    def toString(self):
        return '\n'.join(['\t'.join([str(self._position[row][col]) for col in range(3)]) for row in range(3)])

    def features(self):
        flist = [self.player_turn]
        for row in range(3):
            for col in range(3):
                val = self._position[row][col]
                if val < 0:
                    flist += [0]
                elif val == 0:
                    flist += [-1]
                else:
                    flist += [1]
        return np.array(flist).astype(float)

    def recalcCompressed(self):
        features = self._position[0] + self._position[1] + self._position[2]
        #print(features

        self._compressed = min([sum([features[order[i]]*POWERS_OF_THREE[i] for i in range(9)]) for order in TIC_TAC_TOE_ORDERINGS])

    @property
    def compressed(self):
        return self._compressed

    def __hash__(self):
        return self._compressed


TIC_TAC_TOE_ORDERINGS = ((0, 1, 2, 3, 4, 5, 6, 7, 8),
                         (0, 3, 6, 1, 4, 7, 2, 5, 8),
                         (2, 5, 8, 1, 4, 7, 0, 3, 6),
                         (2, 1, 0, 5, 4, 3, 8, 7, 6),
                         (8, 7, 6, 5, 4, 3, 2, 1, 0),
                         (8, 5, 2, 7, 4, 1, 6, 3, 0),
                         (6, 7, 8, 3, 4, 5, 0, 1, 2),
                         (6, 3, 0, 7, 4, 1, 8, 5, 2))
POWERS_OF_THREE = (1, 3, 9, 27, 81, 243, 729, 2187, 6561)

class TicTacToeMove(object):
    def __init__(self, coords, player_key):
        self.coords = coords
        self.player_key = player_key

        
class HumanTicTacToePlayer(Player):
    def __init__(self, id_length=8, interpreter=None):
        Player.__init__(self, id_length)
        self._interpreter = interpreter

    def getMove(self, game_state):
        moves = game_state.moves
        ret = None

        while True:
            inp = sys.stdin.readline()
            row, col = [int(item) for item in inp.strip().split()]

            for move in moves:
                mrow, mcol = move.coords
                if mrow == row and mcol == col:
                    return move

            print('illegal move: (%d,%d)' % (row, col))


class TicTacToeGame(Game):
    def __init__(self):
        Game.__init__(self)
        #self.super()

        self._state = TicTacToeGameState()

    def start(self, display=True):
        if self.num_players < 2:
            print('ERROR: not enough players')
            return

        while self._state.checkVictory() < 0:
            if display:
                print('current victor: ', self._state.checkVictory())
                print('\n\n\n')
                print(self._state.toString())
            player = self._players[self._state.player_turn]
            move = player.getMove(self._state)
            print(move.coords)
            if self._verifyKey(move.player_key) or True: #TODO use key
                self._state = self._state.enactMove(move)

        victor = self._state.checkVictory()
        if display:
            if victor == 0.5:
                print('Tie!\n\n')
            else:
                print('Player %d won!\n\n' % victor)
            print(self._state.toString())

    def _verifyKey(self, key):
        true_key = self._players[self._state.player_turn]
        return true_key == key