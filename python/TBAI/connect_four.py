from game_utils import *
import copy
import numpy as np
import sys

import torch
from torch.autograd import Variable


CONNECT_FOUR_ROWS = 6
CONNECT_FOUR_COLS = 7


class ConnectFourGameState(GameState):
    def __init__(self):
        GameState.__init__(self)
        self._turn_order = [0, 1]

        self._position = [[] for _ in range(7)]
        self._recalcCompressed()

    @property
    def moves(self):
        moves = []
        for col in range(CONNECT_FOUR_COLS):
            if len(self._position[col]) < CONNECT_FOUR_ROWS:
                moves += [col]

        return moves

    def enactMove(self, move):
        col = move
        if len(self._position[col]) >= CONNECT_FOUR_ROWS:
            print('ERROR: location already filled')
            exit()

        new_state = copy.deepcopy(self)
        new_state._position[col] += [self.player_turn]
        new_state._turn_idx += 1
        new_state._recalcCompressed()
        return new_state

    def checkVictory(self):
        for col in range(CONNECT_FOUR_COLS):
            for row in range(len(self._position[col])):
                for dx, dy in ((-1,1), (0,1), (1,1), (1,0)):
                    #print('reset vals, (dx,dy) = (%d,%d)' % (dx, dy))
                    vals = []
                    for step in range(4):
                        tcol, trow = col + dx*step, row + dy*step
                        if tcol < 0 or tcol >= CONNECT_FOUR_COLS:
                            break
                        if trow < 0 or trow >= len(self._position[tcol]):
                            break
                        #print('vals += %d at (%d,%d)' % (self._position[tcol][trow], tcol, trow))
                        vals += [ self._position[tcol][trow] ]
                    if len(vals) < 4:
                        continue
                    if max(vals) == min(vals):
                        #print(vals)
                        return vals[0]

        if self._turn_idx >= CONNECT_FOUR_COLS * CONNECT_FOUR_ROWS:
            return 0.5

        return -1

    def toString(self):
        ret = ''
        for row in range(CONNECT_FOUR_ROWS-1,-1,-1):
            for col in range(CONNECT_FOUR_COLS):
                if row >= len(self._position[col]):
                    ret += '. '
                elif self._position[col][row] == 0:
                    ret += 'O '
                else:
                    ret += 'X '
            ret += '\n'
        return ret

    def features(self):
        feature_list = []
        for col in range(CONNECT_FOUR_COLS):
            feature_list += [[]]
            for row in range(CONNECT_FOUR_ROWS):
                if row >= len(self._position[col]):
                    feature_list[-1] += [0]
                else:
                    feature_list[-1] += [2*self._position[col][row] - 1]

        state_vals = [self.player_turn, self._turn_idx]

        ret = np.array(feature_list).astype(float), np.array(state_vals)
        ret = [torch.from_numpy(val).type(torch.FloatTensor) for val in ret]
        return ret

    def _recalcCompressed(self):
        col_states = [tuple(self._position[col]) for col in range(CONNECT_FOUR_COLS)]
        cmp1 = tuple(col_states)
        col_states.reverse()
        cmp2 = tuple(col_states)
        self._compressed = min(cmp1, cmp2)

    @property
    def compressed(self):
        return self._compressed

    def __hash__(self):
        return self._compressed

        
class HumanConnectFourPlayer(Player):
    def __init__(self, id_length=8, interpreter=None):
        Player.__init__(self, id_length)
        self._interpreter = interpreter

    def getMove(self, game_state):
        moves = game_state.moves
        ret = None

        while True:
            inp = sys.stdin.readline()
            items = [int(item) for item in inp.strip().split()]
            col = items[0]

            if col in moves:
                return col

            print('illegal move: %d' % (col))


class ConnectFourGame(Game):
    def __init__(self):
        Game.__init__(self)
        self._state = ConnectFourGameState()