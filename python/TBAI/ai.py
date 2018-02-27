'''Skeleton classes of heuristic minimax AIs.'''

from player import Player
from gamestate import GameState
from pq import PriorityQueue

import numpy as np
import math

class AIPlayer(Player):
    '''General intelligent player, uses A* minimax. '''
    def __init__(self, num_features=0, feature_extractor=None, architecture=None):
        '''Initialize player with instructions of how to create heuristic.
        Args:
            num_features: <int> length of feature vector
            feature_extractor: <State --> array-like> function to extract features from any state
            architecture: <Unknown?> specification of neural-net architecture
        '''
        Player.__init__(self)

        self._num_features = num_features
        self._feature_extractor = feature_extractor
        if not feature_extractor:
            self._feature_extractor = lambda(state): []

        #TODO: use architecture to initialize model
        #self.initialize_model()

    def heur(self, state):
        '''Heuristic estimate of win probability.'''
        victor = state.checkVictory()
        if victor >= 0:
            return float(victor)
        features = self._feature_extractor(state)
        # TODO: use neural net
        return 0.5

    def getMove(self, state):
        #curate a list of states to check
        hash_fn = lambda(node): node.id
        value_fn = lambda(node): node.depth
        pq = PriorityQueue(hash_fn, value_fn)

        root = StateNode(state, real_turn=state.player_turn, prob_power=2.)
        pq.add(root)

        nchecked = 0
        while nchecked < 100000 and len(pq): #terminal condition
            next = pq.pop()
            heur_val = self.heur(next.state)
            new_nodes = next.check(heur_val)

            for new_node in new_nodes:
                pq.add(new_node)
            nchecked += 1
        print 'Checked %d states' % nchecked

        # find best move
        best_node = None
        for node in root.children:
            print node.value
            print node.state.toString()
            if not best_node or (node.value - best_node.value) * (2*state.player_turn - 1) > 0:
                best_node = node

        #return state.moves[0] #TEMP
        return best_node.move


class StateNode(object):
    '''One position in the tree of play.

    '''
    def __init__(self, state, real_turn=0, parent=None, id=None, move=None, prob_power=1.):
        self._state = state
        self._parent = parent
        self._id = id
        self._move = move
        self._real_turn = real_turn

        self._checked = False
        self._children = []
        if id is None:
            self._id = np.random.randint(2e9)
        if parent:
            self._depth = parent._depth + 1
        else:
            self._depth = 0

        self._self_value = 0.5
        self._reported_value = 0.5
        self._expected_value = 0.5

        self._pending_moves = []

        self._prob_power = prob_power

    def check(self, heur_val):
        '''Gives a node a heuristic value.
        Returns:
            [StateNode]: list of added nodes
        '''
        self._checked = True

        self._self_value = heur_val
        self._reported_value = heur_val
        self._expected_value = heur_val

        #TODO: update parent value
        parent_ret = []
        if self._parent:
            self.parent.recalcValue()
            parent_ret = self._parent._addChild()
        if heur_val < 1. and heur_val > 0.:
            self._pending_moves = np.random.permutation(self._state.moves).tolist() #FUTURE: some way of ordering moves
            return parent_ret + self._addChild()
        else: 
            return parent_ret

    def _addChild(self):
        if not len(self._pending_moves):
            return []

        move = self._pending_moves.pop(-1)

        new_state = self._state.enactMove(move)
        new_node = StateNode(new_state, parent=self, move=move, real_turn=self._real_turn, prob_power=self._prob_power)
        self._children.append(new_node)
        return [new_node]

    def _updateExpectedValue(self):
        nUnknown = len(self._pending_moves)

    def recalcValue(self, verbose=False):
        # TODO: optimize to only update based on last change
        sign = 2*self.state.player_turn - 1 # direction of sign is good

        win_probs = [_winProb(child.value) for child in self.children]
        #print win_probs
        #print self.state.toString()
        choice_probs = _choiceProbs(probs=win_probs,
                                    turn_sign=sign, 
                                    my_turn=self.state.player_turn != self._real_turn, # TODO: why not when they're equal?
                                    num_unknown=len(self._pending_moves),
                                    prob_power=self._prob_power )
        if verbose:
            print win_probs
            print choice_probs

        p_novel = 1 - float(len(win_probs)) / float(len(win_probs) + len(self._pending_moves))

        sum_value = p_novel * _winProb(self._self_value)
        #sum_weight = 0.
        for i in range(len(win_probs)):
            sum_value += win_probs[i] * choice_probs[i]
            #sum_weight += choice_probs[i]

        self._expected_value = sum_value
        if self.parent and abs(self._expected_value - self._reported_value) > 0.0001:
            # TODO make condition smarter
            self.parent.recalcValue()

    @property
    def parent(self):
        return self._parent

    @property
    def children(self):
        return self._children

    @property
    def value(self):
        ### SHOULD ONLY BE ACCESSED BY PARENT ###
        self._reported_value = self._expected_value
        return self._reported_value

    @property
    def id(self):
        return self._id

    @property
    def depth(self):
        return self._depth

    @property
    def state(self):
        return self._state

    @property
    def move(self):
        return self._move

def _winUProb(heur_val):
    return math.exp(math.copysign(math.log(1+abs(heur_val)),
                                  heur_val))

def _choiceProbs(probs, turn_sign, my_turn=False, num_unknown=0, prob_power=1.):
    my_turn = not my_turn
    player_turn = 0.5 + 0.5*turn_sign
    prob_power = 2.
    if not len(probs):
        return

    ntotal = len(probs) + num_unknown

    directed_probs = []
    for tp in probs:
        directed_probs += [tp*turn_sign + 1 - player_turn]

    if my_turn:
        best_idx = 0
        best_val = probs[0]
        for i in range(1, len(probs)):
            v = directed_probs[i]
            if v > best_val:
                best_val = v
                best_idx = i

        ret = [0] * len(probs)
        ret[best_idx] = float(len(probs))/float(ntotal)
        return ret
    elif sum(directed_probs) == 0:
        # TODO: verify
        return [1./float(ntotal)] * len(probs)
    else:
        uprobs = []
        scale = 0.
        for tp in probs:
            p = tp*turn_sign + 1 - player_turn
            choice_prob = p**prob_power
            uprobs += [choice_prob]
            scale += choice_prob
        scale *= float(ntotal) / float(len(probs))
        for i in range(len(uprobs)):
            uprobs[i] /= scale
        return uprobs


def _winProb(heur_val):
    #p = _winUProb(heur_val)
    #return p/(p+1./p)
    return heur_val