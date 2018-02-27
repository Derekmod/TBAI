'''Skeleton classes of heuristic minimax AIs.'''

from player import Player
from gamestate import GameState
from pq import PriorityQueue

import numpy as np
import math

'''PENDING
add certainty or uncertainty
    end states have perfect certainty
    unexplored states have perfect uncertainty
    Must be used in recalculating value
    If changes significantly, recalculate parent
Stronger priority
    use uncertainty
    use derivative of uncertainty
Update threshholds
    consistent specification
    minimal computation at update
Rework choiceProbs
    add uncertainty to input and output
    remove 'my_turn'
        replace with some specification to choose maximum
        implement minimum?
    assign unseen values to the value of parent
Implement State comparisons so that we don't check states twice
clean recalcValue
'''

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
        '''Heuristic estimate of win probability.
        Args:
            state: <State> state of game
        Returns:
            <float> win probability. Lies in [0,1]
            <float> uncertainty. Lies in [0,?)
        '''
        # TODO output uncertainty
        victor = state.checkVictory()
        if victor >= 0:
            return float(victor)
        features = self._feature_extractor(state)
        # PENDING: use neural net
        return 0.5

    def getMove(self, state):
        '''Make the AI make a move.
        Args:
            state: <State>
        Returns:
            <Move> in the move_list of 'state'
        '''
        hash_fn = lambda(node): 0 #TODO remove
        value_fn = lambda(node): node.depth #TODO strengthen
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
        # PENDING: add randomness
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
    Attributes:
        state: <State>
        parent: <StateNode>
        children: [StateNode]
        move: <Move> move EXECUTED to reach this state
        depth: <int> number of moves since root state
        value: <float> in [0,1] (ONLY ACCESSED BY PARENT) #TODO issue
        uncertainty: <float> in [0,?)
    '''
    def __init__(self, state, real_turn=0, parent=None, move=None, prob_power=1.):
        '''Create StateNode.
        Args:
            state: <State>
            real_turn: <int> turn of root state
            parent: <StateNode>
            move: <Move> move executed to reach this state
            prob_power: <float> assumed skill of play #PENDING rework
        '''
        self._state = state
        self._real_turn = real_turn
        self._parent = parent
        self._move = move
        self._prob_power = prob_power

        self._checked = False
        self._children = []
        self._depth = 0
        if parent:
            self._depth = parent._depth + 1

        # TODO: use parent value instead of 0.5
        self._self_value = 0.5
        self._reported_value = 0.5
        self._expected_value = 0.5
        self._uncertainty = 1. #PENDING, should be maximum here

        self._pending_moves = []

    def check(self, heur_val, uncertainty=1.):
        '''Gives a node a heuristic value.
        Tells parent to update if there was significant change.
        Returns:
            [StateNode]: list of new nodes to explore
        '''
        self._checked = True

        self._self_value = heur_val
        self._reported_value = heur_val
        self._expected_value = heur_val

        # TODO: clean
        parent_ret = []
        if self._parent:
            self.parent.recalcValue()
            parent_ret = self._parent._addChild()
        if heur_val < 1. and heur_val > 0.: #PENDING use uncertainty instead
            self._pending_moves = np.random.permutation(self._state.moves).tolist() #FUTURE: some way of ordering moves
            return parent_ret + self._addChild()
        else: 
            return parent_ret

    def _addChild(self):
        '''Uses a pending move to create child.
        If there is no pending move, will do nothing. #TODO check
        Returns:
            <StateNode> new node to explore
        '''
        if not len(self._pending_moves):
            return []

        move = self._pending_moves.pop(-1)

        new_state = self._state.enactMove(move)
        new_node = StateNode(new_state, parent=self, move=move, real_turn=self._real_turn, prob_power=self._prob_power)
        self._children.append(new_node)
        return [new_node]

    def recalcValue(self):
        '''Recalculates expected value.
        If changed, tells parent to recalculate.
        '''
        # TODO: optimize to only update based on last change
        sign = 2*self.state.player_turn - 1 # direction of sign is good

        values = [child.value for child in self.children]
        choice_probs = _choiceProbs(values=values,
                                    turn_sign=sign, 
                                    my_turn=self.state.player_turn != self._real_turn, # TODO: why not when they're equal?
                                    num_unknown=len(self._pending_moves),
                                    prob_power=self._prob_power )

        p_novel = 1 - float(len(values)) / float(len(values) + len(self._pending_moves))

        sum_value = p_novel * self._self_value
        #sum_weight = 0.
        for i in range(len(values)):
            sum_value += values[i] * choice_probs[i]
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
    def depth(self):
        return self._depth

    @property
    def state(self):
        return self._state

    @property
    def move(self):
        return self._move

def _choiceProbs(values, turn_sign, my_turn=False, num_unknown=0, prob_power=1.):
    '''Probability that a player will take certain actions.
    Gives unseen moves average probability.
    Args:
        values: [float [0,1]] win probabilities
        turn_sign: <int -1,1> sign of player_turn
        my_turn: <bool> whether the player is myself
        num_unknown: <int> number of unseen states
        prob_power: <float> p_choice propto values**power
    Returns:
        [float [0,1]] probabilities of player choosing each action
            does NOT sum to 1 unless num_unknown is 0    
    '''
    player_turn = 0.5 + 0.5*turn_sign
    if not len(values):
        return []

    ntotal = len(values) + num_unknown

    directed_values = []
    for tp in values:
        directed_values += [tp*turn_sign + 1 - player_turn]

    if my_turn:
        best_idx = 0
        best_val = values[0]
        for i in range(1, len(values)):
            v = directed_values[i]
            if v > best_val:
                best_val = v
                best_idx = i

        ret = [0] * len(values)
        ret[best_idx] = float(len(values))/float(ntotal)
        return ret
    elif sum(directed_values) == 0:
        # TODO: verify
        return [1./float(ntotal)] * len(values)
    else:
        uprobs = []
        scale = 0.
        for tp in values:
            p = tp*turn_sign + 1 - player_turn
            choice_prob = p**prob_power
            uprobs += [choice_prob]
            scale += choice_prob
        scale *= float(ntotal) / float(len(values))
        for i in range(len(uprobs)):
            uprobs[i] /= scale
        return uprobs