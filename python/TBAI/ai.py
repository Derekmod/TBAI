'''Skeleton classes of heuristic minimax AIs.'''

from __future__ import print_function
from game_utils import Player
from pq import PriorityQueue

from HeuristicDataset import HeuristicDataset, get_loader
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import time

import numpy as np
import math
from qmath import *
import random
import copy
#from asyncio import Queue
from multiprocessing import Manager, Process, TimeoutError, Pipe

'''PENDING
Stronger priority
    use uncertainty
    use derivative of uncertainty
    IGNORE PREVIOUS? - use total choice probability of state
Rework recalc
    use uncertainty
        if uncertainty changes, recalc parent
        compute uncertainty
        compute derivative of uncertainty
        surprise = err^2 + w|uncertainty - err^2|
    Update threshholds
        consistent specification
    minimal computation at update
Rework choiceProbs
    add uncertainty to input and output
    implement minimum/maximum
    assign unseen values to the value of parent
PQ
    Accelerate pop-push combined functionality in pq
    Implement max-items in pq
handle case where all choice probs are 0, still need value of parent
Stabalize 'depth' (currently can be corrupted when de-duping

Training heuristic:
    Every time a node is updated, add it to a list of nodes to train on
    At termination:
        order the list of training nodes (by how surprising they are and how probable they are)
        use the most surprising examples as training points for the network
'''

class AIPlayer(Player):
    '''General intelligent player, uses A* minimax. '''
    def __init__(self, num_features=0, feature_extractor=None, model=None, max_uncertainty=8., max_states=100, train_iterations=0,
                 recalc_power=1.1, q_choice=1):
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
            self._feature_extractor = lambda state: []

        self._max_uncertainty = max_uncertainty
        self._max_states = max_states

        self._model = model
        self.train_iterations = train_iterations
        self._recalc_power = recalc_power
        self.q_choice = q_choice

    def heur(self, state, train=True):
        '''Heuristic estimate of win probability.
        Args:
            state: <State> state of game
        Returns:
            <float [0,1]> win probability
            <float [0, max]> uncertainty
        '''
        victor = state.checkVictory()
        if victor >= 0:
            return (float(victor), 0)
        if self._model:
            #features = self._feature_extractor(state)
            #features = [Variable(torch.from_numpy(var).type(torch.FloatTensor)) for var in state.features()]
            features = [Variable(var) for var in state.features()]
            ret = self._model.forward(features)
            return ret.data[0][0], ret.data[0][1]
        else:
            return 0.5, self._max_uncertainty

    def getMove(self, state, nslaves=1):
        '''Make the AI make a move.
        Args:
            state: <State>
        Returns:
            <Move> in the move_list of 'state'
        '''
        hash_fn = lambda node: node.state.compressed #TODO remove
        #value_fn = lambda node: node._global_log_prob #TODO strengthen
        value_fn = lambda node: node.depth
        pq = PriorityQueue(hash_fn, value_fn)

        player_info = PlayerInfo(turn = state.player_turn,
                                 prob_power = 0.1,
                                 max_uncertainty = self._max_uncertainty,
                                 q=self.q_choice)
        root = StateNode(state, player_info)
        pq.add(root)

        redundant = dict()

        slave_procs = []
        slave_pipes = []
        for pidx in range(nslaves):
            slave_pipe, pipe = Pipe()
            proc = Process(target=AIPlayer.evalStates, args=(self, pipe))

            slave_procs += [proc]
            slave_pipes += [slave_pipe]

            proc.start()

        nchecked = 0
        target_slave = 0
        flying_nodes = 0
        while nchecked < self._max_states and len(pq) + flying_nodes: #terminal condition
            if len(pq):
                next = pq.pop()
                next_state = next.state
                compressed = next_state.compressed
                if compressed not in redundant:
                    redundant[compressed] = next
                else:
                    original = redundant[compressed]
                    #next = redundant[compressed]
                    #for new_node in next.parent._addChild():
                    for new_node in next.reportRedundant(original):
                        pq.add(new_node)
                    continue

                pipe = slave_pipes[target_slave]
                pipe.send(next_state)
                flying_nodes += 1
                nchecked += 1

            for pipe in slave_pipes:
                if pipe.poll():
                    try:
                        obj = pipe.recv()
                        flying_nodes -= 1
                        if not obj:
                            print('ERROR: slave closed before master [E1]')
                        heur_bundle, compressed = obj
                        new_nodes = redundant[compressed].check(heur_bundle)
                        for new_node in new_nodes:
                            pq.add(new_node)
                    except EOFError:
                        print('ERROR: slave closed before master [E2]')


        for pipe in slave_pipes:
            pipe.send(None)

        active_pipes = copy.copy(slave_pipes)
        while active_pipes:
            for pipe in active_pipes:
                try:
                    obj = pipe.recv()
                    if not obj:
                        active_pipes.remove(pipe)
                        continue
                    heur_bundle, compressed = obj
                    redundant[compressed].check(heur_bundle)
                except EOFError:
                    pipes.remove(pipe)



        if self.train_iterations:
            X = []
            Y = []
            while len(player_info.training_nodes):
                _, training_state, value, err = player_info.training_nodes.pop()
                #value = node._expected_value
                #err = (node._expected_value - node._self_value) ** 2
            
                x = training_state.features()
                y = torch.FloatTensor([value, err])

                X += [x]
                Y += [y]

            if self._model:
                self.train(X, Y)

        #cleanNode(root)
        #for child in root.children:
        #    child.recalcValue(verbose=True)

        # find best move
        # PENDING: add randomness
        best_node = None
        moves = []
        uprobs = []
        for node in root.children:
            #print(node.value, node._self_value # TODO re add
            #print(node.state.toString()) # TODO re add
            if not best_node or (node.value - best_node.value) * (2*state.player_turn - 1) > 0:
                best_node = node

            moves += [node.move]
            uprobs += [get_uprob(get_utility(node.value, state.player_turn), node.uncertainty, player_info)]

        if self.train_iterations > 0:
            prob_scale = random.uniform(0, sum(uprobs))
            for i in range(len(uprobs)):
                prob_scale -= uprobs[i]
                if prob_scale <= 0:
                    return moves[i]

        #return state.moves[0] #TEMP
        return best_node.move

    def evalStates(self, pipe):
        while True:
            try:
                state = pipe.recv()
                if not state:
                    pipe.send(None)
                    return
                heur_bundle = self.heur(state)
                pipe.send( (heur_bundle, state.compressed) )
            except EOFError:
                pipe.send(None)
                return


        

        




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
    def __init__(self, state, player_info, parent=None, move=None):
        '''Create StateNode.
        Args:
            state: <State>
            real_turn: <int> turn of root state
            parent: <StateNode>
            move: <Move> move executed to reach this state
            prob_power: <float> assumed skill of play #PENDING rework
        '''
        self._state = state
        self._parents = set()
        self._global_log_prob = 0.
        if parent:
            self._parents.add(parent)
            #if parent._global_log_prob is None:
            #    print('STATENODE HAS NO GLOBAL LOG PROB:')
            #    print(parent.state.toString())
            self._global_log_prob = parent._global_log_prob - math.log(parent._max_children)
        #else:
        #    print('STATENODE HAS NO PARENT:')
        #    print(state.toString())
        self._parent = parent
        self._move = move
        
        self._player_info = player_info

        self._checked = False
        self._children = set()
        self._compressed_children = set()
        self._child_values = dict()
        self._child_uprobs = dict()
        self._prob_scale = 0.
        self._depth = 0
        self._clamped = False
        if parent:
            self._depth = parent._depth + 1
        self._max_children = 0

        self._pending_moves = []

        #if self._global_log_prob is None:
        #    print('CURRENT STATENODE HAS NO LOG PROB:')
        #    print(state.toString())

    def pickLeaf(self):
        '''Randomly picks a leaf node.
        Returns:
            <StateNode>: parent of node picked
            [Move]: list of moves to arrive at leaf from self.parent
        '''
        if not self._checked:
            print('ERROR: PICKING UNCHECKED NODE')
        if self._prob_scale < 1e-5 or random.uniform(0, 1) > float(len(self.children)) / self._max_children:
            ret = self._addChild()
            if not len(ret):
                print('ERROR: picked child without unexplored children')
            return ret[0]
            #move = np.random.choice(self._pending_moves)
            #return self, [move]
        
        return self.pickChild().pickLeaf()

    def pickChild(self, randomize=True):
        cplist = []
        scale = 0.
        for child in self.children:
            if not child._clamped:
                uprob = self._child_uprobs[child.state.compressed]
                cplist.append( (uprob, child) )
                scale += uprob
        if not len(cplist):
            print('ERROR: picking with no unclamped children, but self is {} clamped'.format(self._clamped))

        #cplist = [(self._child_uprobs[child.state.compressed], child) for child in self.children]
        if not randomize:
            cplist.sort(reverse=True)
            return cplist[0][1]
        return stochasticChoice(cplist, self._prob_scale)

    def check(self, heur_bundle):
        '''Gives a node a heuristic value.
        Tells parent to update if there was significant change.
        Returns:
            [StateNode]: list of new nodes to explore
        '''
        heur_val, uncertainty = heur_bundle

        self._checked = True

        self._self_value = heur_val
        self._reported_value = heur_val
        self._expected_value = heur_val
        self._uncertainty = uncertainty
        self._total_uncertainty = uncertainty

        if uncertainty < 1e-5:
            self._clamped = True
            if self._parent:
                self._parent.checkClamped()

        # TODO: get (unscaled) choice prob
        #parent_utility = get_utility(self._self_value, self.parent.state.player_turn)
        #self._choice_uprob = parent_utility ** self._player_info.prob_power
        # self.parent.choice_scale += self.choice_uprbo
        
        #for tp in values:
        #    p = tp*turn_sign + 1 - player_turn
        #    choice_prob = p**prob_power
        #    uprobs += [choice_prob]
        #    scale += choice_prob
        #scale *= float(ntotal) / float(len(values))
        #for i in range(len(uprobs)):
        #    uprobs[i] /= scale
        #return uprobs

        # TODO: tell parent about my unscaled choice prob
        # TODO: use parent's total choice prob to calculate my own choice prob

        added_nodes = []
        for parent in self._parents:
            parent.registerChild(self)
        if self._parent:
            added_nodes += self._parent._addChild()
        if uncertainty > 0: # PENDING: and choice_prob > 0.
            self._pending_moves = np.random.permutation(self._state.moves).tolist() #FUTURE: some way of ordering moves
            self._max_children = len(self._pending_moves)
            added_nodes += self._addChild()
        return added_nodes

    def checkClamped(self):
        for child in self.children:
            if not child._clamped:
                return
        self._clamped = True
        if self._parent:
            self._parent.checkClamped()

    def _addChild(self):
        '''Uses a pending move to create child.
        If there is no pending move, will do nothing. #TODO check
        Returns:
            [StateNode] new node to explore (can be empty)
        '''
        if not self._pending_moves:
            return []

        move = self._pending_moves.pop(-1)

        new_state = self._state.enactMove(move)
        new_node = StateNode(new_state, self._player_info, self, move)
        #self._children.add(new_node)
        return [new_node]

    def registerChild(self, child):
        '''Registers a node as a CHECKED child. '''
        compressed = child.state.compressed
        if compressed in self._compressed_children:
            return

        self._children.add(child)
        self._compressed_children.add(compressed)

        value = child._expected_value
        self._child_values[compressed] = value
        uprob = get_uprob(get_utility(value, self.state.player_turn),
                          self._player_info)
        self._child_uprobs[compressed] = uprob
        self._prob_scale += uprob

        # update uprobs and self value
        self.recalcValue()  # TODO accelerate

    def updateChildValue(self, child, value):
        # TODO: implement
        key = child.state.compressed
        self._child_values[key] = value
        self._prob_scale -= self._child_uprobs[key]
        uprob = get_uprob(get_utility(value, self.state.player_turn),
                                            self._player_info)
        self._child_uprobs[key] = uprob
        self._prob_scale += uprob

        self.recalcValue()

    def addParent(self, parent):
        self._parents.add(parent)

    def reportRedundant(self, original):
        if not self._parent:
            return []

        #self._parent._children.remove(self)
        #self._max_children -= 1
        for parent in self._parents:
            if parent in original._parents:
                parent._max_children -= 1
            parent.registerChild(original)
            original.addParent(parent)
        return self._parent._addChild()

    def recalcValue(self, verbose=False, propogate=True):
        '''Recalculates expected value.
        If changed, tells parent to recalculate.
        '''
        if verbose:
            print([self._child_values[compressed_child] for compressed_child in self._compressed_children])
            print([self._child_uprobs[compressed_child] for compressed_child in self._compressed_children])
            print('max children:', self._max_children)
        p_novel = float(self._max_children - len(self._children)) / float(self._max_children)
        if verbose:
            print('p_novel:', p_novel)
        seen_value = 0.
        if self._prob_scale > 1e-9:
            uvalue = sum([self._child_values[child.state.compressed] * self._child_uprobs[child.state.compressed]
                            for child in self._children])
            seen_value = uvalue / self._prob_scale * (1-p_novel)
        else:
            p_novel = 1.  # FUTURE: revisit
        self._expected_value = seen_value + self._self_value * p_novel

        err = (self._expected_value - self._self_value) ** 2
        surprise = err
        if surprise > 1e-10:
            self._player_info.training_nodes.addNoDuplicate((surprise, self.state, self._expected_value, err))

        if self._parents and surprise > 1e-9 and propogate:
            self._reported_value = self._expected_value
            # FUTURE make condition smarter
            for parent in self._parents:
                parent.updateChildValue(self, self._reported_value)

        return p_novel
                
    @property
    def children(self):
        return self._children

    @property
    def value(self):
        #self._reported_value = self._expected_value
        #return self._reported_value
        return self._expected_value

    @property
    def uncertainty(self):
        return self._total_uncertainty

    @property
    def depth(self):
        return self._depth

    @property
    def state(self):
        return self._state

    @property
    def move(self):
        return self._move
    
def get_utility(value, player_turn):
    sign = 2*player_turn - 1
    return sign*value + 1 - player_turn

def get_uprob(utility, uncertainty, q=-1.):
    if utility <= 0.:
        return 0.
    #return (utility + 0.05) ** player_info.prob_power
    #return (utility + player_info.utility_cap) / (1 + player_info.utility_cap - utility)
    return -qlog(1. - utility, q)

def get_probs(utilities, uncertainties, est_val, max_children, player_info):
    uprobs = []
    sum_uprob = 0.
    for i in range(len(utilities)):
        utility = utilities[i]
        uncertainty = uncertainties[i]

        if utility == 1.:
            probs = [0.] * max_children
            probs[i] = 1.
            return probs

        uprob = get_uprob(utility, uncertainty, player_info.q)
        uprobs += [uprob]
        sum_uprob += uprob

    probs = [uprob / sum_uprob * len(utilities) / max_children for uprob in uprobs]
    probs += [1. - float(len(utiltiies)) / max_children ]

    return probs


    

class PlayerInfo(object):
    '''Stores player info.'''
    def __init__(self, turn=0, prob_power=1., max_uncertainty=1., training_nodes=None, utility_cap=1e-1, q=0.5):
        self.turn = turn
        self.prob_power = prob_power
        self.max_uncertainty = max_uncertainty
        self.training_nodes = training_nodes
        if self.training_nodes is None:
            self.training_nodes = PriorityQueue(lambda x: x[1].compressed, lambda x: x[0]) # TODO: implement max items
        self.utility_cap = utility_cap
        self.q = q


def stochasticChoice(weighted_list, scale=1.):
    z = random.uniform(0, scale)
    for i in range(len(weighted_list)):
        weight, item = weighted_list[i]
        z -= weight
        if z <= 0:
            return item
    print('ERROR: bad scale')