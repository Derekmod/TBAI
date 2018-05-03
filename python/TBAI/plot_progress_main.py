import os
import matplotlib.pyplot as plt

from tictactoe import TicTacToeGame, HumanTicTacToePlayer
from connect_four import ConnectFourGame, HumanConnectFourPlayer
from game_utils import Player
from ai import AIPlayer

from tictactoe_ai import TicTacToeNet
from connect_four_ai import ConnectFourNet
import torch

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-v", "--verbose", action="store_true",
                    help="increase output verbosity")
parser.add_argument("-t", "--train", action="store_true")
parser.add_argument("--reference", type=str, help="net to compare ai against")
parser.add_argument("--dir", help="directory of nets", type=str)
parser.add_argument("--max-states", type=int, help="max # of states to search", default=100)
parser.add_argument("--human-turn", type=int, help="turn order of human (0 or 1)", default=1)
parser.add_argument("--game", type=str, help="ttt, C4", default='ttt')
parser.add_argument("--results-fn", type=str, help="name of results file", default="results.csv")
#parser.add_argument("--pass-through", type=bool, help="pass-through neural net?", default=False)
#parser.add_argument("--sideways-net", type=bool, help="sideways neural net?", default=False)
#parser.add_argument("--ref-pass-through", type=bool, help="pass-through neural net?", default=False)
#parser.add_argument("--ref-sideways-net", type=bool, help="sideways neural net?", default=False)
args = parser.parse_args()

#model = TicTacToeNet()
#if args.load:
#    model = torch.load(args.load)
#player1 = AIPlayer(0, lambda x: x, model=model, max_states=args.max_states)


if __name__ == '__main__':
    if args.game == 'ttt':
        #model = TicTacToeNet(pass_through_states=args.pass_through,
        #                     sideways_net=args.sideways_net)
        modeltype = TicTacToeNet
        gametype = TicTacToeGame
        humantype = HumanTicTacToePlayer
    elif args.game == 'C4':
        #model = ConnectFourNet(pass_through_states=args.pass_through,
        #                       sideways_net=args.sideways_net)
        modeltype = ConnectFourNet
        gametype = ConnectFourGame
        humantype = HumanConnectFourPlayer

    ref_model = torch.load(args.reference)
    ref_ai = AIPlayer(model=ref_model, max_states=args.max_states)

    files = os.listdir(args.dir)
    eps = []
    for fn in files:
        _, fn = fn.split('_')
        fn, _ = fn.split('.')
        eps += [int(fn)]
    eps.sort()
    winrates = []
    for ep in eps:
        model = torch.load(os.path.join(args.dir, 'model_{}.dat'.format(ep) ) )
        test_ai = AIPlayer(model=model, max_states=args.max_states)
        nwins = 0.
        ngames = 100
        for sample in range(ngames):
            game = gametype()
            turn = sample % 2
            if turn == 1:
                game.registerPlayer(ref_ai)
                game.registerPlayer(test_ai)
            else:
                game.registerPlayer(test_ai)
                game.registerPlayer(ref_ai)
            winner = game.start(display=False)
            nwins += abs(1.-turn - winner)
        winrates += [nwins / float(ngames)]

    for i in range(eps):
        print('%d,%f' % (eps[i], winrates[i]))
    stream = open(args.results_fn, 'w')
    for i in range(eps):
        stream.write('%d,%f\n' % (eps[i], winrates[i]))


    plt.plot(eps, winrates)
    plt.show()