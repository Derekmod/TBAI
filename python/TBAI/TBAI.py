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
parser.add_argument("--load", type=str, help="file to load weights from")
parser.add_argument("--save", help="file to save weights to", type=str)
parser.add_argument("--ngames", type=int, help="max # of games for training", default=1)
parser.add_argument("--nepochs", type=int, help="max # of epochs for training", default=10)
parser.add_argument("--max-states", type=int, help="max # of states to search", default=100)
parser.add_argument("--human-turn", type=int, help="turn order of human (0 or 1)", default=1)
parser.add_argument("--game", type=str, help="ttt, C4", default='ttt')
args = parser.parse_args()

#model = TicTacToeNet()
#if args.load:
#    model = torch.load(args.load)
#player1 = AIPlayer(0, lambda x: x, model=model, max_states=args.max_states)


if __name__ == '__main__':
    if args.game == 'ttt':
        model = TicTacToeNet()
        gametype = TicTacToeGame
        humantype = HumanTicTacToePlayer
    elif args.game == 'C4':
        model = ConnectFourNet()
        gametype = ConnectFourGame
        humantype = HumanConnectFourPlayer

    if args.load:
        model = torch.load(args.load)

    ai = AIPlayer(0, lambda x: x, model, max_states=args.max_states)

    if args.train:
        ai.train_iterations = args.nepochs
        for game_num in range(args.ngames):
            training_game = gametype()
            training_game.registerPlayer(ai)
            training_game.registerPlayer(ai)
            training_game.start(display=False)
            print('finished %d games' % (game_num+1))

            if args.save:
                torch.save(model, args.save)
    else:
        human = humantype()

        game = gametype()
        if args.human_turn == 0:
            game.registerPlayer(human)
            game.registerPlayer(ai)
        else:
            game.registerPlayer(ai)
            game.registerPlayer(human)
        game.start()