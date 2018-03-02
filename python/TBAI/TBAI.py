from tictactoe import *
from player import Player
from ai import AIPlayer

from tictactoe_ai import TicTacToeNet

game = TicTacToeGame()

model = TicTacToeNet()
player1 = AIPlayer(0, lambda x: x, model=model)
#player1 = Player()
#player2 = Player()
player2 = HumanTicTacToePlayer()

player1.key = game.registerPlayer(player1)
player2.key = game.registerPlayer(player2)

game.start()