from tictactoe import *
from player import Player
from ai import AIPlayer

from tictactoe_ai import TicTacToeNet

model = TicTacToeNet()
player1 = AIPlayer(0, lambda x: x, model=model)
#player1 = Player()
#player2 = Player()
player2 = HumanTicTacToePlayer()

for _ in range(5):
    game = TicTacToeGame()

    player1.key = game.registerPlayer(player1)
    player2.key = game.registerPlayer(player2)
