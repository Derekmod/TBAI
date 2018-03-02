from tictactoe import TicTacToeGame, HumanTicTacToePlayer
from game_utils import Player
from ai import AIPlayer

from tictactoe_ai import TicTacToeNet

model = TicTacToeNet()
player1 = AIPlayer(0, lambda x: x, model=model, max_states=1000)
#player1 = Player()
#player2 = Player()
player2 = HumanTicTacToePlayer()

for _ in range(10000):
    training_game = TicTacToeGame()
    training_game.registerPlayer(player1)
    training_game.registerPlayer(player1)
    training_game.start()

player1._max_states = 10

for _ in range(1000):
    game = TicTacToeGame()

    player1.key = game.registerPlayer(player1)
    player2.key = game.registerPlayer(player2)

    game.start()