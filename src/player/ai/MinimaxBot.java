package player.ai;

import game.Game;
import game.GameMove;
import game.GameState;
import player.Player;

import java.util.Iterator;
import java.util.function.Function;

/**
 * Created by Derek on 1/7/2017.
 */
public class MinimaxBot extends Player {

    private Function<GameState, Iterator<GameMove>> moveFinder;
    private Function<GameState, Double> myHeuristic;

    public MinimaxBot(String name, Function<GameState, Iterator<GameMove>> moves, Function<GameState, Double> heuristic) {
        super(name);

        this.moveFinder = moves;
        this.myHeuristic = heuristic;
    }

    @Override
    public void notifyUpdate(String gameId) {
        Game g = this.myGames.get(gameId);
        String pid = this.myPlayerIds.get(gameId);

        int playerNum = g.getPlayerNum(pid);

        GameState s = g.getState(pid);

        //TODO: find best move

//        g.makeMove(pid, bestMove);
    }
}
