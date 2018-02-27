package player.ai;

import game.GameState;

/**
 * Created by Derek on 1/8/2017.
 */
public abstract class Heuristic {

    public class HeuristicReturn {
        public int advantage = 0;
        public int uncertainty = 0;
    }

    public abstract HeuristicReturn evaluateState(GameState s);
}
