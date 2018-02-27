package game;

import player.Player;

import java.util.Collection;

/**
 * Created by Derek on 1/7/2017.
 */
public abstract class GameFactory {


    public abstract void createGame(Collection<Player> players);
}
