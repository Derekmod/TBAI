package player;

import game.Game;

import java.util.HashMap;

/**
 * Created by Derek on 1/7/2017.
 */
public class Player {

    protected String myName;
    protected HashMap<String, Game> myGames; // GameId -> Game
    protected HashMap<String, String> myPlayerIds; // GameId -> PlayerId

    public Player(String name) {
        this.myName = name;
        this.myGames = new HashMap<String,Game>();
    }

    public void addGame(String gameId, Game g, String playerId) {
        if (this.myGames.containsKey(gameId)) {
            //TODO: throw error
        }

        this.myGames.put(gameId, g);
        this.myPlayerIds.put(gameId, playerId);
    }

    public void notifyUpdate(String gameId) {
    }

    public String getName() { return this.myName; }
}
