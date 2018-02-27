package game;

import java.util.ArrayList;
import java.util.Collection;
import java.util.HashMap;
import java.util.Random;
import player.Player;

/**
 * Created by Derek on 1/7/2017.
 */
public abstract class Game {
    protected static final int ID_LENGTH = 6;

//    public abstract class Move {
//
//    }
//    public abstract class State {
//
//    }

    private String myId;
    protected HashMap<String, Player> myPlayers;
    protected ArrayList<String> myPlayerIds;

    protected int victor = -1;



    public Game(String gameId, Collection<Player> players) {
        this.myId = gameId;
        this.myPlayers = new HashMap<String, Player>();
        this.myPlayerIds = new ArrayList<String>();

        for (Player p : players) {
            String id = this.createId();
            this.myPlayers.put(id, p);
            this.myPlayerIds.add(id);
            p.addGame(this.getId(), this, id);
        }

    }


    /**
     * Handles a player move
     * @param playerId id of player enacting
     * @param m must be of right type
     */
    public abstract void makeMove(String playerId, GameMove m);
    public abstract GameState getState(String playerId);

    public int getPlayerNum(String playerId) {
        return this.myPlayerIds.indexOf(playerId);
    }

    protected String createId() {
        String ret;
        do {
            ret = genId();
        } while (this.myPlayers.containsKey(ret));

        return ret;
    }
    private static String genId() {
        String ret = "";
        Random rand = new Random();
        String search = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789";

        for (int i=0; i<ID_LENGTH; i++) {
            ret += search.charAt(rand.nextInt(search.length()));
        }

        return ret;
    }


    public String getId() { return this.myId; }
}
