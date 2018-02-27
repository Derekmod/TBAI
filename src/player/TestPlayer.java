package player;

/**
 * Created by Derek on 1/8/2017.
 */
public class TestPlayer extends Player {
    public TestPlayer(String name) {
        super(name);
    }

    public String getPlayerId(String gameId) {
        return this.myGames.get(gameId);
    }
}
