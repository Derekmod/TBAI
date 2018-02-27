package double_pente;

import game.*;
import javafx.util.Pair;
import player.Player;

import java.util.ArrayList;
import java.util.Collection;
import java.util.HashMap;

/**
 * Created by Derek on 1/7/2017.
 */
public class DoublePente extends Game {
    private static final int[] TURN_ORDER = {1, 1, 0, 0};

    private int turn;
    private DoublePenteMove previousMove;
    private int minY, maxY, minX, maxX;

    private HashMap<InfiniteGoLocation, Integer> myBoard; // (y,x) -> player


    public DoublePente(String gameId, Collection<Player> players) {
        super(gameId, players);

        this.myBoard = new HashMap<InfiniteGoLocation, Integer>();
        this.myBoard.put(new InfiniteGoLocation(0, 0), 0);

        this.turn = 0;
        this.minY = this.minX = -4;
        this.maxY = this.maxX = 4;
    }

    /**
     * PlayerIds are set to ["0","1"] and null players
     * doesn't really apply for games of imperfect information
     * @return duplicate board
     */
    public DoublePente copy(String playerId) {
        DoublePente ret = new DoublePente(this.getId(), new ArrayList<Player>());

        for (int pn = 0; pn<this.myPlayerIds.size(); pn++) {
            ret.myPlayers.put("" + pn, null);
            ret.myPlayerIds.add("" + pn);
        }

        ret.minX = this.minX;
        ret.maxX = this.maxX;
        ret.minY = this.minY;
        ret.maxY = this.maxY;

        ret.previousMove = new DoublePenteMove(this.previousMove.getY(), this.previousMove.getX());

        ret.victor = this.victor;

        return ret;
    }

    @Override
    public void makeMove(String playerId, GameMove m) {
        int playerTurn = TURN_ORDER[this.turn];
        if (playerId != this.myPlayerIds.get(playerTurn)) {
            //TDO: return error - not the right turn
            return;
        }

        DoublePenteMove move = (DoublePenteMove) m;
        InfiniteGoLocation loc = new InfiniteGoLocation(move.getY(), move.getX());
        if (TURN_ORDER[this.turn] == TURN_ORDER[(this.turn + 3) % TURN_ORDER.length]) {
            int deltaY = move.getY() - this.previousMove.getY();
            int deltaX = move.getX() - this.previousMove.getX();
            if (deltaX == 0 || deltaY == 0 || (deltaY / deltaX) * (deltaX / deltaY) != 0) {
                //TODO: check
                //TDO: return error - conflicting moves
                return;
            }
        }

        //check location not already used
        if (this.myBoard.containsKey(loc)) {
            //TDO: return error - location occupied
            return;
        }

        if (loc.getY() > this.maxY || loc.getY() < this.minY ||
                loc.getX() > this.maxX || loc.getX() < this.minX) {
            //TDO: return error - out of bounds
            return;
        }

        this.myBoard.put(loc, playerTurn);

        this.victor = this.checkVictory(loc);

        this.checkCaptures(loc);

        //extend bounds of board if necessary
        this.maxY = Math.max(this.maxY, move.getY() + 4);
        this.maxX = Math.max(this.maxX, move.getX() + 4);
        this.minY = Math.min(this.minY, move.getY() - 4);
        this.minX = Math.min(this.minX, move.getX() - 4);

        this.previousMove = move;
        this.turn = (this.turn + 1) % TURN_ORDER.length;
        this.myPlayers.get(this.myPlayerIds.get(TURN_ORDER[this.turn])).notifyUpdate(this.getId());
    }

    @Override
    public GameState getState(String playerId) {
        DoublePenteState ret = new DoublePenteState(this.myBoard);

        ret.maxX = this.maxX;
        ret.maxY = this.maxY;
        ret.minX = this.minX;
        ret.minY = this.minY;

        ret.victor = this.victor;

        return ret;
    }

    private int playerAt(InfiniteGoLocation loc) {
        if (!this.myBoard.containsKey(loc)) {
            return -1;
        }

        return this.myBoard.get(loc);
    }

    private void checkCaptures(InfiniteGoLocation loc) {
        if (!this.myBoard.containsKey(loc)) {
            //TDO: error - loc not found
            return;
        }
        int playerTurn = this.myBoard.get(loc);

        for (int ydir = -1; ydir <= 1; ydir++) {
            for (int xdir = -1; xdir <= 1; xdir++) {
                if (ydir == 0 && xdir == 0) {
                    continue;
                }

                if (this.playerAt(loc.offset(ydir*3, xdir*3)) != playerTurn) {
                    continue;
                }

                int other = this.playerAt(loc.offset(ydir, xdir));
                if (other < 0 || other == playerTurn) {
                    continue;
                }

                if (this.playerAt(loc.offset(ydir*2, xdir*2)) != other) {
                    continue;
                }

//                System.out.println("Surrounded stones, capped by " + loc + " and " + loc.offset(ydir*3, xdir*3));
                this.myBoard.remove(loc.offset(ydir, xdir));
                this.myBoard.remove(loc.offset(ydir*2, xdir*2));
            }
        }
    }

    private int checkVictory(InfiniteGoLocation loc) {
        int victor = this.playerAt(loc);
        if (victor < 0) {
            return victor;
        }

        for (int ydir = -1; ydir <= 0; ydir++) {
            for (int xdir = -1; xdir <= 1; xdir++) {
                if (ydir == 0 && xdir == 0) {
                    continue;
                }

                int consecutive = 1;
                for (int d=1; this.playerAt(loc.offset(ydir*d, xdir*d))==victor; d++) {
                    consecutive++;
                }
                for (int d = -1; this.playerAt(loc.offset(ydir*d, xdir*d))==victor; d--) {
                    consecutive++;
                }
                if (consecutive >= 5) {
                    System.out.println(this.myPlayers.get(this.myPlayerIds.get(victor)).getName() + " wins!!");
                    return victor;
                }
            }
        }

        return -1;
    }
}
