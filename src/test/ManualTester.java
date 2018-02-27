package test;

import double_pente.*;
import player.*;

import java.util.ArrayList;
import java.util.HashMap;

/**
 * Created by Derek on 1/7/2017.
 */
public class ManualTester {

    public static void main(String[] args) {
        TestPlayer p1 = new TestPlayer("Alice");
        TestPlayer p2 = new TestPlayer("Bob");

        ArrayList<Player> players = new ArrayList<Player>();
        players.add(p1);
        players.add(p2);

        DoublePente g = new DoublePente("game1", players);

        String pid1 = p1.getPlayerId("game1");
        String pid2 = p2.getPlayerId("game1");

        DoublePenteMove tm = new DoublePenteMove(-1, -1);
        g.makeMove(pid2, tm);

        tm = new DoublePenteMove(1, 2);
        g.makeMove(pid2, tm);


        tm = new DoublePenteMove(1, 1);
        g.makeMove(pid1, tm);

        tm = new DoublePenteMove(-2, 0);
        g.makeMove(pid1, tm);


        tm = new DoublePenteMove(2, 2);
        g.makeMove(pid2, tm);

        tm = new DoublePenteMove(-2, 1);
        g.makeMove(pid2, tm);


        tm = new DoublePenteMove(3, 0);
        g.makeMove(pid1, tm);

        tm = new DoublePenteMove(-1, 3);
        g.makeMove(pid1, tm);


        tm = new DoublePenteMove(1, 1);
        g.makeMove(pid2, tm);

        tm = new DoublePenteMove(2, 3);
        g.makeMove(pid2, tm);


        tm = new DoublePenteMove(4, 2);
        g.makeMove(pid1, tm);

        tm = new DoublePenteMove(5, 6);
        g.makeMove(pid1, tm);


        tm = new DoublePenteMove(0, 0);
        g.makeMove(pid2, tm);

        tm = new DoublePenteMove(7, 8);
        g.makeMove(pid2, tm);


        tm = new DoublePenteMove(6, 6);
        g.makeMove(pid1, tm);

        tm = new DoublePenteMove(2, 9);
        g.makeMove(pid1, tm);


        tm = new DoublePenteMove(-2, -2);
        g.makeMove(pid2, tm);

        tm = new DoublePenteMove(7, 8);
        g.makeMove(pid2, tm);


        DoublePenteState s = (DoublePenteState)g.getState(pid2);
        for (InfiniteGoLocation loc : s.board.keySet()) {
            System.out.println(players.get(s.board.get(loc)).getName() + " at " + loc);
        }
    }
}
