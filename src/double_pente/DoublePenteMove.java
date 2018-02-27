package double_pente;


import game.GameMove;

/**
 * Created by Derek on 1/7/2017.
 */
public class DoublePenteMove extends GameMove {

    private int y,x;

    public DoublePenteMove(int y, int x) {
        this.y = y;
        this.x = x;
    }

    public int getY() { return this.y; }
    public int getX() { return this.x; }
}
