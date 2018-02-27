package double_pente;

/**
 * Created by Derek on 1/7/2017.
 */
public class InfiniteGoLocation {
    private int y, x;

    public InfiniteGoLocation(int y, int x) {
        this.y = y;
        this.x = x;
    }

    public InfiniteGoLocation offset(int deltaY, int deltaX) {
        return new InfiniteGoLocation(this.y + deltaY, this.x + deltaX);
    }

    public InfiniteGoLocation offsetY(int deltaY) {
        return this.offset(deltaY, 0);
    }

    public InfiniteGoLocation offsetX(int deltaX) {
        return this.offset(0, deltaX);
    }


    public int getY() { return this.y; }
    public int getX() { return this.x; }

    @Override
    public String toString() {
        return "(" + this.y + "," + this.x + ")";
    }

    @Override
    public int hashCode() {
        int HASH_BASE = 10007;
        return this.y + this.x * HASH_BASE;
    }

    @Override
    public boolean equals(Object other) {
        InfiniteGoLocation o = (InfiniteGoLocation) other;

        return (this.y == o.y) && (this.x == o.x);
    }
}