package ffn;

import java.util.function.Function;

/**
 * Created by Derek on 1/8/2017.
 */
public class Matrix {
    private double[][] dat;
    private int width, height;

    public Matrix(int width, int height) {
        this.width = width;
        this.height = height;
        this.dat = new double[height][width];
    }

    public Matrix times(Matrix other) {
        int d = this.height;
        assert (d == other.width);

        Matrix ret = new Matrix(this.width, other.height);

        for (int i=0; i<ret.height; i++) {
            for (int j=0; j<ret.width; j++) {
                for (int k=0; k<d; k++) {
                    ret.dat[j][i] += this.dat[k][i] * other.dat[j][k];
                }
            }
        }

        return ret;
    }

    public Matrix times(double a) {
        Matrix ret = new Matrix(this.width, this.height);

        for (int i=0; i<this.height; i++) {
            for (int j=0; j<this.width; j++) {
                ret.dat[j][i] = this.dat[j][i] * a;
            }
        }

        return ret;
    }

    public Matrix vertcat(Matrix other) {
        assert (this.width == other.width);

        Matrix ret = new Matrix(this.width, this.height + other.height);

        for (int j=0; j<this.width; j++) {
            for (int i1=0; i1<this.height; i1++) {
                ret.dat[j][i1] = this.dat[j][i1];
            }
            for (int i2=0; i2<other.height; i2++) {
                ret.dat[j][this.height + i2] = other.dat[j][i2];
            }
        }

        return ret;
    }

    public Matrix cellfun(Function<Double, Double> f) {
        Matrix ret = new Matrix(this.width, this.height);

        for (int i=0; i<this.height; i++) {
            for (int j=0; j<this.width; j++) {
                ret.dat[i][j] = f.apply(this.dat[i][j]);
            }
        }

        return ret;
    }
}
