package ffn;

import java.util.function.Function;

/**
 * Created by Derek on 1/9/2017.
 */
public class Derivable implements Function<Double, Double> {

    private Function<Double, Double> inner;
    private Function<Double, Double> deriv;




    public static Derivable LogitFunction() {
        Derivable ret = new Derivable();
        ret.inner = (Double a) -> {
            if (a < 0) {
                return 1.0/(1.0 + Math.exp(-2*a));
            } else {
                return 1.0 - 1.0/(1.0+Math.exp(2*a));
            }
        };

        ret.deriv = (Double a) -> 2*ret.inner.apply(a)*ret.inner.apply(-a);

        return ret;
    }

    public static Derivable ArctanFunction() {
        Derivable ret = new Derivable();
        ret.inner = (Double a) -> Math.atan(a);
        ret.deriv = (Double a) -> 1.0/(1.0 + a*a);

        return ret;
    }

    public static Derivable HingeFunction(double threshhold) {
        Derivable ret = new Derivable();
        ret.inner = (Double a) -> Math.max(0, a-threshhold);
        ret.deriv = (Double a) -> (a >= threshhold) ? 1.0 : 0.0;

        return ret;
    }
    public static Derivable HingeFunction() {
        return Derivable.HingeFunction(0);
    }


    @Override
    public Double apply(Double a) {
        return this.inner.apply(a);
    }

    public Function<Double, Double> derivative() {
        return this.deriv;
    }
}
