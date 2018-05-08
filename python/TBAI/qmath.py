import math


def qlog(x, q=0.):
    q = float(q)
    if q == 0.:
        return math.log(x)
    return (x ** q - 1.) / q

def qexp(x, q=0.):
    q = float(q)
    if q == 0.:
        return math.exp(x)
    return (1. + q*x) ** (1./q)

def qlogit(p, q=0.):
    return qlog(p, q) - qlog(1.-p, q)

def qsigmoid(x, q=0.):
    return qexp(x, q) / (qexp(x, q) + qexp(-x, q) )

def log_sum(values):
    sum = values[0]
    for v in values[1:]:
        sup = max(sum, v)
        inf = min(sum, v)
        sum = sup + math.log(1 + math.exp(inf - sup))
    return sum