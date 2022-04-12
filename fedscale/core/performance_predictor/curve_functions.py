import numpy as np



def pow3(x, c, a, alpha):
    return c - a * x**(-alpha)

def pow4(x, c, a, b, alpha):
    return c - (a*x+b)**-alpha

def log_power(x, a, b, c):
    #logistic power
    return a/(1.+(x/np.exp(b))**c)

def weibull(x, alpha, beta, kappa, delta):
    """
    Weibull modell

    http://www.pisces-conservation.com/growthhelp/index.html?morgan_mercer_floden.htm

    alpha: upper asymptote
    beta: lower asymptote
    k: growth rate
    delta: controls the x-ordinate for the point of inflection
    """
    return alpha - (alpha - beta) * np.exp(-(kappa * x)**delta)

def mmf(x, alpha, beta, kappa, delta):
    """
        Morgan-Mercer-Flodin

        description:
        Nonlinear Regression page 342
        http://bit.ly/1jodG17
        http://www.pisces-conservation.com/growthhelp/index.html?morgan_mercer_floden.htm  

        alpha: upper asymptote
        kappa: growth rate
        beta: initial value
        delta: controls the point of inflection
    """
    return alpha - (alpha - beta) / (1. + (kappa * x)**delta)

def janoschek(x, a, beta, k, delta):
    """
        http://www.pisces-conservation.com/growthhelp/janoschek.htm
    """
    return a - (a - beta) * np.exp(-k*x**delta)

def ilog2(x, c, a):
    x = 1 + x
    assert(np.all(x>1))
    return c - a / np.log(x)

def exp3(x, c, a, b):
    return c - np.exp(-a*x+b)

def exp4(x, c, a, b, alpha):
    return c - np.exp(-a*(x**alpha)+b)

def dr_hill_zero_background(x, theta, eta, kappa):
    return (theta* x**eta) / (kappa**eta + x**eta)