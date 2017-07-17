import numpy as np
from scipy.special import gamma as Gamma


class Weibull:
    """
    Continuous Weibull distribution
    """
    
    def __init__(self, lamb, k):
        self.lamb = lamb
        self.k = k
    
    def pmf(self, t):        
        return ((self.k/self.lamb) 
                * (t/self.lamb)**(self.k-1) 
                * np.exp(-(t/self.lamb)**self.k))
        
    def loghazard(self, t):
        return (-self.k * np.log(self.lamb)
                + np.log(self.k)
                + (self.k-1) * np.log(t))
    
    def hazard(self, t):
        idx, = np.where(t == 0)
        t[idx] = 1
        h = np.exp(self.loghazard(t))
        h[idx] = 0
        return h
    
    def survival(self, t):
        return np.exp( - (t/self.lamb) ** self.k)
    
    def cdf(self, t):
        return 1 - self.survival(t)
    
    def mean(self):
        return self.lamb * Gamma( 1 + 1/self.k)
    
    def mode(self):
        return self.lamb * (1 - 1/self.k)**(1/self.k) 

class DiscreteWeibull:
    """
    Reparametrized Discrete weibull distribution, using q = exp(-lamb**(-k)).
    This is both computationally and conceptually more convenient.
    """
    def __init__(self, lamb, k):
        assert lamb > 0
        assert k > 0
        self.lamb = lamb
        self.k = k
    
    def survival(self, t):
        return  np.exp( - (t/self.lamb)**self.k)
    
    def pmf(self, t):
        return self.survival(t) - self.survival(t+1)
    
    def hazard(self, t):
        return 1 - np.exp((t/self.lamb)**self.k - ((t+1)/self.lamb)**self.k)
    
    def mean(self):
        return self.lamb * Gamma(1 + 1/self.k)
    
    def var(self):
        return (self.lamb**2 
                * ( Gamma(1 + 2/self.k) 
                   -Gamma(1 + 1/self.k)**2 ))
    
class SingleParamDiscreteWeibull(DiscreteWeibull):
    """
    Single parameter version of the discrete Weibull. 
    It uses lamb = tau/gamma(1 + 1/k) to ensure that the mean of the 
    (continuous) distribution is exactly tau and sets k = log(tau) + k0
    with k0 a fixed parameter ensuring that for lamb=1 you get immediate death.
    """
    
    def __init__(self, gamma, k0=6):
        self.k = np.log(gamma) + k0
        self.lamb = gamma / Gamma(1 + 1/self.k)