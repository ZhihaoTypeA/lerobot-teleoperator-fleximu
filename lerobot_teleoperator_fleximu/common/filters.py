#Filter definitions

import numpy as np

class OneEuroFilter:
    def __init__(self, t0, x0, min_cutoff=1.0, beta=0.0, dcutoff=1.0):
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.dcutoff = dcutoff
        self.x_prev = np.array(x0, dtype=float)
        self.dx_prev = np.zeros_like(self.x_prev)
        self.t_prev = t0

    def smoothing_factor(self, t_e, cutoff):
        r = 2 * np.pi * cutoff * t_e
        return r / (r + 1)
    
    def __call__(self, t, x):
        t_e = t - self.t_prev
        if t_e <= 0.0:
            return self.x_prev
        dx = (x - self.x_prev) / t_e
        a_d = self.smoothing_factor(t_e, self.dcutoff)
        dx_hat = a_d * dx + (1 - a_d) * self.dx_prev

        cutoff = self.min_cutoff + self.beta * np.abs(dx_hat)
        a = self.smoothing_factor(t_e, cutoff)
        x_hat = a * x + (1 - a) * self.x_prev

        self.x_prev = x_hat
        self.dx_prev = dx_hat
        self.t_prev = t
        return x_hat
    
class DeadbandEMA:
    def __init__(self, alpha=0.25, deadband=0.01, x0=0.0):
        self.alpha = alpha
        self.deadband = deadband
        self.x = x0

    def __call__(self, x_new):
        if abs(x_new - self.x) < self.deadband:
            return self.x
        self.x = (1 - self.alpha) * self.x + self.alpha * x_new
        return self.x
