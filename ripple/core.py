import numpy as np
from scipy.interpolate import RectBivariateSpline
from .utils import Axis_Ratio_Cartesian, Rotate_Cartesian
import functools
import matplotlib.pyplot as plt
from scipy.integrate import quad

def coordinate_transform(func):
    @functools.wraps(func)
    def wrap_coordinate_transform(self, X, Y, *args, **kwargs):
        XX, YY = Axis_Ratio_Cartesian(self["q"], X - self["x0"], Y - self["y0"], -self["pa"])
        return func(self, XX, YY, *args, **kwargs)
    return wrap_coordinate_transform
def coordinate_transform_gradient(func):
    @functools.wraps(func)
    def wrap_coordinate_transform_gradient(self, X, Y, *args, **kwargs):
        XX, YY = Rotate_Cartesian(X - self["x0"], Y - self["y0"], self["pa"])
        fX, fY = func(self, XX, YY*self["q"], *args, **kwargs)
        return np.array(Rotate_Cartesian(fX, fY*self["q"], -self["pa"]))# fixme why factor of q?
    return wrap_coordinate_transform_gradient
def coordinate_transform_laplacian(func):
    @functools.wraps(func)
    def wrap_coordinate_transform_laplacian(self, X, Y, *args, **kwargs):
        XX, YY = Rotate_Cartesian(X - self["x0"], Y - self["y0"], self["pa"])
        F = func(self, XX, YY*self["q"], *args, **kwargs)
        F[1,1] *= self["q"]**2
        F[1,0] *= self["q"]
        F[0,1] *= self["q"]
        return F
    return wrap_coordinate_transform_laplacian

def _Edz(z, Omega_r, Omega_m, Omega_k, Omega_L):
    return 1 / np.sqrt(Omega_r*(1+z)**4 + Omega_m*(1+z)**3 + Omega_k*(1+z)**2 + Omega_L*(1+z))
def _Edzt(z, Omega_r, Omega_m, Omega_k, Omega_L):
    return 1 / ((1+z)*np.sqrt(Omega_r*(1+z)**4 + Omega_m*(1+z)**3 + Omega_k*(1+z)**2 + Omega_L*(1+z)))

class Universe(object):
    cosmology = {"Omega_r": 0., "Omega_m": 0.321, "Omega_L": 0.679, "c": 299792.458, "H0": 73.}
    def __init__(self, **kwargs):
        for key in kwargs:
            if key in Universe.cosmology:
                Universe.cosmology[key] = kwargs[key]
    @property
    def dH(self):
        # Hubble distance
        return self.cosmology["c"] / self.cosmology["H0"]
    @property
    def Omega_r(self):
        return self.cosmology["Omega_r"]
    @property
    def Omega_m(self):
        return self.cosmology["Omega_m"]
    @property
    def Omega_L(self):
        return self.cosmology["Omega_L"]
    @property
    def Omega_k(self):
        return 1. - self.Omega_r - self.Omega_m - self.Omega_L
    def dC(self, zi, zf):
        # Comoving distance
        return self.dH * quad(_Edz, zi, zf, args = (self.Omega_r, self.Omega_m, self.Omega_k, self.Omega_L))[0]
    def dT(self, zi, zf):
        # Light-travel distance
        return self.dH * quad(_Edzt, zi, zf, args = (self.Omega_r, self.Omega_m, self.Omega_k, self.Omega_L))[0]
    def dM(self, zi, zf):
        # Transverse comoving distance
        if np.isclose(self.Omega_k, 0):
            return self.dC(zi, zf)
        elif self.Omega_k > 0:
            return self.dH * np.sinh(np.sqrt(self.Omega_k) * self.dC(zi, zf) / self.dH) / np.sqrt(self.Omega_k)
        else:
            return self.dH * np.sin(np.sqrt(abs(self.Omega_k)) * self.dC(zi, zf) / self.dH) / np.sqrt(abs(self.Omega_k))
    def dA(self, zi, zf):
        # Angular diameter distance
        return self.dM(zi, zf) / (1 + zi)
    def dL(self, zi, zf):
        # Luminosity distance
        return self.dM(zi, zf) * (1 + zi)
    
class BaseLens(Universe):
    default_params = {}
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.params = {}
        for key in kwargs:
            if key in self.default_params:
                self.params[key] = kwargs[key]

    @property
    def mass(self):
        return 0.

    def _gradient(self, X, Y):
        return np.zeros((2,) + X.shape)

    def _laplacian(self, X, Y):
        return np.zeros((2,2) + X.shape)
    
    def potential(self, X, Y):
        return np.zeros(X.shape)
    
    def alpha(self, X, Y):
        return np.array(self._gradient(X, Y))

    def kappa(self, X, Y):
        Lp = self._laplacian(X, Y)
        return 0.5 * (Lp[0][0] + Lp[1][1])

    def gamma(self, X, Y):
        Lp = self._laplacian(X, Y)
        return np.array([0.5*(Lp[0][0] - Lp[1][1]), Lp[1][0]])
    
    def detA(self, X, Y):
        Lp = self._laplacian(X, Y)
        return (1.0-Lp[0][0])*(1.0-Lp[1][1])-Lp[0][1]*Lp[1][0]

    def __getitem__(self, key):
        return self.params.get(key, self.default_params[key])

class BaseSource(Universe):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.params = {}
        for key in kwargs:
            if key in self.default_params:
                self.params[key] = kwargs[key]
                
    def __getitem__(self, key):
        return self.params.get(key, self.default_params[key])

    def __call__(self, X, Y):
        return np.zeros(X.shape)
    
class BasePlane(Universe):
    def __init__(self, z = 0., **kwargs):
        super().__init__(**kwargs)
        self.z = z
    
    def D(self, plane):
        return self.dM(self.z, plane.z)
            
class BaseMultiPlane(Universe):
    def __init__(self, planes, **kwargs):
        super().__init__(**kwargs)
        self.planes = planes
        
        
