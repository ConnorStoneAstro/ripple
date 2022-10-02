import numpy as np
from multiplane_core import coordinate_transform, coordinate_transform_gradient, coordinate_transform_laplacian, BaseModel, BaseLens, BaseSource
from utils import Axis_Ratio_Cartesian, Rotate_Cartesian


class SIE(BaseModel):
    default_params = {"q": 1., "pa": 0., "norm": 1., "x0": 0., "y0": 0., "core": 0.}
    
    @coordinate_transform_gradient
    def gradient(self, X, Y):
        r = np.sqrt(X**2 + Y**2 + self["core"]**2)
        return X/r, Y/r

    @coordinate_transform_laplacian
    def laplacian(self, X, Y):
        r = np.sqrt(X**2 + Y**2 + self["core"]**2)
        psi12 = -X*Y/r**3
        return np.array(((Y**2 / r**3, psi12),(psi12, X**2 / r**3)))
        
    @coordinate_transform
    def __call__(self, X, Y):
        return self["norm"] * np.sqrt(X**2 + Y**2 + self["core"]**2)
    

class Gaussian(BaseModel):
    default_params = {"q": 1., "pa": 0., "norm": 1., "x0": 0., "y0": 0., "sigma": 1.}

    @coordinate_transform_gradient
    def gradient(self, X, Y):
        return -self["norm"] * np.array([X, Y]) * np.exp(-0.5*(X**2 + Y**2)/self["sigma"]**2) / self["sigma"]**2

    @coordinate_transform_laplacian
    def laplacian(self, X, Y):
        G = self["norm"] * np.exp(-0.5*(X**2 + Y**2)/self["sigma"]**2) / self["sigma"]**2
        psi12 = X * Y / self["sigma"]**2
        return np.array((((X**2 / self["sigma"]**2 - 1), psi12),(psi12, (Y**2 / self["sigma"]**2 - 1)))) * G
        
    @coordinate_transform
    def __call__(self, X, Y):
        return self["norm"] * np.exp(-0.5*(X**2 + Y**2)/self["sigma"]**2)

class PointMass(BaseModel):
    default_params = {"x0": 0., "y0": 0., "norm": 1., "q": 1., "pa": 0.}

    @coordinate_transform_gradient
    def gradient(self, X, Y):
        return - self["norm"] * np.array([X, Y]) / (X**2 + Y**2)

    @coordinate_transform_laplacian
    def laplacian(self, X, Y):
        r2 = X**2 + Y**2
        psi12 = 2 * X * Y 
        return np.array(((Y**2 - X**2, psi12),(psi12, X**2 - Y**2))) / r2**2
        
    @coordinate_transform
    def __call__(self, X, Y):
        return self["norm"] * np.log(np.sqrt(X**2 + Y**2))

class NFW(BaseModel):
    default_params = {"x0": 0., "y0": 0., "q": 1., "pa": 0., "ks": 1.}

    @coordinate_transform_gradient
    def gradient(self, X, Y):
        r = np.sqrt(X**2 + Y**2)
        h = np.log(r/2)
        h[r < 1] += 2 * np.arctan(np.sqrt((r[r < 1] - 1)/(r[r < 1] + 1))) / np.sqrt(r[r < 1]**2 - 1)
        h[r > 1] += 2 * np.arctanh(np.sqrt((1 - r[r < 1])/(r[r < 1] + 1))) / np.sqrt(r[r < 1]**2 - 1)
        h[r == 1] += 1
        return - 4 * self["ks"] * h np.array((X, Y)) / r**2

    @coordinate_transform_laplacian
    def laplacian(self, X, Y):
        raise NotImplementedError("sorry not for the NFW")
        
    @coordinate_transform
    def __call__(self, X, Y):
        r = np.sqrt(X**2 + Y**2)
        g = 0.5 * np.log(r/2)**2
        g[r < 1] += 2 * np.arctan(np.sqrt((r[r < 1]-1)/(r[r < 1]+1)))**2
        g[r > 1] -= 2 * np.arctanh(np.sqrt((1 - r[r < 1])/(r[r < 1]+1)))**2
        return 4 * self["ks"] * g
    
    
class SIE_Lens(BaseLens):
    def __init__(self, q = 1., pa = 0., norm = 1., core = 0., x0 = 0., y0 = 0.):
        super().__init__()        
        self.q = q
        self.pa = pa
        self.norm = norm
        self.core = core
        self.x0 = x0
        self.y0 = y0
        self.potential = SIE(q = self.q, pa = self.pa, norm = self.norm, core = self.core, x0 = self.x0, y0 = self.y0)
    @property
    def mass(self):
        return self.norm

class PointMass_Lens(BaseLens):
    def __init__(self, x0 = 0., y0 = 0., norm = 1.):
        self.norm = norm
        self.x0 = x0
        self.y0 = y0
        self.potential = PointMass(norm = self.norm, x0 = self.x0, y0 = self.y0)
    @property
    def mass(self):
        return self.norm
        
class Gaussian_Source(BaseSource):
    def __init__(self, q = 1., pa = 0., norm = 1., x0 = 0., y0 = 0., sigma = 1.):
        super().__init__()
        
        self.q = q
        self.pa = pa
        self.norm = norm
        self.x0 = x0
        self.y0 = y0
        self.sigma = sigma
        self.source = Gaussian(q = self.q, pa = self.pa, norm = self.norm, x0 = self.x0, y0 = self.y0, sigma = self.sigma)
    
    
