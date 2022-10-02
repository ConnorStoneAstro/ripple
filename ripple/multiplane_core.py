import numpy as np
from scipy.interpolate import RectBivariateSpline
from utils import Axis_Ratio_Cartesian, Rotate_Cartesian
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

class Image(Universe):
    def __init__(self, image, pixelscale, **kwargs):
        super().__init__(**kwargs)
        self.image = image
        self.pixelscale = pixelscale
        self._sampler = None
        self._gradient_samples = None
        self._gradient = None
        self._laplacian_samples = None
        self._laplacian = None
        self.grid = np.array((
            (np.arange(self.image.shape[0]) - (self.image.shape[0]-1)/2)*self.pixelscale,
            (np.arange(self.image.shape[1]) - (self.image.shape[1]-1)/2)*self.pixelscale
        ))
        
    def gradient(self, X, Y):
        if self._gradient is None:
            self._gradient_samples = np.gradient(self.image, self.pixelscale)
            self._gradient_samples[0], self._gradient_samples[1] = self._gradient_samples[1], self._gradient_samples[0]
            self._gradient = (
                RectBivariateSpline(self.grid[0],self.grid[1],self._gradient_samples[0]),
                RectBivariateSpline(self.grid[0],self.grid[1],self._gradient_samples[1]),                
            )
        return np.array((self._gradient[0](X, Y, grid = False), self._gradient[1](X, Y, grid = False))) 

    def laplacian(self, X, Y):
        if self._laplacian is None:
            self._laplacian_samples = np.array((
                np.gradient(self._gradient_samples[0], self.pixelscale),
                np.gradient(self._gradient_samples[1], self.pixelscale),
            ))
            self._laplacian_samples[0][0], self._laplacian_samples[0][1] = self._laplacian_samples[0][1], self._laplacian_samples[0][0]
            self._laplacian_samples[1][0], self._laplacian_samples[1][1] = self._laplacian_samples[1][1], self._laplacian_samples[1][0]
            self._laplacian = (
                (
                    RectBivariateSpline(self.grid[0],self.grid[1],self._laplacian_samples[0][0]),
                    RectBivariateSpline(self.grid[0],self.grid[1],self._laplacian_samples[0][1]),
                ),(
                    RectBivariateSpline(self.grid[0],self.grid[1],self._laplacian_samples[1][0]),
                    RectBivariateSpline(self.grid[0],self.grid[1],self._laplacian_samples[1][1]),            
                )
            )
        return np.array((
            (self._laplacian[0][0](X, Y, grid = False), self._laplacian[0][1](X, Y, grid = False)),
            (self._laplacian[1][0](X, Y, grid = False), self._laplacian[1][1](X, Y, grid = False)),
        ))
    
    def __call__(self, X, Y):
        if self._sampler is None:
            self._sampler = RectBivariateSpline(self.grid[0],self.grid[1],self.image)
        return self._sampler(X, Y, grid = False)

class BaseModel(Universe):
    def __init__(self, **params):
        super().__init__(**params)
        self.params = params

    def __getitem__(self, key):
        return self.params.get(key, self.default_params[key])

    def gradient(self, X, Y):
        return np.zeros((2,) + X.shape)

    def laplacian(self, X, Y):
        return np.zeros((2,2) + X.shape)
    
    def __call__(self, X, Y):
        return np.zeros(X.shape)
    
class BaseLens(Universe):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.potential = None

    @property
    def mass(self):
        return 0.
        
    def alpha(self, X, Y):
        return np.array(self.potential.gradient(X, Y))

    def kappa(self, X, Y):
        Lp = self.potential.laplacian(X, Y)
        return 0.5 * (Lp[0][0] + Lp[1][1])

    def gamma(self, X, Y):
        Lp = self.potential.laplacian(X, Y)
        return np.array([0.5*(Lp[0][0] - Lp[1][1]), Lp[1][0]])
    
    def detA(self, X, Y):
        Lp = self.potential.laplacian(X, Y)
        return (1.0-Lp[0][0])*(1.0-Lp[1][1])-Lp[0][1]*Lp[1][0]
    
class BaseSource(Universe):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.source = None
    def __call__(self, X, Y):
        return self.source(X, Y)
    
class BasePlane(Universe):
    def __init__(self, z = 0., **kwargs):
        super().__init__(**kwargs)
        self.z = z
    
    def D(self, plane):
        return self.dM(self.z, plane.z)
    
class SourcePlane(BasePlane):
    def __init__(self, sources, **kwargs):
        super().__init__(**kwargs)
        self.sources = sources
        
    def __call__(self, X, Y):
        return sum(S(X, Y) for S in self.sources)
    
class LensPlane(BasePlane):
    def __init__(self, lenses, **kwargs):
        super().__init__(**kwargs)
        self.lenses = lenses

    @property
    def einstein_radius(self):
        return np.sqrt(sum(lens.mass for lens in self.lenses))
    def potential(self, X, Y):
        return sum(lens.potential(X, Y) for lens in self.lenses)
    def alpha(self, X, Y):
        return sum(lens.alpha(X, Y) for lens in self.lenses)
    def kappa(self, X, Y):
        return sum(lens.kappa(X, Y) for lens in self.lenses)
    def gamma(self, X, Y):
        Lp = sum(lens.potential.laplacian(X, Y) for lens in self.lenses)
        return np.array((0.5*(Lp[0][0] - Lp[1][1]), Lp[1][0]))
    def detA(self, X, Y):
        Lp = sum(lens.potential.laplacian(X, Y) for lens in self.lenses)
        return (1.0-Lp[0][0])*(1.0-Lp[1][1])-Lp[0][1]*Lp[1][0]
    def project_ray_mesh(self, X, Y, image, source):
        alpha = self.alpha(X, Y)
        return np.array([X - alpha[0], Y - alpha[1]])    

    def critical_lines(self, ax, show = True, resolution = 1000, fov = 2., **kwargs):
        if isinstance(resolution, int):
            resolution = (resolution, resolution)
        if isinstance(fov, float):
            fov = (fov, fov)
        contour_params = {
            "colors": "r",
            "linestyles": "--",
            "extent": (-fov[1]*self.einstein_radius, fov[1]*self.einstein_radius,-fov[0]*self.einstein_radius, fov[0]*self.einstein_radius)
        }
        contour_params.update(kwargs)
        XX, YY = np.meshgrid(np.linspace(contour_params["extent"][0], contour_params["extent"][1], resolution[1]), np.linspace(contour_params["extent"][2], contour_params["extent"][3], resolution[0]))
        CS = ax.contour(self.detA(XX, YY), levels = [0.0], **contour_params)
        return CS
    
    def caustics(self, ax, show = True, cl_kwargs = {}, **kwargs):
        cl = self.critical_lines(ax, **cl_kwargs)
        paths = cl.collections[0].get_paths()
        if len(paths) == 0:
            return
        for path in paths:
            vertices = path.interpolated(5).vertices
            x1 = np.array(list(float(vs[0]) for vs in vertices))
            x2 = np.array(list(float(vs[1]) for vs in vertices))
            
            a = self.alpha(x1, x2)
            y1 = x1 - a[0]
            y2 = x2 - a[1]
            caustic_params = {"color": "b", "linestyle": "-"}
            if path is paths[0]:
                caustic_params["label"] = "causitcs"
            caustic_params.update(kwargs)
            ax.plot(y1, y2, **caustic_params)
        
class ImagePlane(BasePlane):
    def __init__(self, shape, fov = None, pixelscale = None, **kwargs):
        super().__init__(**kwargs)
        assert (fov is not None) or (pixelscale is not None), "one of fov or pixelscale must be given"
        
        self.shape = np.array(shape)
        if fov is not None:
            self.fov = np.array(fov)
            self.pixelscale = self.fov / self.shape
        else:
            if isinstance(pixelscale, float):
                self.pixelscale = np.array([pixelscale, pixelscale])
            else:
                self.pixelscale = np.array(pixelscale)
            self.fov = self.shape * self.pixelscale
            
        self.XX, self.YY = np.meshgrid((np.arange(self.shape[1]) - (self.shape[1] - 1)/2)*self.pixelscale[1], (np.arange(self.shape[0]) - (self.shape[0] - 1)/2)*self.pixelscale[0])

    @property
    def plt_extent(self):
        return (-self.fov[1]/2, self.fov[1]/2, -self.fov[0]/2, self.fov[0]/2)
    def image(self, lens, source):
        X, Y = lens.project_ray_mesh(self.XX, self.YY, image = self, source = source)
        return source(X, Y)
        
class MultiPlane(Universe):
    def __init__(self, planes, **kwargs):
        super().__init__(**kwargs)
        self.planes = planes
        
class MultiLensPlane(MultiPlane):
    def __init__(self, planes, **kwargs):
        super().__init__(planes, **kwargs)

    def alpha_recursive(self, Xim1, Xi, image, source, i = 0):
        if i == len(self.planes):
            return Xi
        Dim1i = self.dM(image.z if i == 0 else self.planes[i-1].z, self.planes[i].z)
        Diip1 = self.dM(self.planes[i].z, self.planes[i+1].z if len(self.planes) > (i+1) else source.z)
        D0i = self.dM(image.z, self.planes[i].z)
        D = self.dM(image.z, source.z) / self.dM(self.planes[i].z, source.z)
        Xip1 = (Diip1 / Dim1i + 1) * Xi - (Diip1 / Dim1i) * Xim1 - Diip1 * D * self.planes[i].alpha(Xi[0]/D0i, Xi[1]/D0i)
        return self.alpha_recursive(Xi, Xip1, image, source, i + 1)
            
    def project_ray_mesh(self, X, Y, image, source):
        theta = self.alpha_recursive(0., self.dM(image.z, self.planes[0].z) * np.array([X, Y]), image, source) / self.dM(image.z, source.z)
        return theta
    
class MultiSourcePlane(MultiPlane):
    def __init__(self, planes, **kwargs):
        super().__init__(planes, **kwargs)
        
