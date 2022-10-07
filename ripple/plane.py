import numpy as np
from .core import BasePlane, BaseMultiPlane, BaseLens, BaseSource
from copy import deepcopy

class SourcePlane(BasePlane):
    def __init__(self, sources, **kwargs):
        kwargs["z"] = kwargs.get("z", 2.)
        super().__init__(**kwargs)
        self.sources = sources
        
    def __call__(self, X, Y):
        return sum(S(X, Y) for S in self.sources)
    
class LensPlane(BasePlane):
    def __init__(self, lenses, **kwargs):
        kwargs["z"] = kwargs.get("z", 1.)
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
        Lp = sum(lens._laplacian(X, Y) for lens in self.lenses)
        return np.array((0.5*(Lp[0][0] - Lp[1][1]), Lp[1][0]))
    def detA(self, X, Y):
        Lp = sum(lens._laplacian(X, Y) for lens in self.lenses)
        return (1.0-Lp[0][0])*(1.0-Lp[1][1])-Lp[0][1]*Lp[1][0]
    def project_ray_mesh(self, X, Y, image, source):
        alpha = self.alpha(X, Y)
        return np.array([X - alpha[0], Y - alpha[1]])    
        
class ImagePlane(BasePlane):
    def __init__(self, shape, fov = None, pixelscale = None, **kwargs):
        kwargs["z"] = kwargs.get("z", 0.)
        super().__init__(**kwargs)
        assert (fov is not None) or (pixelscale is not None), "one of fov or pixelscale must be given"
        
        self.shape = np.array(shape)
        if fov is not None:
            self.fov = fov
        else:
            self.pixelscale = pixelscale
            
        self.XX, self.YY = np.meshgrid((np.arange(self.shape[1]) - (self.shape[1] - 1)/2)*self.pixelscale[1], (np.arange(self.shape[0]) - (self.shape[0] - 1)/2)*self.pixelscale[0])

    @property
    def pixelscale(self):
        return self._pixelscale
    @pixelscale.setter
    def pixelscale(self, value):
        if isinstance(value, float):
            self._pixelscale = np.array((value, value))
        else:
            self._pixelscale = np.array(value)
            
        self._fov = self.shape * self._pixelscale
    @property
    def fov(self):
        return self._fov
    @fov.setter
    def fov(self, value):
        if isinstance(value, float) or isinstance(value, int):
            self._fov = np.array((value, value))
        else:
            self._fov = np.array(value)
            
        self._pixelscale = self._fov / self.shape
        
    @property
    def plt_extent(self):
        return (-self.fov[1]/2, self.fov[1]/2, -self.fov[0]/2, self.fov[0]/2)
    def image(self, lens, source):
        if isinstance(lens, BaseLens):
            lens = LensPlane([lens])
        if isinstance(source, BaseSource):
            source = SourcePlane([source])
        X, Y = lens.project_ray_mesh(self.XX, self.YY, image = self, source = source)
        return source(X, Y)

class MultiLensPlane(BaseMultiPlane):
    def __init__(self, planes, **kwargs):
        super().__init__(planes, **kwargs)
        
    def alpha_recursive(self, Xim1, Xi, image, source, i = 0, alphas = None):
        if i == len(self.planes):
            if alphas is not None:
                return alphas
            else:
                return Xi
            
        Dim1i = self.dM(image.z if i == 0 else self.planes[i-1].z, self.planes[i].z)
        Diip1 = self.dM(self.planes[i].z, self.planes[i+1].z if len(self.planes) > (i+1) else source.z)
        D0i = self.dM(image.z, self.planes[i].z)
        D =  self.dM(image.z, source.z) / self.dM(self.planes[i].z, source.z)
        Xip1 = (Diip1 / Dim1i + 1) * Xi - (Diip1 / Dim1i) * Xim1 - Diip1 * D * self.planes[i].alpha(Xi[0]/D0i, Xi[1]/D0i)
        
        if alphas is not None:
            alphas.append(Xip1 / self.dM(image.z, source.z))
        return self.alpha_recursive(Xi, Xip1, image, source, i + 1, alphas)
            
    def project_ray_mesh(self, X, Y, image, source):
        theta = self.alpha_recursive(0., self.dM(image.z, self.planes[0].z) * np.array([X, Y]), image, source, i = 0, alphas = None) / self.dM(image.z, source.z)
        return theta
    
class MultiSourcePlane(BaseMultiPlane):
    def __init__(self, planes, **kwargs):
        super().__init__(planes, **kwargs)
