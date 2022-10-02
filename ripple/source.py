import numpy as np
from .core import coordinate_transform, BaseSource
        
class Gaussian_Source(BaseSource):
    default_params = {"q": 1., "pa": 0., "norm": 1., "x0": 0., "y0": 0., "sigma": 1.}
    @coordinate_transform
    def __call__(self, X, Y):
        return (self["norm"] / (np.sqrt(np.pi)*self["sigma"])) * np.exp(-0.5*(X**2 + Y**2)/self["sigma"]**2)
    
class Image_Source(BaseSource):
    default_params = {"norm": 1.}
    
    def __init__(self, image, pixelscale = None, fov = None, **kwargs):
        super().__init__(**kwargs)
        assert (fov is not None) or (pixelscale is not None), "one of fov or pixelscale must be given"
        self.image = image
        self._image_sampler = None

        self.pixel_sum = np.sum(self.image)
        self.shape = np.array(self.image.shape)
        if fov is not None:
            self.fov = fov
        else:
            self.pixelscale = pixelscale

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
        if isinstance(value, float):
            self._fov = np.array((value, value))
        else:
            self._fov = np.array(value)
        self._pixelscale = self._fov / self.shape

    @property
    def grid(self):
        return np.array((
            (np.arange(self.image.shape[0]) - (self.image.shape[0]-1)/2)*self.pixelscale[0],
            (np.arange(self.image.shape[1]) - (self.image.shape[1]-1)/2)*self.pixelscale[1]
        ))
        
    def __call__(self, X, Y):
        if self._image_sampler is None:
            self._image_sampler = RectBivariateSpline(self.grid[0],self.grid[1],self.image)
        return self["norm"] * self._image_sampler(X, Y, grid = False)
    
