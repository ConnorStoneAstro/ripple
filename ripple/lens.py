import numpy as np
from .core import coordinate_transform, coordinate_transform_gradient, coordinate_transform_laplacian, BaseLens

class Null_Lens(BaseLens):
    pass

class SIE_Lens(BaseLens):
    default_params = {"q": 1., "pa": 0., "norm": 1., "x0": 0., "y0": 0., "core": 0.}
    
    @property
    def mass(self):
        return self["norm"]
    
    @coordinate_transform_gradient
    def _gradient(self, X, Y):
        r = np.sqrt(X**2 + Y**2 + self["core"]**2)
        return self["norm"] * np.array((X/r, Y/r))

    @coordinate_transform_laplacian
    def _laplacian(self, X, Y):
        r = np.sqrt(X**2 + Y**2 + self["core"]**2)
        psi12 = -X*Y/r**3
        return self["norm"] * np.array(((Y**2 / r**3, psi12),(psi12, X**2 / r**3)))
        
    @coordinate_transform
    def potential(self, X, Y):
        return self["norm"] * np.sqrt(X**2 + Y**2 + self["core"]**2)
    

class Gaussian_Lens(BaseLens):
    default_params = {"q": 1., "pa": 0., "norm": 1., "x0": 0., "y0": 0., "sigma": 1.}

    @coordinate_transform_gradient
    def _gradient(self, X, Y):
        return - (self["norm"] / (np.sqrt(np.pi)*self["sigma"]**3)) * np.array([X, Y]) * np.exp(-0.5*(X**2 + Y**2)/self["sigma"]**2)

    @coordinate_transform_laplacian
    def _laplacian(self, X, Y):
        G = (self["norm"] / (np.sqrt(np.pi)*self["sigma"]**3)) * np.exp(-0.5*(X**2 + Y**2)/self["sigma"]**2)
        psi12 = X * Y / self["sigma"]**2
        return np.array((((X**2 / self["sigma"]**2 - 1), psi12),(psi12, (Y**2 / self["sigma"]**2 - 1)))) * G
        
    @coordinate_transform
    def potential(self, X, Y):
        return (self["norm"] / (np.sqrt(np.pi)*self["sigma"])) * np.exp(-0.5*(X**2 + Y**2)/self["sigma"]**2)

class PointMass_Lens(BaseLens):
    default_params = {"x0": 0., "y0": 0., "mass": 1., "q": 1., "pa": 0.}

    @property
    def mass(self):
        return self["mass"]
    
    @coordinate_transform_gradient
    def _gradient(self, X, Y):
        return - self["mass"] * np.array([X, Y]) / (X**2 + Y**2)

    @coordinate_transform_laplacian
    def _laplacian(self, X, Y):
        r2 = X**2 + Y**2
        psi12 = 2 * X * Y 
        return self["mass"] * np.array(((Y**2 - X**2, psi12),(psi12, X**2 - Y**2))) / r2**2
        
    @coordinate_transform
    def potential(self, X, Y):
        return self["mass"] * np.log(np.sqrt(X**2 + Y**2))

class ImagePotential_Lens(BaseLens):
    default_params = {"norm": 1.}

    def __init__(self, image, pixelscale = None, fov = None, **kwargs):
        super().__init__(**kwargs)
        assert (fov is not None) or (pixelscale is not None), "one of fov or pixelscale must be given"
        self.image = image
        self._image_sampler = None
        self.gradient = None
        self._gradient_sampler = None
        self.laplacian = None
        self._laplacian_sampler = None

        self.pixel_sum = np.sum(self.image)
        self.shape = np.array(self.image.shape)
        if fov is not None:
            self.fov = np.array(fov)
            self.pixelscale = self.fov / self.shape
        else:
            if isinstance(pixelscale, float):
                self.pixelscale = np.array([pixelscale, pixelscale])
            else:
                self.pixelscale = np.array(pixelscale)
            self.fov = self.shape * self.pixelscale

    @property
    def mass(self):
        return self["norm"] * self.pixel_sum
    
    @property
    def grid(self):
        return np.array((
            (np.arange(self.image.shape[0]) - (self.image.shape[0]-1)/2)*self.pixelscale,
            (np.arange(self.image.shape[1]) - (self.image.shape[1]-1)/2)*self.pixelscale
        ))
        
    def _gradient(self, X, Y):
        if self._gradient_sampler is None:
            self.gradient = np.gradient(self.image, self.pixelscale)
            self.gradient[0], self.gradient[1] = self.gradient[1], self.gradient[0]
            self._gradient_sampler = (
                RectBivariateSpline(self.grid[0],self.grid[1],self.gradient[0]),
                RectBivariateSpline(self.grid[0],self.grid[1],self.gradient[1]),                
            )
        return self["norm"] * np.array((self._gradient_sampler[0](X, Y, grid = False), self._gradient_sampler[1](X, Y, grid = False)))
    
    def _laplacian(self, X, Y):
        if self._laplacian_sampler is None:
            self.laplacian = np.array((
                np.gradient(self._gradient_samples[0], self.pixelscale),
                np.gradient(self._gradient_samples[1], self.pixelscale),
            ))
            self.laplacian[0][0], self.laplacian[0][1] = self.laplacian[0][1], self.laplacian[0][0]
            self.laplacian[1][0], self.laplacian[1][1] = self.laplacian[1][1], self.laplacian[1][0]
            self._laplacian_sampler = (
                (
                    RectBivariateSpline(self.grid[0],self.grid[1],self.laplacian[0][0]),
                    RectBivariateSpline(self.grid[0],self.grid[1],self.laplacian[0][1]),
                ),(
                    RectBivariateSpline(self.grid[0],self.grid[1],self.laplacian[1][0]),
                    RectBivariateSpline(self.grid[0],self.grid[1],self.laplacian[1][1]),            
                )
            )
        return self["norm"] * np.array((
            (self._laplacian_sampler[0][0](X, Y, grid = False), self._laplacian_sampler[0][1](X, Y, grid = False)),
            (self._laplacian_sampler[1][0](X, Y, grid = False), self._laplacian_sampler[1][1](X, Y, grid = False)),
        ))
    
    def potential(self, X, Y):
        if self._image_sampler is None:
            self._image_sampler = RectBivariateSpline(self.grid[0],self.grid[1],self.image)
        return self["norm"] * self._image_sampler(X, Y, grid = False)
        
    
# class NFW_Lens(BaseLens):
#     default_params = {"x0": 0., "y0": 0., "q": 1., "pa": 0., "ks": 1.}

#     @coordinate_transform_gradient
#     def _gradient(self, X, Y):
#         r = np.sqrt(X**2 + Y**2)
#         h = np.log(r/2)
#         h[r < 1] += 2 * np.arctan(np.sqrt((r[r < 1] - 1)/(r[r < 1] + 1))) / np.sqrt(r[r < 1]**2 - 1)
#         h[r > 1] += 2 * np.arctanh(np.sqrt((1 - r[r < 1])/(r[r < 1] + 1))) / np.sqrt(r[r < 1]**2 - 1)
#         h[r == 1] += 1
#         return - 4 * self["ks"] * h np.array((X, Y)) / r**2

#     @coordinate_transform_laplacian
#     def _laplacian(self, X, Y):
#         raise NotImplementedError("sorry not for the NFW")
        
#     @coordinate_transform
#     def potential(self, X, Y):
#         r = np.sqrt(X**2 + Y**2)
#         g = 0.5 * np.log(r/2)**2
#         g[r < 1] += 2 * np.arctan(np.sqrt((r[r < 1]-1)/(r[r < 1]+1)))**2
#         g[r > 1] -= 2 * np.arctanh(np.sqrt((1 - r[r < 1])/(r[r < 1]+1)))**2
#         return 4 * self["ks"] * g
