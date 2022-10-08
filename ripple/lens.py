import numpy as np
from .core import coordinate_transform, coordinate_transform_gradient, coordinate_transform_laplacian, BaseLens, coordinate_shift, coordinate_rotate, coordinate_rotate_rev
from astropy.convolution import convolve_fft, convolve
from scipy.interpolate import RectBivariateSpline
from scipy.integrate import dblquad

class Null_Lens(BaseLens):
    pass

class PointMass_Lens(BaseLens):
    default_params = {"x0": 0., "y0": 0., "norm": 1.}

    @property
    def mass(self):
        return self["norm"]
    
    @coordinate_shift
    def _gradient(self, X, Y):
        return - self["norm"] * np.array([X, Y]) / (X**2 + Y**2)

    @coordinate_shift
    def _laplacian(self, X, Y):
        r2 = X**2 + Y**2
        psi12 = 2 * X * Y 
        return self["norm"] * np.array(((Y**2 - X**2, psi12),(psi12, X**2 - Y**2))) / r2**2
        
    @coordinate_shift
    def potential(self, X, Y):
        return self["norm"] * np.log(np.sqrt(X**2 + Y**2))

class PowerLawPotential_Lens(BaseLens):
    default_params = {"q": 1., "pa": 0., "norm": 1., "x0": 0., "y0": 0., "alpha": 1.}

    @coordinate_transform_gradient
    def _gradient(self, X, Y):
        pass
    
    @coordinate_transform_laplacian
    def _laplacian(self, X, Y):
        pass
    
    @coordinate_transform
    def potential(self, X, Y):
        return self["norm"] * (X**2 + (Y/self["q"])**2 + self["core"]**2)**(self["alpha"]/2) - self["norm"]*self["core"]**self["alpha"]
    
class PowerLawDensity_Lens(BaseLens):
    default_params = {"q": 1., "pa": 0., "norm": 1., "x0": 0., "y0": 0., "alpha": 1.}

    @coordinate_transform_gradient
    def _gradient(self, X, Y):
        pass
    
    @coordinate_transform_laplacian
    def _laplacian(self, X, Y):
        pass
    
    @coordinate_transform
    def potential(self, X, Y):
        pass
    
    def kappa(self, X, Y):
        return (self["norm"]**(2-self["alpha"])) / (2 * (self["core"]**2 + X**2 + (Y/self["q"])**2)**(1 - self["alpha"]/2))

class SIS_Lens(BaseLens):
    default_params = {"norm": 1., "x0": 0., "y0": 0.}

    @property
    def mass(self):
        return self["norm"]

    @coordinate_shift
    def _gradient(self, X, Y):
        r = np.sqrt(X**2 + Y**2)
        return self["norm"] * np.array((X/r, Y/r))

    @coordinate_shift
    def _laplacian(self, X, Y):
        r3 = np.sqrt(X**2 + Y**2)**3
        psi12 = -X*Y/r3
        return self["norm"] * np.array(((Y**2 / r3, psi12),(psi12, X**2 / r3)))

    @coordinate_shift
    def potential(self, X, Y):
        return self["norm"] * np.sqrt(X**2 + Y**2)

class SIE_Lens(BaseLens):
    default_params = {"q": 0.9, "pa": 0., "norm": 1., "x0": 0., "y0": 0.}
    # See https://articles.adsabs.harvard.edu/pdf/1994A%26A...284..285K
    # Kornmann et al. 1993
    # maybe try: https://arxiv.org/pdf/astro-ph/0102341v2.pdf
    
    @property
    def mass(self):
        return self["norm"]
    
    @coordinate_shift
    @coordinate_rotate_rev
    def _gradient(self, X, Y):
        qprime = np.sqrt(1 - self["q"]**2)
        # psi = np.sqrt(self["q"]*(self["core"]**2 + X**2) + Y**2)
        # return self["norm"] * self["q"] * np.array((np.arctan(qprime * X / (self["core"] + psi)), np.arctanh(qprime * Y / (self["q"]**2 * self["core"] + psi)))) / qprime
        theta = np.arctan2(Y, X)
        return self["norm"] * np.sqrt(self["q"]) * np.array((
            np.arcsinh(qprime * np.cos(theta) / self["q"]),
            np.arcsin(qprime * np.sin(theta)),
        )) / qprime

    @coordinate_shift
    @coordinate_rotate
    def _laplacian(self, X, Y):
        r = np.sqrt(X**2 + (self["q"]*Y)**2)
        kappa = np.sqrt(self["q"]) / (2 * r)
        theta = np.arctan2(Y, X)
        psi12 = kappa * np.sin(2*theta)
        return self["norm"] * np.array(((2 * kappa * np.sin(theta)**2, psi12),(psi12, 2 * kappa * np.cos(theta)**2)))

    @coordinate_shift
    @coordinate_rotate
    def kappa(self, X, Y):
        r = np.sqrt(X**2 + (self["q"]*Y)**2)
        return self["norm"] * np.sqrt(self["q"]) / (2 * r)

    @coordinate_shift
    @coordinate_rotate
    def detA(self, X, Y):
        kappa = np.sqrt(self["q"]) / (2 * np.sqrt(X**2 + (self["q"]*Y)**2))
        return 1 - 2 * self["norm"] * kappa
        
    @coordinate_shift
    @coordinate_rotate
    def gamma(self, X, Y):
        theta = np.arctan2(Y, X)
        kappa = np.sqrt(self["q"]) / (2 * np.sqrt(X**2 + (self["q"]*Y)**2))
        return -self["norm"] * kappa * np.array((np.cos(2*theta), np.sin(2*theta)))
    
    @coordinate_shift
    @coordinate_rotate
    def potential(self, X, Y):
        qprime = np.sqrt(1 - self["q"]**2)
        r = np.sqrt(X**2 + (self["q"]*Y)**2)
        theta = np.arctan2(Y, X)
        term1 = np.sqrt(self["q"]) * r / qprime
        term2 = np.sin(theta) * np.arcsin(qprime * np.sin(theta)) + np.cos(theta) * np.arcsinh(qprime * np.cos(theta) / self["q"])
        return self["norm"] * term1 * term2
    

class NIS_Lens(BaseLens):
    default_params = {"norm": 1., "x0": 0., "y0": 0., "core": 0.}

    @property
    def mass(self):
        return self["norm"]

    @coordinate_shift
    def _gradient(self, X, Y):
        r2 = X**2 + Y**2
        m = np.sqrt(X**2 + Y**2 + self["core"]**2)
        return self["norm"] * m * np.array((X, Y)) / r2

    @coordinate_shift
    def _laplacian(self, X, Y):
        r2 = X**2 + Y**2
        rc = np.sqrt(X**2 + Y**2 + self["core"]**2)
        psi11 = Y**4 + (self["core"]**2 + 1)*Y**2 + (1 - self["core"])*X**2 + self["core"] * (X**2 - Y**2) * rc
        psi22 = X**4 + (self["core"]**2 + 1)*X**2 + (1 - self["core"])*Y**2 + self["core"] * (Y**2 - X**2) * rc
        psi12 = - X * Y * (r2 + 2*self["core"]**2 - 2*self["core"]*rc)
        return self["norm"] * np.array(((psi11, psi12),(psi12, psi22))) / (r2 * rc)

    @coordinate_shift
    def kappa(self, X, Y):
        return self["norm"] / (2 * np.sqrt(X**2 + Y**2 + self["core"]**2))
    
    @coordinate_shift
    def potential(self, X, Y):
        r = np.sqrt(X**2 + Y**2 + self["core"]**2)
        return self["norm"] * (r - self["core"]*np.log(self["core"] + r))

class NIE_Lens(BaseLens):
    default_params = {"q": 0.9, "pa": 0., "norm": 1., "x0": 0., "y0": 0., "core": 0.1}
    # See https://articles.adsabs.harvard.edu/pdf/1994A%26A...284..285K
    # Kornmann et al. 1993
    # maybe try: https://arxiv.org/pdf/astro-ph/0102341v2.pdf

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if kwargs.get("do_warn", True):
            print("WARNING: currently only kappa works for NIE, otherwise its jst an SIE")
    
    @property
    def mass(self):
        return self["norm"]
    
    @coordinate_shift
    @coordinate_rotate_rev
    def _gradient(self, X, Y):
        qprime = np.sqrt(1 - self["q"]**2)
        # psi = np.sqrt(self["q"]*(self["core"]**2 + X**2) + Y**2)
        # return self["norm"] * self["q"] * np.array((np.arctan(qprime * X / (self["core"] + psi)), np.arctanh(qprime * Y / (self["q"]**2 * self["core"] + psi)))) / qprime
        theta = np.arctan2(Y, X)
        return self["norm"] * np.sqrt(self["q"]) * np.array((
            np.arcsinh(qprime * np.cos(theta) / self["q"]),
            np.arcsin(qprime * np.sin(theta)),
        )) / qprime

    @coordinate_shift
    @coordinate_rotate
    def _laplacian(self, X, Y):
        r = np.sqrt(X**2 + (self["q"]*Y)**2)
        kappa = np.sqrt(self["q"]) / (2 * r)
        theta = np.arctan2(Y, X)
        psi12 = kappa * np.sin(2*theta)
        return self["norm"] * np.array(((2 * kappa * np.sin(theta)**2, psi12),(psi12, 2 * kappa * np.cos(theta)**2)))

    @coordinate_shift
    @coordinate_rotate
    def kappa(self, X, Y):
        r = np.sqrt(X**2 + (self["q"]*Y)**2 + self["core"]**2)
        return self["norm"] * np.sqrt(self["q"]) / (2 * r)

    @coordinate_shift
    @coordinate_rotate
    def detA(self, X, Y):
        kappa = np.sqrt(self["q"]) / (2 * np.sqrt(X**2 + (self["q"]*Y)**2))
        return 1 - 2 * self["norm"] * kappa
        
    @coordinate_shift
    @coordinate_rotate
    def gamma(self, X, Y):
        theta = np.arctan2(Y, X)
        kappa = np.sqrt(self["q"]) / (2 * np.sqrt(X**2 + (self["q"]*Y)**2))
        return -self["norm"] * kappa * np.array((np.cos(2*theta), np.sin(2*theta)))
    
    @coordinate_shift
    @coordinate_rotate
    def potential(self, X, Y):
        qprime = np.sqrt(1 - self["q"]**2)
        r = np.sqrt(X**2 + (self["q"]*Y)**2)
        theta = np.arctan2(Y, X)
        term1 = np.sqrt(self["q"]) * r / qprime
        term2 = np.sin(theta) * np.arcsin(qprime * np.sin(theta)) + np.cos(theta) * np.arcsinh(qprime * np.cos(theta) / self["q"])
        return self["norm"] * term1 * term2
    
    
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
    
class ImagePotential_Lens(BaseLens):
    default_params = {"norm": 1.}

    def __init__(self, image, pixelscale = None, fov = None, **kwargs):
        super().__init__(**kwargs)
        assert (fov is not None) or (pixelscale is not None), "one of fov or pixelscale must be given"
        self.image = image
        self._image_sampler = None
        self._gradient_samples = None
        self._gradient_sampler = None
        self.laplacian = None
        self._laplacian_sampler = None

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
    def mass(self):
        return self["norm"] * self.pixel_sum
    
    @property
    def grid(self):
        return np.array((
            (np.arange(self.image.shape[1]) - (self.image.shape[1]-1)/2)*self.pixelscale[1],
            (np.arange(self.image.shape[0]) - (self.image.shape[0]-1)/2)*self.pixelscale[0],
        ))
        
    def _gradient(self, X, Y):
        if self._gradient_sampler is None:
            if self._image_sampler is None:
                self.potential(0,0)
            self._gradient_sampler = (
                self._image_sampler.partial_derivative(1,0),
                self._image_sampler.partial_derivative(0,1),
            )
            # self._gradient_samples = np.gradient(self.image)
            # self._gradient_sampler = (
            #     RectBivariateSpline(self.grid[0],self.grid[1],self._gradient_samples[0]/self.pixelscale[0]),
            #     RectBivariateSpline(self.grid[0],self.grid[1],self._gradient_samples[1]/self.pixelscale[1]),                
            # )
        return self["norm"] * np.array((self._gradient_sampler[0](X, Y, grid = False), self._gradient_sampler[1](X, Y, grid = False)))
    
    def _laplacian(self, X, Y):
        if self._laplacian_sampler is None:
            if self._gradient_sampler is None:
                self._gradient(0,0)
            
            self._laplacian_sampler = ((
                self._image_sampler.partial_derivative(2,0),
                self._image_sampler.partial_derivative(1,1),
            ),(
                self._image_sampler.partial_derivative(1,1),
                self._image_sampler.partial_derivative(0,2),
            ))
                                       
            # self.laplacian = np.array((
            #     np.gradient(self._gradient_samples[0]),
            #     np.gradient(self._gradient_samples[1]),
            # ))
            # self._laplacian_sampler = (
            #     (
            #         RectBivariateSpline(self.grid[0],self.grid[1],self.laplacian[0][0]/self.pixelscale[0]),
            #         RectBivariateSpline(self.grid[0],self.grid[1],self.laplacian[0][1]/self.pixelscale[0]),
            #     ),(
            #         RectBivariateSpline(self.grid[0],self.grid[1],self.laplacian[1][0]/self.pixelscale[1]),
            #         RectBivariateSpline(self.grid[0],self.grid[1],self.laplacian[1][1]/self.pixelscale[1]),            
            #     )
            # )
        return self["norm"] * np.array((
            (self._laplacian_sampler[0][0](X, Y, grid = False), self._laplacian_sampler[0][1](X, Y, grid = False)),
            (self._laplacian_sampler[1][0](X, Y, grid = False), self._laplacian_sampler[1][1](X, Y, grid = False)),
        ))
    
    def potential(self, X, Y):
        if self._image_sampler is None:
            self._image_sampler = RectBivariateSpline(self.grid[0],self.grid[1],self.image)
        return self["norm"] * self._image_sampler(X, Y, grid = False)

class ImageKappa_Lens(BaseLens):
    default_params = {"norm": 1.}

    def __init__(self, image, pixelscale = None, fov = None, **kwargs):
        super().__init__(**kwargs)
        assert (fov is not None) or (pixelscale is not None), "one of fov or pixelscale must be given"
        preshape = image.shape
        self.image = image[0:image.shape[0] - 1 + (image.shape[0] % 2),0:image.shape[1] - 1 + (image.shape[1] % 2)]
        self.image /= 5e-4*1e-10 * self.cosmology["c"]**2 / (4 * np.pi * self.cosmology["G"])
        if self.image.shape != preshape:
            print("WARNING: kappa maps must have odd dimensions, slicing off last pixel")
        self._image_sampler = None
        self._gradient_samples = None
        self._gradient_sampler = None
        self.laplacian = None
        self._laplacian_sampler = None
        self._alpha_sampler = None
        self._potential_sampler = None
        
        self.pixel_sum = np.sum(self.image)
        self.shape = np.array(self.image.shape)
        if fov is not None:
            self.fov = fov
        else:
            self.pixelscale = pixelscale

        def kernel_func(x, y):
            return np.log(np.sqrt(x**2 + y**2))
        XX, YY = np.meshgrid(self.kernel_grid[0], self.kernel_grid[1])
        kernel = np.log(XX**2 + YY**2)/2
        kernel[self.image.shape[0], self.image.shape[1]] = (-2.12235 + np.log(self.pixelscale[0]*self.pixelscale[1]))/2 # integral of ln(x) over central pixel
        for i in range(10):
            for j in range(10):
                if i == 0 and j == 0:
                    continue
                fill = dblquad(kernel_func, self.pixelscale[0]*(i-0.5), self.pixelscale[0]*(i+0.5), self.pixelscale[0]*(j-0.5), self.pixelscale[0]*(j+0.5))[0] / (self.pixelscale[0]*self.pixelscale[1])
                if i == 0:
                    kernel[self.image.shape[0], self.image.shape[1] + j] = fill
                    kernel[self.image.shape[0], self.image.shape[1] - j] = fill
                elif j == 0:
                    kernel[self.image.shape[0] + i, self.image.shape[1]] = fill
                    kernel[self.image.shape[0] - i, self.image.shape[1]] = fill
                else:
                    kernel[self.image.shape[0] + i, self.image.shape[1] + j] = fill
                    kernel[self.image.shape[0] - i, self.image.shape[1] - j] = fill
                    kernel[self.image.shape[0] - i, self.image.shape[1] + j] = fill
                    kernel[self.image.shape[0] + i, self.image.shape[1] - j] = fill
        self._potential_map = convolve_fft(self.image, kernel, boundary = "fill", fill_value = 0, normalize_kernel = False)/np.pi
        
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
    def mass(self):
        return self["norm"] * self.pixel_sum
    
    @property
    def grid(self):
        return np.array((
            (np.arange(self.image.shape[1]) - (self.image.shape[1]-1)/2)*self.pixelscale[1],
            (np.arange(self.image.shape[0]) - (self.image.shape[0]-1)/2)*self.pixelscale[0],
        ))
    @property
    def kernel_grid(self):
        return np.array((
            (np.arange(2*self.image.shape[1]+1) - self.image.shape[1])*self.pixelscale[1],
            (np.arange(2*self.image.shape[0]+1) - self.image.shape[0])*self.pixelscale[0],
        ))
        
    def _gradient(self, X, Y):
        if self._gradient_sampler is None:
            if self._potential_sampler is None:
                self.potential(0,0)
            self._gradient_sampler = (
                self._potential_sampler.partial_derivative(1,0),
                self._potential_sampler.partial_derivative(0,1),
            )
        return self["norm"] * np.array((self._gradient_sampler[0](X, Y, grid = False), self._gradient_sampler[1](X, Y, grid = False)))
    
    def _laplacian(self, X, Y):
        if self._laplacian_sampler is None:
            if self._gradient_sampler is None:
                self._gradient(0,0)
            
            self._laplacian_sampler = ((
                self._potential_sampler.partial_derivative(2,0),
                self._potential_sampler.partial_derivative(1,1),
            ),(
                self._potential_sampler.partial_derivative(1,1),
                self._potential_sampler.partial_derivative(0,2),
            ))
            
        return self["norm"] * np.array((
            (self._laplacian_sampler[0][0](X, Y, grid = False), self._laplacian_sampler[0][1](X, Y, grid = False)),
            (self._laplacian_sampler[1][0](X, Y, grid = False), self._laplacian_sampler[1][1](X, Y, grid = False)),
        ))
    
    def potential(self, X, Y):
        if self._potential_sampler is None:
            self._potential_sampler = RectBivariateSpline(self.grid[0],self.grid[1],self._potential_map)
        return self._potential_sampler(X, Y, grid = False)

    def alpha(self, X, Y):
        if self._alpha_sampler is None:
            XX, YY = np.meshgrid(self.kernel_grid[0], self.kernel_grid[1])
            RR = (XX**2 + YY**2)
            kernelX = XX / RR
            kernelY = YY / RR
            kernelX[self.image.shape[0],self.image.shape[1]] = 0
            kernelY[self.image.shape[0],self.image.shape[1]] = 0
            self._alpha_sampler = (
                RectBivariateSpline(self.grid[0],self.grid[1],convolve_fft(self.image/np.pi, kernelY/self.pixelscale[1]**2, boundary = "fill", fill_value = 0, normalize_kernel = False, nan_treatment = 'fill')),  
                RectBivariateSpline(self.grid[0],self.grid[1],convolve_fft(self.image/np.pi, kernelX/self.pixelscale[0]**2, boundary = "fill", fill_value = 0, normalize_kernel = False, nan_treatment = 'fill')),
            )
        return self["norm"] * np.array((self._alpha_sampler[0](X, Y, grid = False), self._alpha_sampler[1](X, Y, grid = False)))
    
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
