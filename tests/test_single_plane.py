import sys
sys.path.append("/home/connor/Programming/ripple/")
from ripple.plane import LensPlane, SourcePlane, ImagePlane
from ripple.lens import SIE_Lens
from ripple.source import Gaussian_Source
from ripple.plot import caustics, critical_lines
import numpy as np
import matplotlib.pyplot as plt

SIE_L = SIE_Lens(q = 0.7, pa = np.pi/4)
Gaussian_S = Gaussian_Source(sigma = 0.2, x0 = 0.)

LP = LensPlane([SIE_L])
SP = SourcePlane([Gaussian_S])
IP = ImagePlane(shape = (1001, 1001), fov = (5,5))

plt.imshow(LP.potential(IP.XX, IP.YY), extent = IP.plt_extent, origin = "lower")
plt.colorbar()
plt.title("potential")
plt.show()

alpha = LP.alpha(IP.XX, IP.YY)

plt.imshow(alpha[0], extent = IP.plt_extent, origin = "lower")
plt.colorbar()
plt.title("alpha 0")
plt.show()
plt.imshow(alpha[1], extent = IP.plt_extent, origin = "lower")
plt.colorbar()
plt.title("alpha 1")
plt.show()


plt.imshow(IP.image(LP,SP), extent = IP.plt_extent, origin = "lower")
plt.colorbar()
caustics(LP, plt.gca())
plt.title("super basic lens")
plt.show()

SIE1 = SIE_Lens(q = 0.5, pa = np.pi/4, norm = 1, core = 0.1, x0 = -1)
SIE2 = SIE_Lens(q = 1., pa = 0, norm = 1.2, core = 0, x0 = 0.8, y0 = 0.5)
LP = LensPlane([SIE1, SIE2])
Gauss = Gaussian_Source(norm = 1, sigma = 0.2, x0 = 0, q = 0.6, pa = (np.pi/4 + 0*np.pi/8) % np.pi)
SP = SourcePlane([Gauss])
IP = ImagePlane(shape = (1001, 1001), fov = (10,10))

alpha = LP.alpha(IP.XX, IP.YY)

plt.imshow(LP.potential(IP.XX, IP.YY), extent = IP.plt_extent, origin = "lower")
plt.colorbar()
plt.title("potential")
plt.show()
plt.quiver(IP.XX[::32,::32], IP.YY[::32,::32], -alpha[0][::32,::32], -alpha[1][::32,::32])
plt.title("alpha vectors")
plt.show()
plt.imshow(alpha[0], extent = IP.plt_extent, origin = "lower")
plt.colorbar()
plt.title("alpha 0")
plt.show()
plt.imshow(alpha[1], extent = IP.plt_extent, origin = "lower")
plt.colorbar()
plt.title("alpha 1")
plt.show()
plt.imshow(IP.image(LP,SP), extent = IP.plt_extent, origin = "lower")
plt.colorbar()
caustics(LP, plt.gca())
plt.title("multi lens")
plt.show()
