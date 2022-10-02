from multiplane_core import LensPlane, SourcePlane, ImagePlane, MultiLensPlane
from multiplane import SIE_Lens, Gaussian_Source
import numpy as np
import matplotlib.pyplot as plt

SIE1 = SIE_Lens(q = 0.5, pa = np.pi/4, norm = 1, core = 0.1, x0 = -1)
SIE2 = SIE_Lens(q = 1., pa = 0, norm = 1.2, core = 0, x0 = 0.8, y0 = 0.5)
LP = LensPlane([SIE1, SIE2])
Gauss = Gaussian_Source(norm = 1, sigma = 0.2, x0 = 0, q = 0.6, pa = (np.pi/4 + 0*np.pi/8) % np.pi)
SP = SourcePlane([Gauss])
IP = ImagePlane(shape = (1001, 1001), fov = (10,10))

alpha = LP.alpha(IP.XX, IP.YY)

# plt.imshow(LP.potential(IP.XX, IP.YY), extent = IP.plt_extent, origin = "lower")
# plt.colorbar()
# plt.title("potential")
# plt.show()
# plt.quiver(IP.XX[::32,::32], IP.YY[::32,::32], -alpha[0][::32,::32], -alpha[1][::32,::32])
# plt.title("alpha vectors")
# plt.show()
# plt.imshow(alpha[0], extent = IP.plt_extent, origin = "lower")
# plt.colorbar()
# plt.title("alpha 0")
# plt.show()
# plt.imshow(alpha[1], extent = IP.plt_extent, origin = "lower")
# plt.colorbar()
# plt.title("alpha 1")
# plt.show()
plt.imshow(IP.image(LP,SP), extent = IP.plt_extent, origin = "lower")
plt.colorbar()
#LP.caustics(plt.gca())
plt.title("multi lens")
plt.savefig("multilens.jpg", dpi = 400)
plt.show()

def multiplane_vid(z, i):
    LP1 = LensPlane([SIE1], z = z)
    LP2 = LensPlane([SIE2], z = 1)
    MLP = MultiLensPlane([LP1, LP2])
    Gauss = Gaussian_Source(norm = 1, sigma = 0.2, x0 = 0, q = 0.6, pa = (np.pi/4 + 0*np.pi/8) % np.pi)
    SP = SourcePlane([Gauss], z = 1.5)
    IP = ImagePlane(shape = (1001, 1001), fov = (10,10), z = 0)

    plt.imshow(IP.image(MLP,SP), extent = IP.plt_extent, origin = "lower")
    plt.colorbar()
    #LP.caustics(plt.gca())
    plt.title(f"multi plane, z = {z:.2f}")
    plt.savefig(f"multiplanevid/multiplane_{i:03d}.jpg", dpi = 400)
    plt.close()

for i, z in enumerate(np.linspace(0.1,0.99, 50)):
    multiplane_vid(z, i)
