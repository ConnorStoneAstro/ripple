import numpy as np
import matplotlib.pyplot as plt

def critical_lines(lensplane, ax, show = True, resolution = 1000, fov = 2., **kwargs):
    if isinstance(resolution, int):
        resolution = (resolution, resolution)
    if isinstance(fov, float):
        fov = (fov, fov)
    contour_params = {
        "colors": "r",
        "linestyles": "--",
        "extent": (-fov[1]*lensplane.einstein_radius, fov[1]*lensplane.einstein_radius,-fov[0]*lensplane.einstein_radius, fov[0]*lensplane.einstein_radius)
    }
    contour_params.update(kwargs)
    if not show:
        contour_params["linewidths"] = 0
    XX, YY = np.meshgrid(np.linspace(contour_params["extent"][0], contour_params["extent"][1], resolution[1]), np.linspace(contour_params["extent"][2], contour_params["extent"][3], resolution[0]))
    CS = ax.contour(lensplane.detA(XX, YY), levels = [0.0], **contour_params)
    return CS

def caustics(lensplane, ax, show = True, cl_kwargs = {}, **kwargs):
    cl = critical_lines(lensplane, ax, show = False, **cl_kwargs)
    paths = cl.collections[0].get_paths()
    if len(paths) == 0:
        return
    caustic_paths = []
    for path in paths:
        vertices = path.interpolated(5).vertices
        x1 = np.array(list(float(vs[0]) for vs in vertices))
        x2 = np.array(list(float(vs[1]) for vs in vertices))
        
        a = lensplane.alpha(x1, x2)
        y1 = x1 - a[0]
        y2 = x2 - a[1]
        caustic_params = {"color": "b", "linestyle": "-"}
        if path is paths[0]:
            caustic_params["label"] = "causitcs"
        caustic_params.update(kwargs)
        if not show:
            caustic_params["linewidth"] = 0
        ax.plot(y1, y2, **caustic_params)
        caustic_paths.append((y1,y2))
    return caustic_paths

def lens_kappa(lensplane, ax, fov = (10,10), shape = (1000,1000), **kwargs):

    pixelscale = np.array(fov) / np.array(shape)
    XX, YY = np.meshgrid(
        (np.arange(shape[0]) - (shape[0]-1)/2)*pixelscale[0],
        (np.arange(shape[1]) - (shape[1]-1)/2)*pixelscale[1]
    )
    
    ax.set_title("Lens Kappa (log)")
    ax.imshow(np.log10(lensplane.kappa(XX, YY)), extent = (-fov[0]/2, fov[0]/2, -fov[1]/2, fov[1]/2), origin = "lower", **kwargs)

def lens_potential(lensplane, ax, fov = (10,10), shape = (1000,1000), **kwargs):

    pixelscale = np.array(fov) / np.array(shape)
    XX, YY = np.meshgrid(
        (np.arange(shape[0]) - (shape[0]-1)/2)*pixelscale[0],
        (np.arange(shape[1]) - (shape[1]-1)/2)*pixelscale[1]
    )

    ax.set_title("Lens Potential")
    ax.imshow(lensplane.potential(XX, YY), extent = (-fov[0]/2, fov[0]/2, -fov[1]/2, fov[1]/2), origin = "lower", **kwargs)

def lens_alpha(lensplane, ax1, ax2, fov = (10,10), shape = (1000,1000), **kwargs):

    pixelscale = np.array(fov) / np.array(shape)
    XX, YY = np.meshgrid(
        (np.arange(shape[0]) - (shape[0]-1)/2)*pixelscale[0],
        (np.arange(shape[1]) - (shape[1]-1)/2)*pixelscale[1]
    )

    alpha = lensplane.alpha(XX, YY)
    ax1.set_title("Lens Alpha$_0$")
    ax1.imshow(alpha[0], extent = (-fov[0]/2, fov[0]/2, -fov[1]/2, fov[1]/2), origin = "lower", **kwargs)
    ax2.set_title("Lens Alpha$_1$")
    ax2.imshow(alpha[1], extent = (-fov[0]/2, fov[0]/2, -fov[1]/2, fov[1]/2), origin = "lower", **kwargs)

def lens_gamma(lensplane, ax1, ax2, fov = (10,10), shape = (1000,1000), **kwargs):
    
    pixelscale = np.array(fov) / np.array(shape)
    XX, YY = np.meshgrid(
        (np.arange(shape[0]) - (shape[0]-1)/2)*pixelscale[0],
        (np.arange(shape[1]) - (shape[1]-1)/2)*pixelscale[1]
    )

    gamma = np.arctan(lensplane.gamma(XX, YY))
    kwargs["cmap"] = kwargs.get("cmap", "seismic")
    gamma_extreme = max(np.max(gamma), abs(np.min(gamma)))
    kwargs["vmin"] = kwargs.get("vmin", -gamma_extreme)
    kwargs["vmax"] = kwargs.get("vmax",  gamma_extreme)
    
    ax1.set_title("Lens Gamma$_0$ (arctan)")
    ax1.imshow(gamma[0], extent = (-fov[0]/2, fov[0]/2, -fov[1]/2, fov[1]/2), origin = "lower", **kwargs)
    ax2.set_title("Lens Gamma$_1$ (arctan)")
    ax2.imshow(gamma[1], extent = (-fov[0]/2, fov[0]/2, -fov[1]/2, fov[1]/2), origin = "lower", **kwargs)

def source_distribution(sourceplane, ax, fov = (10,10), shape = (1000,1000), **kwargs):

    pixelscale = np.array(fov) / np.array(shape)
    XX, YY = np.meshgrid(
        (np.arange(shape[0]) - (shape[0]-1)/2)*pixelscale[0],
        (np.arange(shape[1]) - (shape[1]-1)/2)*pixelscale[1]
    )
    ax.set_title("Source Distribution")
    ax.imshow(sourceplane(XX, YY), extent = (-fov[0]/2, fov[0]/2, -fov[1]/2, fov[1]/2), origin = "lower", **kwargs)
