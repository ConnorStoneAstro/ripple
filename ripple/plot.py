import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import ListedColormap

def critical_lines(lensplane, ax, show = True, resolution = 1000, fov = 2., **kwargs):
    print("fov", fov)
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

def lens_alpha(lensplane, ax1, ax2, fov = (10,10), shape = (1000,1000), ax3 = None, **kwargs):

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
    if not ax3 is None:
        ax3.set_title("Lens |Alpha|")
        ax3.imshow(np.sqrt(alpha[0]**2 + alpha[1]**2), extent = (-fov[0]/2, fov[0]/2, -fov[1]/2, fov[1]/2), origin = "lower", **kwargs)

def lens_gamma(lensplane, ax1, ax2, fov = (10,10), shape = (1000,1000), ax3 = None, **kwargs):
    
    pixelscale = np.array(fov) / np.array(shape)
    XX, YY = np.meshgrid(
        (np.arange(shape[0]) - (shape[0]-1)/2)*pixelscale[0],
        (np.arange(shape[1]) - (shape[1]-1)/2)*pixelscale[1]
    )

    gamma = lensplane.gamma(XX, YY)
    atgamma = np.arctan(gamma)
    kwargs["cmap"] = kwargs.get("cmap", "seismic")
    atgamma_extreme = max(np.max(atgamma), abs(np.min(atgamma)))
    kwargs["vmin"] = kwargs.get("vmin", -atgamma_extreme)
    kwargs["vmax"] = kwargs.get("vmax",  atgamma_extreme)
    
    ax1.set_title("Lens Gamma$_0$ (arctan)")
    ax1.imshow(atgamma[0], extent = (-fov[0]/2, fov[0]/2, -fov[1]/2, fov[1]/2), origin = "lower", **kwargs)
    ax2.set_title("Lens Gamma$_1$ (arctan)")
    ax2.imshow(atgamma[1], extent = (-fov[0]/2, fov[0]/2, -fov[1]/2, fov[1]/2), origin = "lower", **kwargs)
    if not ax3 is None:
        ax3.set_title("Lens log|Gamma|")
        ax3.imshow(np.log10(np.sqrt(gamma[0]**2 + gamma[1]**2)), extent = (-fov[0]/2, fov[0]/2, -fov[1]/2, fov[1]/2), origin = "lower", **kwargs)

def source_distribution(sourceplane, ax, fov = (10,10), shape = (1000,1000), **kwargs):

    pixelscale = np.array(fov) / np.array(shape)
    XX, YY = np.meshgrid(
        (np.arange(shape[0]) - (shape[0]-1)/2)*pixelscale[0],
        (np.arange(shape[1]) - (shape[1]-1)/2)*pixelscale[1]
    )
    ax.set_title("Source Distribution")
    ax.imshow(sourceplane(XX, YY), extent = (-fov[0]/2, fov[0]/2, -fov[1]/2, fov[1]/2), origin = "lower", **kwargs)

def rays_3D(imageplane, lensplane, sourceplane, ax3d, units = "angular", Nrays = 25, samples = None, surface_kwargs = {}, ray_kwargs = {}):
    # fixme update from multiplane version
    ax3d.set_title("Rays")

    image = imageplane.image(lensplane, sourceplane)

    # sample rays using image as probability distribution
    if samples is None:
        flat = image.flatten()
        samples = np.random.choice(a=flat.size, p=flat/np.sum(flat), size = Nrays)
        samplesY, samplesX = np.unravel_index(samples, image.shape)
        samplesX = (samplesX - (image.shape[0]-1)/2) * imageplane.pixelscale[0]
        samplesY = (samplesY - (image.shape[1]-1)/2) * imageplane.pixelscale[1]
    else:
        samplesX, samplesY = samples[0], samples[1]

    # Propogate rays back to source
    alpha = lensplane.alpha(samplesX, samplesY)
    finalX = samplesX - alpha[0]
    finalY = samplesY - alpha[1]

    Dimage = imageplane.dM(0, imageplane.z)
    Dlens = imageplane.dM(0, lensplane.z)
    Dsource = imageplane.dM(0, sourceplane.z)

    DAimage = 0. if units == "physical" else 1.
    DAlens = imageplane.dA(0, lensplane.z) if units == "physical" else 1.
    DAsource = imageplane.dA(0, sourceplane.z) if units == "physical" else 1.

    surface_kwargs["alpha"] = surface_kwargs.get("alpha", 0.3)
    # Plot the image plane
    if units == "angular":
        ax3d.plot_surface(
            imageplane.XX, np.zeros(image.shape) + Dimage, imageplane.YY,
            facecolors = mpl.cm.inferno(image/np.max(image)), zorder = 5, **surface_kwargs
        )

    # Plot the lens plane
    kappa = np.log10(lensplane.kappa(imageplane.XX, imageplane.YY))
    ax3d.plot_surface(
        imageplane.XX*DAlens, np.zeros(image.shape)+Dlens, imageplane.YY*DAlens,
        facecolors = mpl.cm.inferno(kappa/np.max(kappa)), zorder = 3, **surface_kwargs
    )

    # Plot the source plane
    source = sourceplane(imageplane.XX, imageplane.YY)
    ax3d.plot_surface(
        imageplane.XX*DAsource, np.zeros(image.shape)+Dsource, imageplane.YY*DAsource,
        facecolors = mpl.cm.inferno(source/np.max(source)), zorder = 1, **surface_kwargs
    )

    # Plot the rays
    for i in range(Nrays):
        ax3d.plot3D([samplesX[i]*DAimage,samplesX[i]*DAlens], [Dimage,Dlens],[samplesY[i]*DAimage,samplesY[i]*DAlens], color = "r", linewidth = 0.5, zorder = 4)
        ax3d.plot3D([samplesX[i]*DAlens,finalX[i]*DAsource], [Dlens,Dsource],[samplesY[i]*DAlens,finalY[i]*DAsource], color = "r", linewidth = 0.5, zorder = 2)
        
    ax3d.view_init(20, -20)

    return samplesX, samplesY, finalX, finalY

def rays_3D_multiplane(imageplane, multilensplane, sourceplane, ax3d, units = "angular", Nrays = 25, lens_display = "kappa", samples = None, cmap = None, view_init = (20,-20), surface_kwargs = {}, ray_kwargs = {}):

    ax3d.set_title("Rays")

    image = imageplane.image(multilensplane, sourceplane)

    if cmap is None:
        cmap = mpl.cm.inferno

    # sample rays using image as probability distribution
    if samples is None:
        flat = image.flatten()
        flat -= np.min(flat)
        samples = np.random.choice(a=flat.size, p=flat/np.sum(flat), size = Nrays)
        samplesY, samplesX = np.unravel_index(samples, image.shape)
        samplesX = (samplesX - (image.shape[0]-1)/2) * imageplane.pixelscale[0]
        samplesY = (samplesY - (image.shape[1]-1)/2) * imageplane.pixelscale[1]
    else:
        samplesX, samplesY = samples[0], samples[1]
    Nrays = len(samplesX)

    Dimage = imageplane.dM(0, imageplane.z)
    Dsource = imageplane.dM(0, sourceplane.z)

    DAimage = 0. if units == "physical" else 1.
    DAsource = imageplane.dA(0, sourceplane.z) if units == "physical" else 1.

    surface_kwargs["rcount"] = surface_kwargs.get("rcount", 300)
    surface_kwargs["ccount"] = surface_kwargs.get("ccount", 300)
    tmpalpha = surface_kwargs.get("alpha", 0.1)
    surface_kwargs["alpha"] = 1.

    face_forward = (view_init[1] % 360) > 180
    
    # Plot the image plane
    if units == "angular":
        ax3d.plot_surface(
            imageplane.XX, np.zeros(image.shape) + Dimage, imageplane.YY,
            facecolors = cmap(image/np.max(image)), zorder = (2 + 2*len(multilensplane)) if face_forward else 0, **surface_kwargs
        )
    
    del surface_kwargs["alpha"]
    mod_cmap = cmap(np.arange(cmap.N))
    clarity = np.ones(cmap.N) * tmpalpha #np.linspace(0, 1, cmap.N)
    clarity[0] = 0
    #clarity = np.sqrt(clarity)
    mod_cmap[:, -1] = clarity
    mod_cmap = ListedColormap(mod_cmap)
    #surface_kwargs["alpha"] = tmpalpha
    # Plot the source plane
    source = sourceplane(imageplane.XX, imageplane.YY)
    ax3d.plot_surface(
        imageplane.XX*DAsource, np.zeros(image.shape)+Dsource, imageplane.YY*DAsource,
        facecolors = mod_cmap(source/np.max(source)), zorder = 0 if face_forward else (2 + 2*len(multilensplane)), **surface_kwargs
    )

    surface_kwargs["rcount"] = 200 #surface_kwargs.get("rcount", 200)
    surface_kwargs["ccount"] = 200 #surface_kwargs.get("ccount", 200)
    my_cmap = cmap(np.arange(cmap.N))
    clarity = np.linspace(0, 1, cmap.N)
    clarity[clarity < 0.2] = 0
    #clarity = np.sqrt(clarity)
    my_cmap[:, -1] = clarity
    my_cmap = ListedColormap(my_cmap)

    for ilp, lensplane in enumerate(multilensplane):
        # Plot the lens plane
        Dlens = imageplane.dM(0, lensplane.z)
        DAlens = imageplane.dA(0, lensplane.z) if units == "physical" else 1.
        if lens_display == "kappa":
            lens_repr = np.log10(lensplane.kappa(imageplane.XX, imageplane.YY))
        elif lens_display == "potential":
            lens_repr = -lensplane.potential(imageplane.XX, imageplane.YY)
        lens_repr -= np.min(lens_repr[np.isfinite(lens_repr)])
        ax3d.plot_surface(
            imageplane.XX*DAlens, np.zeros(image.shape)+Dlens, imageplane.YY*DAlens,
            facecolors = my_cmap(lens_repr/np.max(lens_repr)), zorder = (2*len(multilensplane) - 2*ilp) if face_forward else (2 + 2*ilp), **surface_kwargs
        )

    alphas = multilensplane.alpha_recursive(0., multilensplane.dM(imageplane.z, multilensplane.planes[0].z) * np.array([samplesX, samplesY]), imageplane, sourceplane, alphas = [])
    
    # Plot the rays
    Dlens = imageplane.dM(0, multilensplane.planes[0].z)
    DAlens = imageplane.dA(0, multilensplane.planes[0].z) if units == "physical" else 1.
    for i in range(Nrays):
        ax3d.plot3D([samplesX[i]*DAimage,samplesX[i]*DAlens], [Dimage,Dlens],[samplesY[i]*DAimage,samplesY[i]*DAlens], color = "r", zorder = (1 + 2*len(multilensplane)) if face_forward else 1, linewidth = 0.5)
        
    for ilp, lensplane in enumerate(multilensplane):
        if ilp == 0:
            startX = samplesX
            startY = samplesY
        else:
            startX = alphas[ilp-1][0]
            startY = alphas[ilp-1][1]
        Dlens1 = imageplane.dM(0, multilensplane.planes[ilp].z)
        DAlens1 = imageplane.dA(0, multilensplane.planes[ilp].z) if units == "physical" else 1.
        if ilp + 1 == len(multilensplane):
            Dlens2 = imageplane.dM(0, sourceplane.z)
            DAlens2 = imageplane.dA(0, sourceplane.z) if units == "physical" else 1.
        else:
            Dlens2 = imageplane.dM(0, multilensplane.planes[ilp+1].z)
            DAlens2 = imageplane.dA(0, multilensplane.planes[ilp+1].z) if units == "physical" else 1.
        for i in range(Nrays):
            ax3d.plot3D([startX[i]*DAlens1,alphas[ilp][0][i]*DAlens2], [Dlens1,Dlens2],[startY[i]*DAlens1,alphas[ilp][1][i]*DAlens2], color = "r", zorder = (2*len(multilensplane) - 1 - 2*ilp) if face_forward else (3 + 2*ilp), linewidth = 0.5)
        
    ax3d.view_init(*view_init)

    return samplesX, samplesY, alphas[-1][0], alphas[-1][1]
