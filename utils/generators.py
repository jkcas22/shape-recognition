import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection, LineCollection


def gen_shapes(rnd=np.random.default_rng(), num_shapes=25,max_radius=1/20,min_radius=1/40, no_rotation=False, no_scaling=False):
    """Generates some random geometric shapes, i.e. the corresponding parameter set

    Parameters
    ----------
    rnd : Generator
        The random generator for position, size, form and rotation of the basic shapes
    num_shapes : int
        Number of shapes to be generated
    max_radius : float
        Maximum circumcircle size relative to 1
    min_radius : float
        Minimum circumcircle size relative to 1
    no_rotation : bool
        Flag for the shapes to be rotated or not
    no_scaling : bool
        Flag for the shapes to be scaled between min and max circumcircle

    Returns
    -------
    params : ndarray
        An array of size=(num_shapes,5) of all the parameters of the shapes
    """

    params = np.zeros((num_shapes,5))
    params[:,0] = rnd.integers(2,5,size=num_shapes)
    params[:,1:5] = rnd.random((num_shapes,4))
    if no_rotation:
        params[:,4] = 0
    if no_scaling:
        params[:,3] = max_radius
    else:
        params[:,3] = min_radius+params[:,3]*(max_radius-min_radius)
    params[:,1] = params[:,3]+np.multiply(params[:,1],1-2*params[:,3])
    params[:,2] = params[:,3]+np.multiply(params[:,2],1-2*params[:,3])
    idx = params[:,0] < 3
    params[idx,4] = 0
    idx = np.logical_not(idx)
    params[idx,4] = np.multiply(params[idx,4],2*np.pi/params[idx,0])
    return params


def gen_noise(rnd=np.random.default_rng(), num_noise=500, max_line=1/20, min_line=1/80):
    """Generates some random lines aka noise, i.e. the corresponding parameter set

    Parameters
    ----------
    rnd : Generator
        The random generator for position, size and rotation of the lines
    num_noise : int
        Number of lines to be generated
    max_line : float
        Maximum length of a line relative to 1
    min_line : float
        Minimum length of a line relative to 1

    Returns
    -------
    params : ndarray
        An array of size=(num_noise,4) of all the parameters of the lines
    """

    params = rnd.random((num_noise,4))
    params[:,2] = min_line+params[:,2]*(max_line-min_line)
    params[:,0] = params[:,2]+np.multiply(params[:,0],1-2*params[:,2])
    params[:,1] = params[:,2]+np.multiply(params[:,1],1-2*params[:,2])
    params[:,2:4] = np.c_[params[:,0]+np.multiply(params[:,2],np.cos(params[:,3]*2*np.pi)),params[:,1]+np.multiply(params[:,2],np.sin(params[:,3]*2*np.pi))]
    return params


def gen_image(shapes, noise = None, rnd=np.random.default_rng(), im_size=160, max_lw=0.15, min_lw=0.1, show_center=False):
    """Generates an image with geometric shapes and noise on it

    Parameters
    ----------
    shapes : ndarray
        A list of shapes to draw
    noise : ndarray
        A list of lines to draw
    rnd : Generator
        Generator for random numbers representing the line width of different shapes
    im_size : int
        The width and hight of the image (in pixel)
    max_line : float
        Maximum line width
    min_line : float
        Minimum line width
    show_center : bool
        Flag for drawing each center of the shapes circumcircles

    Returns
    -------
    img : ndarray
        An array of size=(im_size,im_size) containing the greyscale values of each pixel
    sha : ndarray
        The array of shape parameters relativ to im_size
    nse : ndarray
        The array of noise i.e. line parameters relativ to im_size
    """
    sha = np.c_[shapes.copy(),rnd.random(len(shapes))]
    sha[:,1:4] = sha[:,1:4]*im_size
    sha[:,5] = min_lw+sha[:,5]*(max_lw-min_lw)
    plt.axis('scaled')
    plt.axis('off')
    plt.xlim(0, im_size)
    plt.ylim(0, im_size)
    plt.subplots_adjust(bottom=0.0, left=0.0, right=1.0, top=1.0)
    ax = plt.gca()
    patches = []
    for s in sha:
        if s[0] < 3:
            patches.append(matplotlib.patches.Circle(s[1:3], radius=s[3], lw=s[5], fc='b', fill=False))
        else:
            patches.append(matplotlib.patches.RegularPolygon(s[1:3],numVertices=int(s[0]),radius=s[3],orientation=s[4],lw=s[5],fc='b',fill=False))
    if show_center:
        for s in sha:
            patches.append(matplotlib.patches.Circle(s[1:3], radius=.5, lw=2, fc='b'))
    ax.add_collection(PatchCollection(patches, match_original=True))
    if noise is None:
        nse = None
    else:
        nse = np.c_[noise.copy(),rnd.random(len(noise))]
        nse[:,0:4] = nse[:,0:4]*im_size
        nse[:,4] = min_lw+nse[:,4]*(max_lw-min_lw)
        ax.add_collection(LineCollection(nse[:,0:4].reshape((len(nse),2,2)),linewidths=nse[:,4]))
    fig = plt.gcf()
    fig.set(figwidth=1, figheight=1, dpi=im_size)
    fig.canvas.draw()
    fig.canvas.flush_events()
    img = np.array(fig.canvas.renderer.buffer_rgba())
    plt.cla()
    return (img[:,:,0],sha,nse)


def gen_details(shapes_im_size, shapes, image, rnd=np.random.default_rng(), max_fluct=0):
    """Extracts a detail image of each shape from an image

    Parameters
    ----------
    shapes_im_size : int
        The size of the resulting detail images 
    shapes : ndarray
        The list of shapes, i.e. the shape parameters relativ to the size of image
    image : ndarray
        The greyscale image containing the shapes
    rnd : Generator
        Generator for random numbers representing the fluctuation of the shapes center
    fluct : int
        maximum fluctuation of the shapes center (in pixels)

    Returns
    -------
    details : ndarray
        A list of images of size=(shapes_im_size, shapes_im_size), one image per shape
    """
    im_size = len(image)
    radius = shapes_im_size // 2
    mid_point = shapes[:,1:3].astype(int)
    if max_fluct>0:
        mid_point = mid_point+(rnd.integers(0,2*max_fluct+2,size=(len(shapes),2))-(max_fluct+1))
    mid_point = np.c_[im_size-mid_point[:,1],mid_point[:,0]]
    lower_left = mid_point-radius
    upper_right = lower_left+shapes_im_size
    details = np.full((len(shapes),shapes_im_size,shapes_im_size),255)
    for i, s in enumerate(details):
        x_min = lower_left[i,0]-1
        x_max = upper_right[i,0]-1
        y_min = lower_left[i,1]
        y_max = upper_right[i,1]
        x_min_cut = max(x_min,0)
        x_max_cut = min(x_max,im_size)
        y_min_cut = max(y_min,0)
        y_max_cut = min(y_max,im_size)
        x_min_off = x_min_cut-x_min
        x_max_off = x_max-x_max_cut
        y_min_off = y_min_cut-y_min
        y_max_off = y_max-y_max_cut
        s[x_min_off:(shapes_im_size-x_max_off),y_min_off:(shapes_im_size-y_max_off)] = \
            image[x_min_cut:x_max_cut,y_min_cut:y_max_cut]
    return details
