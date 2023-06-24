import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection, LineCollection, EllipseCollection


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
        An array of size=(num_shapes,7) of all the parameters of the shapes
    """

    params = np.zeros((num_shapes,7))
    params[:,0] = rnd.integers(2,5,size=num_shapes)
    params[:,1:7] = rnd.random((num_shapes,6))
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


def gen_lines(rnd=np.random.default_rng(), num_lines=500, max_line=1/20, min_line=1/80):
    """Generates some random lines aka noise, i.e. the corresponding parameter set

    Parameters
    ----------
    rnd : Generator
        The random generator for position, size and rotation of the lines
    num_lines : int
        Number of lines to be generated
    max_line : float
        Maximum length of a line relative to 1
    min_line : float
        Minimum length of a line relative to 1

    Returns
    -------
    params : ndarray
        An array of size=(num_lines,6) of all the parameters of the lines
    """

    params = rnd.random((num_lines,6))
    params[:,2] = min_line+params[:,2]*(max_line-min_line)
    params[:,2:4] = np.c_[np.multiply(params[:,2],np.cos(params[:,3]*2*np.pi)),np.multiply(params[:,2],np.sin(params[:,3]*2*np.pi))]
    params[:,0] = np.multiply(params[:,0],1-np.abs(params[:,2]))
    params[:,1] = np.multiply(params[:,1],1-np.abs(params[:,3]))
    minus = np.where(params[:,2]<0)
    params[minus,0] = params[minus,0]-params[minus,2]
    minus = np.where(params[:,3]<0)
    params[minus,1] = params[minus,1]-params[minus,3]
    params[:,2] = params[:,0]+params[:,2]
    params[:,3] = params[:,1]+params[:,3]
    return params


def gen_ellipses(rnd=np.random.default_rng(), num_ellipses=500, max_diam=1/10, min_diam=1/80):
    """Generates some random ellipses aka noise, i.e. the corresponding parameter set

    Parameters
    ----------
    rnd : Generator
        The random generator for position, size and rotation of the lines
    num_ellipses : int
        Number of ellipses to be generated
    max_diam : float
        Maximum diameter of an ellipse, relative to 1
    min_diam : float
        Minimum diameter of an ellipse, relative to 1

    Returns
    -------
    params : ndarray
        An array of size=(num_lines,7) of all the parameters of the ellipses
    """

    params = rnd.random((num_ellipses,7))
    params[:,2:4] = min_diam+params[:,2:4]*(max_diam-min_diam)
    cos2 = np.square(np.cos(params[:,4]*np.pi))
    sin2 = np.square(np.sin(params[:,4]*np.pi))
    a2 = np.square(params[:,2]/2)
    b2 = np.square(params[:,3]/2)
    dist0 = np.sqrt(np.multiply(a2,cos2)+np.multiply(b2,sin2))
    dist1 = np.sqrt(np.multiply(a2,sin2)+np.multiply(b2,cos2))
    params[:,0] = dist0 + np.multiply(params[:,0],1-2*dist0)
    params[:,1] = dist1 + np.multiply(params[:,1],1-2*dist1)
    params[:,4] = params[:,4]*180
    return params


def gen_image(shapes, lines, ellipses, im_size=160, max_lw=0.15, min_lw=0.1, min_gray=0.5, show_center=False):
    """Generates an image with geometric shapes and noise on it

    Parameters
    ----------
    shapes : ndarray
        A list of shapes to draw
    lines : ndarray
        A list of lines to draw
    ellipses : ndarray
        A list of ellipses to draw
    im_size : int
        The width and hight of the image (in pixel)
    max_lw : float
        Maximum line width
    min_lw : float
        Minimum line width
    min_gray : float
        Minimum grayscale value, i.e. maximum light gray
    show_center : bool
        Flag for drawing each center of the shapes circumcircles

    Returns
    -------
    img : ndarray
        An array of size=(im_size,im_size) containing the greyscale values of each pixel
    sha : ndarray
        The array of shape parameters relativ to im_size
    lne : ndarray
        The array of lines i.e. line parameters relativ to im_size
    elp : ndarray
        The array of ellipses i.e. ellipse parameters relativ to im_size
    box : ndarray
        The array of enclosing boxes around shapes
    """
    sha = shapes.copy()
    sha[:,1:4] = sha[:,1:4]*im_size
    sha[:,5] = min_lw+sha[:,5]*(max_lw-min_lw)
    
    eps = ellipses.copy()
    eps[:,0:4] = eps[:,0:4]*im_size
    #eps[:,5] = min_lw+eps[:,5]*(max_lw-min_lw)
    ecoll = EllipseCollection(eps[:,2],eps[:,3],eps[:,4],units='x',offsets=eps[:,0:2],linewidths=min_lw, edgecolors='face',facecolors=np.matmul(eps[:,6].reshape((-1,1)),np.ones((1,3)))*min_gray, zorder=1)

    lns = lines.copy()
    lns[:,0:4] = lns[:,0:4]*im_size
    lns[:,4] = min_lw+lns[:,4]*(max_lw-min_lw)
    lcoll = LineCollection(lns[:,0:4].reshape((len(lns),2,2)),linewidths=lns[:,4], colors=np.matmul(lns[:,5].reshape((-1,1)),np.ones((1,3)))*min_gray, zorder=3)

    box = []
    patches = []
    for s in sha:
        if s[0] < 3:
            patch=matplotlib.patches.Circle(s[1:3], radius=s[3], lw=s[5], ec=s[6]*np.ones(3)*min_gray, fill=False)
        else:
            patch=matplotlib.patches.RegularPolygon(s[1:3],numVertices=int(s[0]),radius=s[3],orientation=s[4],lw=s[5], ec=s[6]*np.ones(3)*min_gray, fill=False)
        box.append(patch.get_extents().get_points())
        patches.append(patch)

    if show_center:
        for s in sha:
            patches.append(matplotlib.patches.Circle(s[1:3], radius=.5, lw=2, ec='0'))

    pcoll = PatchCollection(patches, match_original=True, zorder=2)

    images = np.ones((4,im_size,im_size))
    for i in range(4):
        plt.axis('scaled')
        plt.axis('off')
        plt.xlim(0, im_size)
        plt.ylim(0, im_size)
        plt.subplots_adjust(bottom=0.0, left=0.0, right=1.0, top=1.0)
        
        ax = plt.gca()
        ax.add_collection(pcoll)
        if i%2>0: ax.add_collection(ecoll)
        if i>1: ax.add_collection(lcoll)

        fig = plt.gcf()
        fig.set(figwidth=1, figheight=1, dpi=im_size)
        fig.canvas.draw()
        fig.canvas.flush_events()
        img = np.array(fig.canvas.renderer.buffer_rgba())
        images[i,:,:]=img[:,:,0]
        plt.cla()

    return (images,sha,np.reshape(box, (-1,4), order='F'))


def gen_details(shapes_im_size, shapes, images, rnd=np.random.default_rng(), max_fluct=0):
    """Extracts a detail image of each shape from an image

    Parameters
    ----------
    shapes_im_size : int
        The size of the resulting detail images 
    shapes : ndarray
        The list of shapes, i.e. the shape parameters relativ to the size of image
    images : ndarray
        A list of greyscale images containing the shapes
    rnd : Generator
        Generator for random numbers representing the fluctuation of the shapes center
    fluct : int
        maximum fluctuation of the shapes center (in pixels)

    Returns
    -------
    details : ndarray
        A list of list of images of size=(shapes_im_size, shapes_im_size), one image per shape
    """
    details = np.full((len(images),len(shapes),shapes_im_size,shapes_im_size),255)
    for j, image in enumerate(images):
        im_size = len(image)
        radius = shapes_im_size // 2
        mid_point = shapes[:,1:3].astype(int)
        if max_fluct>0:
            mid_point = mid_point+(rnd.integers(0,2*max_fluct+2,size=(len(shapes),2))-(max_fluct+1))
        mid_point = np.c_[im_size-mid_point[:,1],mid_point[:,0]]
        lower_left = mid_point-radius
        upper_right = lower_left+shapes_im_size
        
        for i, s in enumerate(details[j]):
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
