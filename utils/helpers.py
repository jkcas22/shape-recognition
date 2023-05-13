import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg



def gen_grid_image(images, grid):
    """Generates a single image from a list of images arranged in a grid

    Parameters
    ----------
    images : ndarray
        List of images
    grid : tuple
        The grid size (n,m) 
    
    Returns
    -------
    grid_img : ndarray
        A single image with (n,m) detail images on it
    grid_pos : list
        A list of tuples representing all positions (x,y) in the grid (n,m)
    """

    img_size = len(images[0])
    num_grid = grid[0]*grid[1]

    grid_img = np.zeros((1+(img_size+1)*grid[0],1+(img_size+1)*grid[1]))
    grid_pos = []
    for i, img in enumerate(images[:num_grid]):
        x = i//grid[1]
        y = i%grid[1]
        grid_pos.append((x,y))
        grid_img[1+(img_size+1)*x:1+(img_size+1)*x+img_size,1+(img_size+1)*y:1+(img_size+1)*y+img_size] = img
    return (grid_img, grid_pos)


def show_grid_image(images, grid, file):
    """Generates a single image from a list of images arranged in a grid and shows it 

    Parameters
    ----------
    images : ndarray
        List of images
    grid : tuple
        The grid size (n,m)
    file : string
        path to temporary location of image
    
    Returns
    -------
    grid_pos : list
        A list of tuples representing all positions (x,y) in the grid (n,m)
    """

    grid_image, grid_pos = gen_grid_image(images, grid)
    
    fig = plt.figure(figsize=(grid_image.shape[1],grid_image.shape[0]),dpi=1)
    fig.figimage(grid_image, cmap='gray')
    fig.savefig(file)
    plt.close()
    plt.imshow(mpimg.imread(file))
    return grid_pos