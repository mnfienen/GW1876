import numpy as np
from scipy.interpolate import griddata, NearestNDInterpolator
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable


# get sampled points using nearest neighbor from a grid of data
def sample_from_grid(X,Y,Z,xs,ys):
    '''
    params:
    X and Y:  point locations on the regular grid
    Z: the data at locations X and Y
    xs and ys: the points at which samples are requested
    
    returns:
    zs: the sampled data at locations xs and ys
    '''
    
    # first make an interpolator object for the regular grid
    fullgrid=NearestNDInterpolator((X.ravel(),Y.ravel()),Z.ravel())
    
    # now sample it at the xs and ys locations
    zs = np.array([fullgrid(point) for point in zip(xs.ravel(),ys.ravel())])
    
    return zs
    
# calculate and plot empirical variogram
def plot_empirical_variogram(x,y,data, nbins=25):

    X,Y = np.meshgrid(x,y)
    h = np.sqrt((X-X.T)**2 + (Y-Y.T)**2).ravel()
    d,d1 = np.meshgrid(data, data)
    gam = (1/2*(d-d.T)**2).ravel()
    if nbins>0:
        
        bindiffs = np.ones(nbins)*np.max(h)/2/nbins
        bins = np.hstack(([0],np.cumsum(bindiffs)))
        bindiffs[0] = bins[1]/2
        bincenters = np.cumsum(bindiffs)
    
        empirical_vario = list()
        for i in range(len(bincenters)):
            cinds = np.where((h>bins[i]) & (h<=bins[i+1]))
            empirical_vario.append(np.mean(gam[cinds]))
        h=bincenters
        gam = np.array(empirical_vario)
    fig = plt.figure(figsize=(6,6))
    ax = fig.add_subplot(111)
    plt.scatter(h,gam)
    plt.xlabel('Separation Distance')
    plt.ylabel('Empirical Variogram')
    return h, gam, ax

# scatter plotter for nice-looking plots
def field_scatterplot(x,y,z=None, s=100, xlim=1000, ylim=1000, title=None):
    fig = plt.figure(figsize=(6,6))
    ax = plt.gca()
    if z is not None:
        sc = plt.scatter(x,y,s=s, c=z)
        plt.colorbar(sc)
    
    else:
        plt.scatter(x,y,s=s)
        
    plt.xlim([0,xlim])
    plt.ylim([0,ylim])
    if title is not None:
        plt.title(title)
    ax.set_aspect('equal')
    return ax

def grid_plot(X,Y,Z):
    plt.figure(figsize=(6,6))
    ax = plt.gca()
    im = ax.pcolormesh(X,Y,Z)
    ax.set_aspect('equal')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)    
    plt.colorbar(im, cax=cax)    
    return ax    
    
    