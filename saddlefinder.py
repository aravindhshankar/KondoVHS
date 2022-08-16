import numpy as np
from SourcesTBG import *
from FeynmanHellman import *
from scipy.optimize import fmin_cg, fmin_tnc, curve_fit, newton, bisect, fsolve

def rot2D(vec, theta): 
    mat = np.array([[np.cos(theta), -1.0*np.sin(theta)],[np.sin(theta), np.cos(theta)]])
    return mat@vec


def find_saddle(xGuess, Theta = 1.05 * IN_RADIAN, verbose = False):
    '''
    supply Theta as thetai * IN_RADIAN
    xGuess is a tuple (kx,ky)
    '''
    newtckadd = fmin_tnc(FHfuncfsquare, xGuess, fprime=None, 
                         args=(vFpar, wpar, Theta), approx_grad=True, disp = 5)
    if verbose:
        return newtckadd
    else:
        return newtckadd[0]
    

def data_for_contours(thetai=1.05, X=None, Y=None):
    '''Specify thetai  in degrees'''
    if (not (np.any(X) or np.any(Y))):
        X = np.linspace(-0.05, 0.05, 400)
        Y = np.linspace(-0.05, 0.05, 400)
    
    Zspec = np.array([[generate_spectrum_Twisted_graphene_single_point(vF=vFpar, w=wpar, Theta=thetai*IN_RADIAN,
                                                       kx_fixed = kxi, ky_fixed=kyi, symmetric=True)[4] 
                       for kxi in X] for kyi in Y])
    return (X,Y,Zspec)


def transformer(px,py,kx0,ky0,rotang, first = 'rotation'):
    '''rotang should be in radians'''
    if first == 'shift':
        kx = px + kx0
        ky = py + ky0
        return rot2D([kx,ky],rotang)
    elif first == 'rotation':
        vec = rot2D([px,py],rotang)
        return vec+np.array([kx0,ky0])
    else: 
        raise(Exception('Invalid Choice of first'))
        


def returnpoly(px,py,coeffs):
    n = len(coeffs)-1
    polyval = 0
    for i,coeffval in enumerate(coeffs):
        polyval += (px**(n-i)) * (py**i) * coeffval
    return polyval