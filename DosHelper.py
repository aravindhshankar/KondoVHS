import numpy as np
from saddlefinder import *
from Decorators import *
from FeynmanHellman import *
from MoreSources import *
from scipy.optimize import fmin_cg, fmin_tnc, curve_fit, newton, bisect, fsolve
from scipy.misc import derivative as scider
from functools import partial
from scipy.linalg import eigh,eig,eigvalsh
from scipy.special import comb

IN_RADIAN = np.pi/180
vFpar = 4.31074647887324
wpar=0.11



def ret_alpha_beta_Ev(thetai = 1.2, xGuess=(0.025,0.025)):
    ''' Input thetai in degrees
        Also in this case the alpha,beta come inbuilt with the 0.5 factor included
    '''
    # X,Y,Z = data_for_contours(thetai)
    sol = find_saddle(xGuess,thetai*IN_RADIAN,verbose = True)
    
    sadx, sady = sol[0]
    
    Ev = generate_spectrum_Twisted_graphene_single_point(vFpar,wpar,thetai*IN_RADIAN,sadx,sady,symmetric = True)[4]
    
    #kx0,ky0 = rot2D(sol[0],2*np.pi/3)
    kx0,ky0 = sol[0]
    pdvatfixedky = partial(my_decorator(FHpdvs),  vFpar,  wpar, thetai*IN_RADIAN, ky_fixed = ky0,symmetric = True)
    pdvatfixedkx = partial(my_decorator(FHpdvs),  vFpar,  wpar, thetai*IN_RADIAN, kx0,symmetric = True)
    b = scider(ret_first_decorator(pdvatfixedkx),ky0,dx=1e-8,n=1)
    d = scider(ret_second_decorator(pdvatfixedkx),ky0,dx=1e-8,n=1)
    a = scider(ret_first_decorator(pdvatfixedky),kx0,dx=1e-8,n=1)
    c = scider(ret_second_decorator(pdvatfixedky),kx0,dx=1e-8,n=1)
    HessMat = np.array([[a,b],[c,d]])
    EigVals, EigVectors = eig(HessMat)
    EigVals = np.real(EigVals)
    return (0.5*EigVals,Ev,sol)



def smoothstep(x, x_min=0, x_max=1, N=1):
    x = np.clip((x - x_min) / (x_max - x_min), 0, 1)

    result = 0
    for n in range(0, N + 1):
         result += comb(N + n, n) * comb(2 * N + 1, N - n) * (-x) ** n

    result *= x ** (N + 1)

    return result