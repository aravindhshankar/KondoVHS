import numpy as np
from SourcesTBG import *

#TO_DEGREE = 1.0/0.018326
IN_RADIAN = np.pi/180
vFpar = 4.31074647887324
wpar=0.11
BANDS = 8

def pdvx(x,y, thetai=1.05, hstart = 2**-4): 
    return derivative(lambda x: generate_spectrum_Twisted_graphene_single_point(vF=vFpar, w=wpar, Theta=thetai*IN_RADIAN, 
                                               kx_fixed = x, ky_fixed=y, symmetric=True)[4], x, hstart = hstart)
def pdvy(x,y,thetai = 1.05,hstart = 2**-4):
    return derivative(lambda y: generate_spectrum_Twisted_graphene_single_point(vF=vFpar, w=wpar, Theta=thetai*IN_RADIAN, 
                                               kx_fixed = x, ky_fixed=y, symmetric=True)[4], y, hstart = hstart)

def funcfx(xG,thetai): 
    kxval,kyval = xG
    return np.abs(pdvx(kxval,kyval,thetai = thetai))

def funcfy(xG,thetai): 
    kxval,kyval = xG
    return np.abs(pdvy(kxval,kyval,thetai = thetai))

def funcfadd(xG,thetai):
    kxval,kyval = xG
    return 1e6*(np.abs(pdvx(kxval,kyval,thetai = thetai)) + np.abs(pdvy(kxval,kyval,thetai = thetai)))

def funcfsquare(xG,thetai):
    kxval,kyval = xG
    return 1e6*((pdvx(kxval,kyval,thetai = thetai))**2 + (pdvy(kxval,kyval,thetai = thetai))**2)

def funcfsquareNO(xG,thetai):
    kxval,kyval = xG
    return 1.0*((pdvx(kxval,kyval,thetai = thetai))**2 + (pdvy(kxval,kyval,thetai = thetai))**2)