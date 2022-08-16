import numpy as np


def my_decorator(f,*args,**kwargs):
    def wrapper(*args,**kwargs):
        res = f(*args,**kwargs)
        return np.array(res)
    return wrapper

def ret_first_decorator(f,*args,**kwargs):
    def wrapper(*args,**kwargs):
        res = f(*args,**kwargs)
        return res[0]
    return wrapper

def ret_second_decorator(f,*args,**kwargs):
    def wrapper(*args,**kwargs):
        res = f(*args,**kwargs)
        return res[1]
    return wrapper