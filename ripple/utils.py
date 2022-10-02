import numpy as np

def Rotate_Cartesian(X, Y, theta = 0.):
    return X*np.cos(theta) - Y*np.sin(theta), X*np.sin(theta) + Y*np.cos(theta)

def Axis_Ratio_Cartesian(q, X, Y, theta = 0., inv_scale = False):
    """
    Applies the transformation: R(theta) Q R(-theta)
    where R is the rotation matrix and Q is the matrix which scales the y component by 1/q.
    This effectively counter-rotates the coordinates so that the angle theta is along the x-axis
    then applies the y-axis scaling, then re-rotates everything back to where it was.
    """
    if inv_scale:
        scale = (1 / q) - 1
    else:
        scale = q - 1
    ss = 1 + scale * np.sin(theta)**2
    cc = 1 + scale * np.cos(theta)**2
    s2 = scale * np.sin(2*theta)
    return ss*X - s2*Y/2, -s2*X/2 + cc*Y
