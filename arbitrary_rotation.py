#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  7 10:22:27 2017

Rotation about an arbitrary axis

@author: rohit
"""

import numpy as np
import math

def rotation_a(axis, v, alpha):
    """
    given the axis to rotate about, and vector to rotate, and amount to rotate (+ccw+): 
    rotate v about axis by alpha
    """
    
    alpha = -alpha
    # curretnly assumes they are unt vectors
    
    
    alpha = np.deg2rad(alpha)
    for i in range(len(axis)):
        axis[i] = float(axis[i])
    
    
    # takes care of cases where a 0 may be in denominator
    if axis == [1, 0, 0] or axis == [-1, 0, 0]:
        theta_x, theta_y = 0, 90*axis[0]
    elif axis == [0, 1, 0] or axis == [0, -1, 0]:
        theta_x, theta_y = 90*axis[1], 90
    elif axis == [0, 0, 1] or axis == [0, 0, -1]:
        theta_x, theta_y = 0, 0
        if axis[2] == -1: 
            theta_y = 180
    else:
        # rotate the crystal such that the axis faces the z axis, rotate about this new z axis, and then rotate the entire crystal back so the axis faces where it originally did.
        theta_x = np.rad2deg(np.arcsin(axis[1]/np.sqrt(axis[1]**2 + axis[2]**2)))
        theta_y = -np.rad2deg(np.arccos(np.sqrt(axis[1]**2 + axis[2]**2)/np.sqrt(axis[0]**2 + axis[1]**2 + axis[2]**2)))
    
    theta_x, theta_y = np.deg2rad(theta_x), np.deg2rad(theta_y)
    
    R_x = [[1, 0, 0],[0, np.cos(theta_x), -np.sin(theta_x)], [0, np.sin(theta_x), np.cos(theta_x)]]
    R_y=  [[np.cos(theta_y), 0, np.sin(theta_y)],[0, 1, 0], [-np.sin(theta_y), 0, np.cos(theta_y)]]
    R_alpha = [[np.cos(alpha), -np.sin(alpha), 0], [np.sin(alpha), np.cos(alpha), 0], [0, 0, 1]]
    R_y_i = np.linalg.inv(R_y)
    R_x_i= np.linalg.inv(R_x)  
    return (np.matmul(R_x_i, np.matmul(R_y_i, np.matmul(R_alpha, np.matmul(R_y, np.matmul(R_x, v))))))    # I apologize for this monstrosity
    
        
def initial(u1, u2, theta, B):
    """
    Finds two reflections (100) and (010) for the crystal so the UB matrix can be calculated
    
    """
    
    # Currently only works for crystals with the 100 facing the z axis. NEED TO FIX THIS SO IT WORKS FOR ANY CRYSTAL 
    phi1 = theta
    chi1 = 90
    if u2[0] == 0 and u2[1] > 0:
        phi2 = 90 + theta
    elif u2[0] == 0 and u2[1] < 0:
        phi2 = -90+theta
    else:
        phi2 = theta + np.rad2deg(np.arctan2(float(u2[1]),u2[0]))
    chi2 = 0
    return [1, 0, 0, 0, 1, 0, 0, chi1, phi1, 0, chi2, phi2, B]
 
        
        
if __name__ == "__main__":
    print(rotation_a([0, 0, 1], [1, 0, 0], 45))
    print(initial([0,1,0], [0,0,1], 17.59, _))
