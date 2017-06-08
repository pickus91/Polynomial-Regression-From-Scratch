# -*- coding: utf-8 -*-
"""
Created on Thu Jun  8 13:36:50 2017

@author: picku
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')

def generatePolyPoints(start, stop, num_points, coeff, noiseLevel = 1, plot = 1):
    """
    Generate evenly spaced x and y coordinates in Euclidean space from a polynomial
    with gaussian noise    
    
    Parameters
    ------------
    start : float, 
          Minimum value for the generated x-coordinates
    
    stop  : float
          Maximum value for the generated x-coordinates
    
    num_points : int
               Number of coordinates to generate
    
    coeff : 1-d array, 
    
    noiseLevel : float, optional
              Indicates amount of gaussian noise to add to the y-coordinates
              generated from the polynomial characterized by coeff array. 
              Zero indicates no noise, while each integer increment results in
              ten-fold noise increase
    
    plot : int, values = 1|~1 , optional
        plot == 1 returns a matplotlib plot of the generated coordinates
    
    Returns
    -----------       
    x_pts : 1-d array, shape = [num_points,]     
    y_pts : 1-d array, shape = [num_points,]

    """
    x_pts = np.arange(start, stop, (stop - start)/num_points)
    line = coeff[0]
    
    for i in np.arange(1, len(coeff)):          
        line += coeff[i] * x_pts ** i 

    if noiseLevel > 0:
        y_pts = np.random.normal(-(10 ** noiseLevel), 10 ** noiseLevel, len(x_pts)) + line
    else:
        y_pts = line

    if plot == 1: #Plot option
        plt.figure()
        plt.scatter(x_pts, y_pts)
        plt.xlabel('x')
        plt.ylabel('y')
        
    return x_pts, y_pts
