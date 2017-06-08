# -*- coding: utf-8 -*-
"""
Created on Wed Jun  7 14:17:38 2017

@author: picku
"""

import numpy as np
from scipy import linalg
from collections import OrderedDict
import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')

class PolynomialRegression(object):
        """PolynomialRegression 
        
        Parameters
        ------------
        x_pts : 1-d numpy array, shape = [n_samples,]
        y_pts : 1-d numpy array, shape = [n_samples,]
    
        
        Attributes
        ------------
        theta : 1-d numpy array, shape = [polynomial order + 1,] 
            Ceofficients of fitted polynomial, with theta[0] corresponding
            to the intercept term        
        
        method : str , values = 'normal_equation' | 'gradient_descent'
            Method used for finding optimal values of theta
        
        If gradient descent method is chosen:
        
            costs : 1-d numpy array,
                Cost function values for every iteration of gradient descent
            
            numIters: int
                Number of iterations of gradient descent to be performed
            
        References
        ------------
        https://en.wikipedia.org/wiki/Polynomial_regression
        """

    def __init__(self, x, y):     
        
        self.x = x
        self.y = y      
    
    def standardize(self,data):
        """ Peform feature scaling
        Parameters:
        ------------
        data : numpy-array, shape = [n_samples,]
        
        Returns:
        ---------
        Standardized data                  
        """

        return (data - np.mean(data))/(np.max(data) - np.min(data))
        
    def hypothesis(self, theta, x):
        """ Compute hypothesis, h, where
        h(x) = theta_0*(x_1**0) + theta_1*(x_1**1) + ...+ theta_n*(x_1 ** n)

        Parameters:
        ------------
        theta : numpy-array, shape = [polynomial order + 1,]        
        x : numpy-array, shape = [n_samples,]
        
        Returns:
        ---------
        h(x) given theta values and the training data

        """       
        h = theta[0]
        for i in np.arange(1, len(theta)):
            h += theta[i]*x ** i        
        return h        
        
    def computeCost(self, x, y, theta):
        """ Compute value of cost function J 
        
        Parameters:
        ------------
        x : numpy array, shape = [n_samples,]
        y : numpy array, shape = [n_samples,]
        
        Returns:
        ---------
        Value of cost function J at value theta given the training data
        
        """    
        m = len(y)  
        h = self.hypothesis(theta, x)
        errors = h-y
        
        return (1/(2*m))*np.sum(errors**2) 
        
    def fit(self, method = 'normal_equation', order = 1, tol = 10**-3, numIters = 20, learningRate = 0.01):
        
        """Fit theta to the training data
        
        Parameters
        -----------
        method: string, values = 'normal_equation' | 'gradient_descent'
             Indicates method for which polynomial regression will be performed
            
        order: int, optional
             Order of polynomial fit. Defaults to 1 (linear fit)
             
        numIters: int, optional
             Number of iterations of gradient descent to be performed
            
        learningRate: float, optional
             
        tol : float, optional
            Value indicating the cost value (J(theta)) at which
            gradient descent should terminated. Defaults to 10 ** -3
            
        Returns:
        -----------
        self : object
        
        """

        if method == 'normal_equation': 
            d = {}
            d['x' + str(0)] = np.ones([1,len(x_pts)])[0]    
            for i in np.arange(1, order+1):                
                d['x' + str(i)] = self.x ** (i)        
                
            d = OrderedDict(sorted(d.items(), key=lambda t: t[0]))
            X = np.column_stack(d.values())  

            theta = np.matmul(np.matmul(linalg.pinv(np.matmul(np.transpose(X),X)), np.transpose(X)), self.y)

        elif method == 'gradient_descent':
                
            d = {}
            d['x' + str(0)] = np.ones([1,len(x_pts)])[0]    
            for i in np.arange(1, order+1):                
                d['x' + str(i)] = self.standardize(self.x ** (i))      
                
            d = OrderedDict(sorted(d.items(), key=lambda t: t[0]))
            X = np.column_stack(d.values())  
                
            m = len(self.x)
            theta = np.zeros(order + 1)           
            costs = []
            for i in range(numIters):
             
                h = self.hypothesis(theta, self.x)       
                errors = h-self.y
                theta += -learningRate * (1/m)*np.dot(errors, X)
                cost = self.computeCost(self.x, self.y, theta)
                costs.append(cost)         
                #tolerance check
                if cost < tol:
                    break
                
            self.costs = costs
            self.numIters = numIters
            
        self.method = method    
        self.theta = theta        

        return self
        
    def plot_predictedPolyLine(self):
        """Plot predicted polynomial line using values of theta found
        using normal equation or gradient descent method
        
        Returns
        -----------       
        matploblib figure
        """        
        plt.figure()
        plt.scatter(self.x, self.y, s = 30, c = 'b') 
        line = self.theta[0] #y-intercept 
        label_holder = []
        label_holder.append('%.*f' % (2, self.theta[0]))
        for i in np.arange(1, len(self.theta)):            
            line += self.theta[i] * self.x ** i 
            label_holder.append(' + ' +'%.*f' % (2, self.theta[i]) + r'$x^' + str(i) + '$') 

        plt.plot(self.x, line, label = ''.join(label_holder))        
        plt.title('Polynomial Fit: Order ' + str(len(self.theta)-1))
        plt.xlabel('x')
        plt.ylabel('y') 
        plt.legend(loc = 'best')      

    def plotCost(self):
        """Plot number of gradient descent iterations verus cost function, J,
        values at values of theta
        
        Returns
        -----------       
        matploblib figure
        """        
        if self.method == 'gradient_descent':
            plt.figure()
            plt.plot(np.arange(1, self.numIters+1), self.costs, label = r'$J(\theta)$')
            plt.xlabel('Iterations')
            plt.ylabel(r'$J(\theta)$')
            plt.title('Cost vs Iterations of Gradient Descent')
            plt.legend(loc = 'best')
        else:
            print('plotCost method can only be called when using gradient descent method')
        

        
        
        
