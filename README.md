 # Polynomial-Regression-From-Scratch
Polynomial regression using the normal equation and gradient descent methods.

## Dependencies
* [NumPy](http://www.numpy.org/)
* [Matplotlib](http://matplotlib.org/)
* [Scipy](https://www.scipy.org/)

## Licence
This code is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.

## Synopsis

The goal of polynomial regression is to fit a *nth* degree polynomial to data to establish a general relationship between the independent variable *x* and dependent variable *y*. Polynomial regression is a special form of multiple linear regression, in which the objective is to minimize the cost function given by:
<div align = "center">
<img style="float: center;" src="https://github.com/pickus91/Polynomial-Regression-From-Scratch/blob/master/Figures/cost_function.PNG"  height="100" width="300">
</div>

and the hypothesis is given by the linear model:
<div align = "center">
<img style="float: center;" src="https://github.com/pickus91/Polynomial-Regression-From-Scratch/blob/master/Figures/hypothesis.PNG"  height="165" width="465">
</div>

The ```PolynomialRegression``` class can perform polynomial regression using two different methods: the normal equation and gradient descent. The normal equation method uses the closed form solution to linear regression:
<div align = "center">
<img style="float: center;" src="https://github.com/pickus91/Polynomial-Regression-From-Scratch/blob/master/Figures/normal_EQ.PNG"  height="110" width="190">
</div>
and does not require iterative computations or feature scaling. Gradient descent is an iterative approach that increments theta according to the direction of the gradient of the cost function.

### Code Example 1: Normal Equation Method

```
x_pts, y_pts = generatePolyPoints(0, 50, 100, [5, 1, 1], 
                                  noiseLevel = 2, plot = 1)
PR = PolynomialRegression(x_pts, y_pts)
theta = PR.fit(method = 'normal_equation', order = 2)
PR.plot_predictedPolyLine()
```
<div>
<ul>        
<img style="float: left;" src="https://github.com/pickus91/Polynomial-Regression-From-Scratch/blob/master/Figures/sample_data.png"  height="350" width="400">
<img style="float: right;" src="https://github.com/pickus91/Polynomial-Regression-From-Scratch/blob/master/Figures/normalEQ_polyFit.png"  height="350" width="400">
 </ul>
</div>

### Code Example 2: Gradient Descent Method

```
x_pts, y_pts = generatePolyPoints(0, 50, 100, [5, 1, 1], 
                                  noiseLevel = 2, plot = 1)
PR = PolynomialRegression(x_pts, y_pts)
theta = PR.fit(method = 'gradient_descent',  order = 2, tol = 10**-3, numIters = 100, learningRate = 10**-4)
PR.plot_predictedPolyLine()
PR.plotCost()
```
<div>
<ul>        
<img style="float: left;" src="https://github.com/pickus91/Polynomial-Regression-From-Scratch/blob/master/Figures/sample_data.png"  height="350" width="400">
<img style="float: center;" src="https://github.com/pickus91/Polynomial-Regression-From-Scratch/blob/master/Figures/gradientDescent_polyFit.png"  height="350" width="400">
 </ul>
</div>


<div align = "center">
<img style="float: center;" src="https://github.com/pickus91/Polynomial-Regression-From-Scratch/blob/master/Figures/cost_vs_iterations.png"  height="350" width="400">
</div>
