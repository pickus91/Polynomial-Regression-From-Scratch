# Polynomial-Regression-From-Scratch
Polynomial regression from scratch

## Dependencies
* [NumPy](http://www.numpy.org/)
* [Matplotlib](http://matplotlib.org/)
* [Scipy](https://www.scipy.org/)

## Licence
This code is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.

## Synopsis

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

<div>
<img style="float: center;" src="https://github.com/pickus91/Polynomial-Regression-From-Scratch/blob/master/Figures/cost_vs_iterations.png"  height="350" width="400">
</div>
