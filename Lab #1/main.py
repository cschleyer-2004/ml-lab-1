import numpy as np
import matplotlib.pyplot as plt


'''
Instruction: In this assignment, you will implement Linear Regression with the Gra-
dient Descent algorithm and the Normal Equations. Please attach your code and figures
with your submission
'''

'''
1. Let us assume that for a regression problem, we have a training set consisting of 200
examples of (x,y) pair and that we only have one feature X (which could be the size
of a house, for instance). Also, let us assume that the target variable Y (e.g., the
price of a house) is a quadratic function of the input feature X of the form:
y = θ0 + θ1.x + θ2.x2,
where you can assume θ0, θ1 and θ2 are uniformly distributed over the interval [0,1]
and X ∼N(0,1).
Implement the following:
•Generate the dataset with 200 random observations.

•Implement Gradient Descent to obtain estimate of the parameter vector θ =
[θ0,θ1,θ2]T . You can assume that the algorithm has converged when the cost
function J(θ) calculated in successive iterations of the algorithm is not changing
above 10−4, i.e.: |J(t+1)(θ) −J(t)(θ)| < 10−4, where t denotes the number of
iterations of the algorithm. Show the plot showing the final quadratic fit to the
training dataset.

•Also solve for θ directly using Normal Equations and show the plot of the final
quadratic fit to the dataset.

•Compare the two estimates of θ obtained from Gradient Descent and from the
Normal Equations. Check if you get a zero vector when you subtract the two
estimates of θ, i.e., whether θGD −θNE = ⃗0.
'''

#Note: use np.random.randn() -> standard normal
#Note: use np.random.rand() -> Uniform

#Step 1: Generate the dataset with 200 random observations
data = np.random.randn(200, 1)
print(data)
print(f"Data is {len(data)} observations")

#Step 2:Implement Gradient Descent to obtain estimate of the parameter vector θ = [θ0,θ1,θ2]T

#Note: theta_i = theta_i - alpha * sum( (h_theta(x^(j)) - y^(j)) * x_i^(j) )
def gradient_descent(X, y, theta, alpha, max_iter):
    for i in range(max_iter):
        for j in range():
            theta_i = theta_i - alpha * sum( (h_theta(x^(j)) - y^(j)) * x_i^(j) )

#Step 3: solve for θ directly using Normal Equations
# + show the plot of the final quadratic fit to the dataset.

#Step 4: Compare the two estimates (Gradient Descent & Normal Equations)