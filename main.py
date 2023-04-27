
import numpy as np
import matplotlib.pyplot as plt

def sigmoid(z):
    """
    Computes the sigmoid function.

    Parameters:
    -----------
    z : numpy array of shape (n,)
        The input values.

    Returns:
    --------
    output : numpy array of shape (n,)
        The sigmoid values.
    """
    return 1 / (1 + np.exp(-z))

def log_likelihood(theta, X, y):
    """
    Computes the log-likelihood function for logistic regression.

    Parameters:
    -----------
    theta : numpy array of shape (n,)
        The parameter vector.
    X : numpy array of shape (m, n)
        The training data.
    y : numpy array of shape (m,)
        The corresponding labels for the training data.

    Returns:
    --------
    ll : float
        The value of the log-likelihood function.

    """
    z = np.dot(X, theta)
    ll = np.sum(y*np.log(sigmoid(z)) + (1-y)*np.log(1-sigmoid(z)))
    return ll

def gradient(theta, X, y):
    """
    Computes the gradient of the log-likelihood function for logistic regression.

    Parameters:
    -----------
    theta : numpy array of shape (n,)
        The parameter vector.
    X : numpy array of shape (m, n)
        The training data.
    y : numpy array of shape (m,)
        The corresponding labels for the training data.

    Returns:
    --------
    grad : numpy array of shape (n,)
        The gradient of the log-likelihood function with respect to theta.

    """
    z = np.dot(X, theta)
    error = sigmoid(z) - y
    grad = np.dot(X.T, error)
    return grad

def gradient_descent(X, y, alpha=0.01, iterations=1000):
    """
    Implements the gradient descent algorithm for logistic regression.

    Parameters:
    -----------
    X : numpy array of shape (m, n)
        The training data.
    y : numpy array of shape (m,)
        The corresponding labels for the training data.
    alpha : float, optional (default=0.01)
        The learning rate for the gradient descent algorithm.
    iterations : int, optional (default=1000)
        The number of iterations to run the gradient descent algorithm.

    Returns:
    --------
    theta : numpy array of shape (n,)
        The learned parameter vector.

    """
    m, n = X.shape
    theta = np.zeros(n)
    for i in range(iterations):
        grad = gradient(theta, X, y)
        theta = theta - alpha * grad
        ll = log_likelihood(theta, X, y)
        if i % 100 == 0:
            print(f"Iteration {i}: Log-Likelihood = {ll}")
    return theta

def logistic_regression_newton(X, y, tol=1e-6, max_iter=100):
    """
    Implements Newton's method to maximize the log-likelihood function for logistic regression.

    Parameters:
    -----------
    X: array-like, shape (n_samples, n_features)
        The training input samples.
    y: array-like, shape (n_samples,)
        The target values.
    tol: float, default=1e-6
        Tolerance for stopping criterion.
    max_iter: int, default=100
        Maximum number of iterations to perform.

    Returns:
    --------
    theta: array, shape (n_features,)
        The optimal weights.
    n_iter: int
        The number of iterations performed.
    """

    # Initialize the weights to zeros
    theta = np.zeros(X.shape[1])

    # Loop until convergence or maximum iterations reached
    for i in range(max_iter):

        # Compute the sigmoid function
        z = np.dot(X, theta)
        g = 1.0 / (1.0 + np.exp(-z))

        # Compute the gradient and Hessian
        gradient = np.dot(X.T, y - g)
        hessian = np.dot(X.T * g * (1 - g), X)

        # Update the weights
        delta = np.linalg.solve(hessian, gradient)
        theta += delta

        # Check for convergence
        if np.linalg.norm(delta) < tol:
            break

    # Return the optimal weights and the number of iterations performed
    return theta, i+1

def main():
    print('Logistic Regression in Python')

    df_X = []
    df_Y = []

    with open('hwX.txt') as X:
        for x in X:
            x = x.strip()

            line = x.split(' ')

            if(len(line) == 4):
                df_X.append([float(line[0]),float(line[3])])
            else:
                df_X.append([float(line[0]),float(line[2])])
    
    with open('hwY.txt') as Y:
        for y in Y:
            y = y.strip()
            df_Y.append(float(y))
            
    df_X = np.array(df_X)
    df_Y = np.array(df_Y)


    gd_theta = gradient_descent(df_X,df_Y)
    print()
    print('Results of Gradient Descent: ')
    print('Thetas: ' + str(gd_theta))
    theta, epocs = logistic_regression_newton(df_X,df_Y)

    print()
    print('Results of Newton\'s Method: ')
    print('Thetas: '+ str(theta))
    print('Epocs: ' + str(epocs))

    # Generate a scatter plot of the training data
    plt.scatter(df_X[df_Y==1][:,0], df_X[df_Y==1][:,1], marker='o', label='Class 1')
    plt.scatter(df_X[df_Y==0][:,0], df_X[df_Y==0][:,1], marker='s', label='Class 0')

    # Plot decision boundary
    x1_vals = np.linspace(np.min(df_X[:,0]), np.max(df_X[:,0]), 100)
    x2_vals = -(theta[0] + theta[1]*x1_vals) / theta[1]
    plt.plot(x1_vals, x2_vals, label='Newton Decision Boundary')

     # Plot decision boundary
    gd_x1_vals = np.linspace(np.min(df_X[:,0]), np.max(df_X[:,0]), 100)
    gd_x2_vals = -(gd_theta[0] + gd_theta[1]*gd_x1_vals) / gd_theta[1]
    plt.plot(gd_x1_vals, gd_x2_vals, label='Standard Decision Boundary')

    # Set x and y limits
    plt.xlim(-10, 10)
    plt.ylim(-10, 10)

    # Add labels and title
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('Training Data')

    # Add legend
    plt.legend(loc='upper right')

    # Display the plot
    plt.show()

if __name__ == '__main__':
    main()