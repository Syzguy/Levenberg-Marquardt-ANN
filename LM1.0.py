import numpy as np
import scipy.special as sp
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from copy import deepcopy as dc
import decimal as dec
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error as mse

def cross_validate_regular(rng, step, net, X, Y, verbose=True):
    """
    Returns the optimal regularization term for training the network defined
    by net.
    :param rng: float
    :param step: float
    :param net: List
    :param X: np.ndarray
    :param Y: np.ndarray
    """
    
    # Extracting seperate cross-validation training and testing data sets.
    split = int(len(X)*0.7)
    
    X_train = X[:split]
    Y_train = Y[:split]
    
    X_test = X[split:]
    Y_test = Y[split:]

    # Storing the RSS's in an array.
    regs = np.arange(0.0, rng, step)
    rss = np.zeros((int(rng/step), ), dtype='float64')
    
    # Initializing the network.
    W, B = build(X_train, net)    
    
    # Cross-validating.
    i = 0
    for reg in regs:
        
        # Training the network.   
        Ws, Bs, rms_per_epoch = train(X_train, Y_train, W, B, reg, verbose=False)
        
        # Evaluating results.
        P = propForward(X_test, Ws, Bs)[0].flatten().astype('float64')
        RSS = np.sum((P-Y_test)**2)
        rss[i] = RSS
        i += 1
        
        # Reporting on the step.
        if verbose:
            print 'Lambda: %.2f,    RSS: %.4f' % (reg, RSS)
    
    return regs[np.argmin(rss)], rss

def pca(X, tau=0.01):
    """
    Returns the matrix of eigenvectors with the respect to the covariance
    matrix of X.
    
    :param X: np.array
    :param tau: float
    """
    
    # Calculating the covariance matrix.
    rows, cols = X.shape
    cov = np.zeros((cols, cols), dtype='float64')
    for i in np.arange(rows):
        sample = X[i].reshape(cols, 1)
        cov += np.dot(sample, sample.T) / rows
    
    # Computing the eigenvectors and eigenvalues.
    evals, evecs = np.linalg.eig(cov)   
    
    # Keeping features.
    U = evecs[:, (evals/np.sum(evals))>=tau]
    
    return U       

def wsumsqr(W, B):
    """
    Sums all the values for the weights and the biases.
    
    :param W: List
    :param B: List
    """
    
    total = 0
    for i in np.arange(len(W)):
        total += np.sum(W[i]**2)
        total += np.sum(B[i]**2)
    
    return total

def jacobian(m, WU, D):
    """
    Constructs a Jacobian matrix.
    
    :param m: int
    :param WU: List
    :param D: List
    """
    
    j = np.empty((m, 0), dtype='float64')
    # Constructing the Jacobian.
    for i in np.arange(len(WU)):
        z, y, x = WU[i].shape
        wu = WU[i].reshape(z, y*x)
        j = np.concatenate((j, wu, D[i]), axis=1)

    return j
    
def up(U, W, B):
    """
    Recontructs the List of 2-d arrays from a 1-d array.
    
    :param U: np.array
    :param W: List
    :param B: List
    """
    U = U.flatten()
    nW = []
    nB = []
    off = 0
    for i in np.arange(len(W)):
        y, x = W[i].shape
        nW.append(
            U[off:off + (y*x)].reshape(y,x)
            )
        off += y*x
        
        v = B[i].shape[0]
        nB.append(
            U[off: off+v].reshape(v,)
            )
        off += v
    
    return nW, nB
    
def build(X, net):
    """
    Initializes weights and biases for neural network.
    
    :param net: List
    """
    
    W = []  # List of weights.
    B = []  # List of Biases.    
    
    m, n = X.shape
    net.insert(0, n)
    np.random.seed(0)
    
    for i in np.arange(1, len(net)):
        W.append(
            np.random.rand(net[i-1], net[i]) * np.sqrt(2.0 / net[i-1])
            )
        B.append(
            np.zeros((net[i], ), dtype='float64')
            )
    
    return W, B
    
def propForward(X, W, B):
    """
    Propogates the input the through the network.

    :param X: np.array
    :param W: List
    :param B: List
    """

    # Forward Propogation.        
    N = [X]
    for i in np.arange(len(W)-1):
        N.append(
            # sp.expit(np.dot(N[-1], W[i]) + B[i])
            np.tanh(np.dot(N[-1], W[i]) + B[i])
            )
    P = np.dot(N[-1], W[-1]) + B[-1]
    
    return P, N
    
def propBack(P, Y, N, W, B, m, phi):
    """
    Propogates the error through the network.
    
    :param P: np.array
    :param Y: np.array
    :param m: Int
    """
    
    # Back Propogation.
    prime = lambda x: 1 - (x**2)
    # prime = lambda x: x * (1-x)
    D = [1 / (P - Y)]
    for i in np.arange(len(W)-1, 0, -1):
        D.append(
            prime(N[i]) * np.dot(D[-1], W[i].T)
            )
    D.reverse()
    
    WU = []        
    for i in np.arange(len(N)-1, -1, -1):
        wu = np.zeros((m, N[i].shape[1], D[i].shape[1]))
        for j in np.arange(m):
            wu[j] = np.dot(
                N[i][j].reshape(len(N[i][j]), 1),
                D[i][j].reshape(1, len(D[i][j]))
                )

        WU.append(wu)
    WU.reverse()
    
    for i in np.arange(len(W)):
        WU[i] += (phi * W[i])
        D[i] += (phi * B[i]) 
    
    return WU, D
 
def error_dec(P, Y, W, B, phi, prec):
    """
    Returns the error difference to prec deciaml places.
    
    :param P: np.array
    :param Y: np.array
    :param prec: Int
    """

    # Flattening arrays.
    p = np.float64(P.flatten())
    y = np.float64(Y.flatten())

    # Calculating Mean Square error.
    error = dec.Decimal(0)    
    dec.getcontext().prec = prec
    for i in np.arange(len(P)):        
        error += ((dec.Decimal(p[i]) - dec.Decimal(y[i]))**2)
    error = error / (dec.Decimal(2) * dec.Decimal(len(P)))  
    
    # Calculating Sum of weights and biases.
    weights = dec.Decimal(0)
    for i in np.arange(len(W)):
        weights += dec.Decimal(np.sum(W[i]**2))
        weights += dec.Decimal(np.sum(B[i]**2))
    weights = (dec.Decimal(phi) * weights) / dec.Decimal(2)
    
    return error + weights
        
   
def train(X, Y, W, B, phi, verbose=True):
    """
    Trains the network on the data given by X and Y.
    
    :param X: np.array
    :param Y: np.array
    :param W: List
    :param B: List
    """
    
    rmse_per_epoch = []    
    
    Y = Y.reshape(len(Y), 1)
    m, n = X.shape
    eta = 0.1
    eta_max = 1e15
    delta = 100
    epoch = 1
    
    while delta > 1e-5:
        
        # Calculating first order derivatives.        
        P, N = propForward(X, W, B) # Forward Propogation.
        WU, D = propBack(P, Y, N, W, B, m, phi)   # Back Propogation.
        
        # Caclulating the error.
        error = error_dec(P, Y, W, B, phi, 50)
        
        # Calculating the parameter updates.
        j = jacobian(m, WU, D)  # Jacobian.
        JT = j.T    # Transpose of Jacobian.
        H = np.dot(JT, j)   # Approximate Hessian.
        E = np.abs(Y-P)    # Error Vector.
        G = np.dot(JT, E)    # Gradient Vector.
        
        # Calculating trial update values.
        U = np.dot(np.linalg.inv(H + eta*np.eye(len(H))),G)
        nW, nB = up(U, W, B) 
        P = propForward(X, np.subtract(W, nW), np.subtract(B, nB))[0]
        e = error_dec(P, Y, W, B, phi, 50)
        
        # Evaluating trial update values.
        if e < error:
            eta *= 0.1
            
        else:
            while e >= error and eta < eta_max:
                eta *= 10

                # Calculating trial update values.
                U = np.dot(np.linalg.inv(H + eta*np.eye(len(H))),G)
                nW, nB = up(U, W, B) 
                P = propForward(X, np.subtract(W, nW), np.subtract(B, nB))[0]
                e = error_dec(P, Y, W, B, phi, 50)
                        
        # Updating weghts.
        W = np.subtract(W, nW)
        B = np.subtract(B, nB)
        delta = error - e
                    
        # Reporting on step.
        if verbose==True:
            P = propForward(X, W, B)[0]
            
            # Root mean square error.
            rmse = np.sqrt(np.sum((P-Y)**2) / m)
            
            # R-squared.
            r2 = r2_score(Y, P)
            
            rmse_per_epoch.append((rmse, epoch))
            
            print 'RMSE: %r,    R2: %r,    Epoch: %i,    Delta: %.6f' % (rmse , r2, epoch, float(delta))
        epoch +=1
        
        # Undoing the weight update if the error increased.
        if delta < 0:
            W = np.add(W, nW)
            B = np.add(B, nB)
    
    # Reporting on the final result.
    P = propForward(X, W, B)[0]
            
    # Root mean square error.
    rmse = np.sqrt(np.sum((P-Y)**2) / m)
    
    # R-squared.
    r2 = r2_score(Y, P)        
    print "RMSE: %r,    R2: %r,    Epochs: %i,    phi: %.2f" % (rmse, r2, epoch, phi) 
    
    return W, B, rmse_per_epoch

# Extracting traning data.
df = pd.read_csv('B197/B197.dat', index_col=0).fillna(0)

# Extracting training and testing data.
X_train = df['2012-02-01':'2015-01-31'].ix[:, :'spring'].values
Y_train = df['2012-02-01':'2015-01-31'].ix[:, 'daily_demand'].values
X_test = df['2015-02-01':'2016-01-31'].ix[:, :'spring'].values
Y_test = df['2015-02-01':'2016-01-31'].ix[:, 'daily_demand'].values

# Normalizing the Data.

# Normalizing the training and testing feature sets.
xmean = np.mean(X_train, axis=0)
xstd = np.std(X_train, axis=0)
X_train_n = np.nan_to_num((X_train-xmean) / xstd)
X_test_n = np.nan_to_num((X_test-xmean) / xstd)

# Normalizing the training labels.
ymean = np.mean(Y_train, axis=0)
ystd = np.std(Y_train, axis=0)
Y_train_n = (Y_train-ymean) / ystd

# Transforming the training and testing feature sets using Principle Component Analysis.
U = pca(X_train_n, 0.01)
X_train_n = np.dot(X_train_n, U)
X_test_n = np.dot(X_test_n, U)

# Defining the network structure.
net = [20, 1]

# Finding optimal regularization term by cross-validation.
# lamda, regs = cross_validate_regular(0.21, 0.01, net, X_train_n, Y_train_n)
lamda = 0.0


# Initializing the network.
W, B = build(X_train_n, net)

# Training the network
W, B, rms_per_epoch = train(X_train_n, Y_train_n, W, B, lamda)

# Collecting final results of training.

# Final Prediction.
P = np.round(((propForward(X_test_n, W, B)[0]*ystd) + ymean).flatten(), -1).astype('float64')
P[P<0]=0
# Plotting Final Predicted values vs labels.
sns.set_style("darkgrid")
plt.figure(figsize=(12, 8))
plt.title('Predicted Values vs Observerd values', fontsize=20)
plt.xlabel('Day of the year', fontsize=18)
plt.ylabel('Total Withdrawal amounts', fontsize=18)

observed = plt.plot(Y_test, label='Observed')
predicted = plt.plot(P, marker='o', color='r', label='Predicted')
plt.legend(prop={'size':16})

plt.show()


print 0.5 * mse(P, Y_test)
residuals = np.abs(P - Y_test)
mape = np.sum(residuals / Y_test) / len(P)
stndev = np.std(residuals)
print "MAPE %r,    STD %r,    R %r" % (mape, stndev, r2_score(Y_test, P))