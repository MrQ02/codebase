import numpy as np
import h5py
import matplotlib.pyplot as plt

def sigmoid(Z):
    A = 1/(1+np.exp(-Z))
    assert(A.shape == Z.shape)
    cache = Z
    return A, cache

def tanh(Z):
    A = (np.exp(Z)-np.exp(-Z))/(np.exp(Z)+np.exp(-Z))
    assert(A.shape == Z.shape)
    cache = Z
    return A,cache

def relu(Z):
    A = np.maximum(0,Z)
    assert(A.shape == Z.shape)
    cache = Z
    return A,cache

def leaky_relu(Z):
    epsilon = 0.01 # EDIT HERE FOR DIFFERENT E
    A = np.maximum(epsilon*Z,Z)
    assert(A.shape == Z.shape)
    cache = Z
    return A,cache

def sigmoid_backward(dA, Z):
    g = 1/(1+np.exp(-Z))
    dZ = dA*g*(1-g)
    assert(dZ.shape == Z.shape)
    return dZ

def tanh_backward(dA, Z):
    g = (np.exp(Z)-np.exp(-Z))/(np.exp(Z)+np.exp(-Z))
    dZ = dA*(1-np.power(g,2))
    assert(dZ.shape == Z.shape)
    return dZ

def relu_backward(dA, Z):
    dZ = np.array(dA,copy=True)
    dZ[Z<=0] = 0
    assert(dZ.shape == Z.shape)
    return dZ

def leaky_relu_backward(Z):
    epsilon = 0.01 # EDIT HERE FOR DIFFERENT E
    dZ = np.array(dA,copy=True)
    dZ[Z<=0] = dZ[Z<=0] * epsilon
    assert(dZ.shape == Z.shape)
    return dZ

def init_params(layer_dims):
    np.random.seed(22)
    params = {}
    for l in range(1,len(layer_dims)):
        params['W'+str(l)] = np.random.randn(layer_dims[l],layer_dims[l-1])*0.01
        params['b'+str(l)] = np.zeros((layer_dims[l],1))
        assert(params['W'+str(l)].shape == (layer_dims[l],layer_dims[l-1]))
        assert(params['b'+str(l)].shape == (layer_dims[l],1))
    return params

def linear_forward(A, W, b):
    Z = np.dot(W,A)+b
    assert(Z.shape == (W.shape[0],A.shape[1]))
    cache = (A,W,b)
    return Z,cache

def activation_forward(A_prev, W, b, activation):
    if activation == "sigmoid":
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = sigmoid(Z)
    elif activation == "relu":
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = relu(Z)
    elif activation == "tanh":
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = tanh(Z)
    elif activation == "leaky_relu":
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = leaky_relu(Z)
    
    assert(A.shape == (W.shape[0],A_prev.shape[1]))
    cache = (linear_cache,activation_cache)
    return A,cache

def model_forward(X, params):
    # This model follows "ReLU for all -> Sigmoid".
    caches = []
    A = X 
    L = len(params) // 2
    # ReLU
    for l in range(1,L):
        A_prev = A
        A, cache = activation_forward(A_prev, params['W'+str(l)], params['b'+str(l)], "relu")
        caches.append(cache)
    # Sigmoid
    AL, cache = activation_forward(A, params['W'+str(L)], params['b'+str(L)], "sigmoid")
    caches.append(cache)
    
    assert(AL.shape == (1,X.shape[1]))
    return AL,caches

def get_cost(AL,Y):
    cost = np.squeeze(-1/Y.shape[1] * np.sum(np.multiply(Y,np.log(AL)) + np.multiply((1-Y),np.log(1-AL))))
    assert(cost.shape == ())
    return cost

def linear_backward(dZ, cache):
    A_prev,W,b = cache
    dW = 1/A_prev.shape[1]*np.dot(dZ,A_prev.T)
    db = 1/A_prev.shape[1]*np.sum(dZ, axis=1, keepdims=True)
    dA_prev = np.dot(W.T,dZ)
    assert (dA_prev.shape == A_prev.shape)
    assert (dW.shape == W.shape)
    assert (db.shape == b.shape)
    return dA_prev, dW, db

def activation_backward(dA, cache, activation):
    linear_cache, activation_cache = cache
    if activation == "sigmoid":
        dZ = sigmoid_backward(dA, activation_cache)
        dA_prev, dW, db = linear_backward(dZ, linear_cache)
    elif activation == "relu":
        dZ = relu_backward(dA, activation_cache)
        dA_prev, dW, db = linear_backward(dZ, linear_cache)
    elif activation == "tanh":
        dZ = tanh_backward(dA, activation_cache)
        dA_prev, dW, db = linear_backward(dZ, linear_cache)
    elif activation == "leaky_relu":
        dZ = leaky_relu_backward(dA, activation_cache)
        dA_prev, dW, db = linear_backward(dZ, linear_cache)
    return dA_prev, dW, db
    
def model_backward(AL, Y, caches):
    grads = {}
    L = len(caches)
    m = AL.shape[1]
    Y = Y.reshape(AL.shape)
    dAL = -(np.divide(Y, AL)-np.divide(1-Y,1-AL))
    current_cache = caches[L-1]
    # Sigmoid
    grads["dA" + str(L-1)], grads["dW" + str(L)], grads["db" + str(L)] = activation_backward(dAL, current_cache, "sigmoid")
    # ReLU
    for l in reversed(range(L-1)):
        current_cache = caches[l]
        dA_prev_temp, dW_temp, db_temp = activation_backward(grads["dA" + str(l+1)], current_cache, "relu")
        grads["dA" + str(l)] = dA_prev_temp
        grads["dW" + str(l + 1)] = dW_temp
        grads["db" + str(l + 1)] = db_temp
    return grads
        
def gradient_descent(params, grads, alpha):
    for l in range(1,len(params) // 2):
        params["W"+str(l+1)] -= alpha * grads["dW" + str(l+1)]
        params["b"+str(l+1)] -= alpha * grads["db" + str(l+1)]
    return params

def model(X, Y, layers_dims, alpha=0.01, n_iterations = 3000, print_cost=False):
    np.random.seed(123)
    costs = []
    params = init_params(layers_dims)
    for i in range(0,n_iterations):
        AL,caches = model_forward(X,params)
        cost = get_cost(AL,Y)
        grads = model_backward(AL,Y,caches)
        params = gradient_descent(params,grads,alpha)
        
        if print_cost and i%100 == 0:
            print("Cost after iteration %i: %f" %(i,cost))
        costs.append(cost)
    plt.plot(np.squeeze(costs))
    plt.title("Costs during training")
    plt.xlabel("iterations (per 10s)")
    plt.ylabel('cost')
    plt.show()
    return params

def predict(X_test, Y_test, parameters):
    p = np.zeros((1,X_test.shape[1]))
    probas, caches = model_forward(X_test, parameters)
    for i in range(0, probas.shape[1]):
        p[0,i] = 1 if probas[0,i]>0.5 else 0
    print("Accuracy: "+str(np.sum((p == Y_test)/X_test.shape[1])))
    return p