import numpy as np

class FeedforwardNeuralNetwork:
    
    def __init__(self, layer_dims, learning_rate=1e-5, batch_size=64, activation_type='leaky_relu'):
        assert (len(layer_dims) >= 1) and (layer_dims[-1] == 1), "invalid layer dimensions for neural net"
        assert activation_type in ('leaky_relu', 'sigmoid'), "other activations not implemented yet"
        
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.layer_dims = layer_dims
        self.activation_type = activation_type
        self.layer_weights = []
        self.layer_biases = []
        
        self.fitted = False
    
    def activation_function(self, x):
        if self.activation_type == 'leaky_relu':
            return np.where(x > 0, x, 0.01*x)
        elif self.activation_type == 'sigmoid':
            return np.exp(x) / (1 + np.exp(x))
        else:
            print("Invalid activation type")
    
    def activation_derivative(self, x):
        if self.activation_type == 'leaky_relu':
            return np.where(x > 0, 1, 0.01)
        elif self.activation_type == 'sigmoid':
            return self.sigmoid(x) * (1 - self.sigmoid(x))    
        else:
            print("Invalid activation type")
    
    def mean_squared_error(self, y_pred, y):
        y = self.reshape_y(y)
        return np.mean(np.square(y_pred - y))
        
    def scale_X(self, X):
        if not self.fitted:
            self.X_means = np.mean(X, axis=0)
            self.X_std = np.std(X, axis=0)
        
            self.fitted = True
            
        return (X - self.X_means) / self.X_std
    
    def reshape_y(self, y):
        if len(y.shape) == 1:
            return y[:,None]
        elif len(y.shape) == 2:
            return y
        else:
            print("Invalid y shape")
    
    def initialize_neuralnet(self, X, method='xavier'):
        for i in range(len(self.layer_dims)):
            if i == 0:
                weight_dim1, weight_dim2 = X.shape[1], self.layer_dims[i]
            else:
                weight_dim1, weight_dim2 = self.layer_dims[i-1], self.layer_dims[i]
            self.layer_weights.append(np.random.normal(0, 1 / np.sqrt(weight_dim1), size=(weight_dim1, weight_dim2)))
            self.layer_biases.append(np.zeros(weight_dim2))
    
    def fit(self, X, y, epochs=100):
        X = self.scale_X(X)
        y = self.reshape_y(y)
        self.initialize_neuralnet(X)
        
        for i in range(epochs):
            self.learning_rate *= 0.99
            p = np.random.permutation(len(X))
            X, y = X[p], y[p]
            num_batches = X.shape[0] // self.batch_size
            X_batches, y_batches = np.split(X, num_batches), np.split(y, num_batches)
            for batch_num in range(num_batches):
                self.forwardprop(X_batches[batch_num])
                self.backprop(X_batches[batch_num], y_batches[batch_num])
        
            print(self.mean_squared_error(self.forwardprop(X), y))
        
        self.fitted = True

    def predict(self, X):
        X = self.scale_X(X)
        return self.forwardprop(X)
        
    def forwardprop(self, X):
        self.activations, self.z = [X], [X]
        for i in range(len(self.layer_weights)):
            self.z.append(self.activations[-1] @ self.layer_weights[i] + self.layer_biases[i])
            self.activations.append(self.activation_function(self.z[-1]))
        
        return self.activations[-1]
    
    def backprop(self, X, y):
        for i in range(len(self.layer_weights)):
            layer_index = len(self.layer_weights) - i-1
            if i == 0:
                delta = np.multiply(self.activations[-1] - y, self.activation_derivative(self.z[-1]))
            else:
                delta = np.multiply(delta @ self.layer_weights[layer_index+1].T, self.activation_derivative(self.z[layer_index+1]))
            self.layer_weights[layer_index] -= self.learning_rate * self.activations[layer_index].T @ delta
            self.layer_biases[layer_index] -= self.learning_rate * np.mean(delta, axis=0)
        

def test_class():
    from sklearn.datasets import fetch_california_housing

    X, y = fetch_california_housing(return_X_y=True)
    shuffle = np.random.permutation(len(X))
    X, y = X[shuffle], y[shuffle]

    train_test_cutoff = X.shape[0] * 4//5
    X_train, y_train = X[:train_test_cutoff,:], y[:train_test_cutoff]
    X_test, y_test = X[train_test_cutoff:,:], y[train_test_cutoff:]

    nn = FeedforwardNeuralNetwork([10,1])
    nn.fit(X_train, y_train)
    y_pred = nn.predict(X_test)
    y_pred_train = nn.predict(X_train)

    print("Neural net test loss")
    print(nn.mean_squared_error(y_pred, y_test))
    
    print("Neural net train loss")
    print(nn.mean_squared_error(y_pred_train, y_train))
    
    print("Baseline loss")
    print(nn.mean_squared_error(y.mean(), y))
