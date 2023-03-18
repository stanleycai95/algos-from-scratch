import numpy as np

class Conv2D:
    
    def __init__(self, num_filters, input_shape, filter_size=5, stride=1, learning_rate=3e1):
        assert num_filters >= 1
        assert (filter_size % 2) == 1
        assert len(input_shape) == 4
        assert stride >= 1
        
        self.num_filters = num_filters
        self.filter_size = filter_size
        self.input_shape = input_shape
        self.stride = stride
        self.learning_rate = learning_rate

        self.weights = np.array([np.random.normal(0, np.sqrt(1 / (num_filters * filter_size**2)), 
                                                  size=(filter_size, filter_size)) for i in range(num_filters)])
        self.bias = 0
            
    def get_output_shape(self):
        return int(self.input_shape[1]), int(self.input_shape[2]), self.num_filters
    
    def forwardprop(self, X):
        output_shape = self.get_output_shape()
        conv2d_outputs = np.zeros(shape=(X.shape[0], output_shape[0], output_shape[1], output_shape[2]))

        self.pad_dim = self.filter_size // 2
        X = np.pad(X, ((0,0), (self.pad_dim, self.pad_dim), (self.pad_dim, self.pad_dim), (0,0)), mode='constant', constant_values=0)

        for n in range(X.shape[0]):
            for i in range(conv2d_outputs.shape[1]):
                for j in range(conv2d_outputs.shape[2]):
                    left_bound, top_bound = i * self.stride, j * self.stride
                    curr_segment = X[n, left_bound:left_bound+self.filter_size, top_bound:top_bound+self.filter_size, :]
                    for k in range(self.num_filters):
                        conv2d_outputs[n, i, j, k] = np.sum(np.sum(curr_segment, axis=-1) * self.weights[k,:,:]) + self.bias
        
        return conv2d_outputs
    
    def backprop(self, input_grad, activations, next_layer=None):
        padded_activations = np.pad(activations, ((0,0), (self.pad_dim, self.pad_dim), (self.pad_dim, self.pad_dim), (0,0)), mode='constant', constant_values=0)

        self.learning_rate *= 0.99        
        self.bias -= self.learning_rate * np.mean(input_grad)

        for k in range(self.weights.shape[0]):
            for i in range(self.weights.shape[1]):
                for j in range(self.weights.shape[2]):
                    self.weights[k,i,j] -= self.learning_rate * np.mean(input_grad * padded_activations[:, i:i+input_grad.shape[1],
                                                                                                            j:j+input_grad.shape[2],k])
        output_grad = np.zeros(shape=input_grad.shape)
        for i in range(output_grad.shape[1]):
            for j in range(output_grad.shape[2]):
                top_bound, left_bound = max(self.pad_dim-i, 0), max(self.pad_dim-j, 0)
                bottom_bound, right_bound = min(input_grad.shape[1]-i, self.filter_size), min(input_grad.shape[2]-j, self.filter_size)
                weight_contribution = np.mean(self.weights[:, top_bound:bottom_bound, left_bound:right_bound])
                output_grad[:,i,j] = input_grad[:,i,j] * weight_contribution
        
        print(np.percentile(output_grad, [75 ,25]))
        return output_grad
    
class Pool2D:
    
    def __init__(self, pool_dim=2, pool_type='avg'):
        self.pool_dim = pool_dim
        self.pool_type = pool_type
    
    def forwardprop(self, X):
        pool_outputs = np.zeros(shape=(X.shape[0], X.shape[1]//self.pool_dim, X.shape[2]//self.pool_dim, X.shape[3]))
        
        for n in range(X.shape[0]):
            for i in range(pool_outputs.shape[1]):
                for j in range(pool_outputs.shape[2]):
                    left_bound, top_bound = i * self.pool_dim, j * self.pool_dim
                    curr_segment = X[n, left_bound:left_bound+self.pool_dim, top_bound:top_bound+self.pool_dim, :]
                    pool_outputs[n, i, j, :] = np.mean(curr_segment, axis=(0,1))
        
        return pool_outputs
    
    def backprop(self, grad):
        output_grad = grad
        return output_grad
        
class Flatten:
    
    def forwardprop(self, X):
        self.original_shape = X.shape
        flattened_X = X.reshape(X.shape[0], int(X.shape[1] * X.shape[2] * X.shape[3]))
        return flattened_X
    
    def backprop(self, grad):
        reshaped_grad = grad.reshape(self.original_shape).mean(axis=-1)
        return reshaped_grad

class Dense:
    
    def __init__(self, input_shape, output_shape=1, learning_rate=3e-4):
        
        self.weights = np.random.normal(0, np.sqrt(1 / (input_shape * output_shape)), size=(input_shape, output_shape))
        self.biases = np.zeros(shape=output_shape)
        self.learning_rate = learning_rate
    
    def forwardprop(self, X):
        return (X @ self.weights) + self.biases
    
    def backprop(self, input_grad, activations):
        m = activations.shape[0]
        self.learning_rate *= 0.99

        output_grad = input_grad @ self.weights.T
        
        self.weights -= 1/m * self.learning_rate * activations.T @ input_grad
        self.biases -= 1/m * self.learning_rate * np.mean(input_grad, axis=0)
        
        return output_grad

class Sigmoid:
    
    def forwardprop(self, X):
        sigmoid_X = np.exp(X) / (1 + np.exp(X))
        return sigmoid_X
    
    def backprop(self, input_grad):
        sigmoid_grad = self.forwardprop(input_grad)
        output_grad = input_grad * (sigmoid_grad * (1 - sigmoid_grad))
        return output_grad

class LeakyRelu:
    
    def forwardprop(self, X):
        leakyrelu_X = np.where(X > 0, X, 0.01*X)
        return leakyrelu_X
    
    def backprop(self, input_grad):
        output_grad = input_grad * np.where(input_grad > 0, 1, 0.01)
        return output_grad

class ConvolutionalNeuralNetwork:
    
    def __init__(self, layers_num_filters, batch_size=256, num_epochs=5):
        assert len(layers_num_filters) >= 1, "invalid layer dimensions for neural net"
        
        self.batch_size = batch_size
        self.layers_num_filters = layers_num_filters
        self.layers = []
        self.num_epochs = num_epochs
        
        self.fitted = False
    
    def binary_cross_entropy(self, y_pred, y):
        y = self.reshape_y(y)
        y_pred = np.clip(y_pred, 1e-15, 1-1e-15)
        return -np.mean(y * np.log(y_pred) + (1-y) * np.log(1-y_pred))
    
    def scale_X(self, X):
        if not self.fitted:
            self.X_means = np.mean(X)
            self.X_std = np.std(X)
        
            self.fitted = True
            
        return (X - self.X_means) / self.X_std
    
    def reshape_y(self, y):
        if len(y.shape) == 1:
            return y[:,None]
        elif len(y.shape) == 2:
            return y
        else:
            print("Invalid y shape")
    
    def initialize_neuralnet(self, X):
        for i in range(len(self.layers_num_filters)):
            if i == 0:
                input_shape, num_filters = X.shape, self.layers_num_filters[i]
            else:
                input_dim1, input_dim2, input_channels = self.layers[-3].get_output_shape()
                input_dim1 = input_dim1 / self.layers[-2].pool_dim
                input_dim2 = input_dim2 / self.layers[-2].pool_dim
                input_shape, num_filters = (X.shape[0], input_dim1, input_dim2, input_channels), self.layers_num_filters[i]
                
            self.layers.append(Conv2D(num_filters=num_filters, input_shape=input_shape))
            self.layers.append(Pool2D())
            self.layers.append(LeakyRelu())
            
        input_dim1, input_dim2, input_channels = self.layers[-3].get_output_shape()
        input_dim1 = input_dim1 // self.layers[-2].pool_dim
        input_dim2 = input_dim2 // self.layers[-2].pool_dim
        self.layers.append(Flatten())
        self.layers.append(Dense(int(input_dim1 * input_dim2 * input_channels)))
        self.layers.append(Sigmoid())
    
    def fit(self, X, y):
        X = self.scale_X(X)
        y = self.reshape_y(y)
        self.initialize_neuralnet(X)
        
        for i in range(self.num_epochs):
            p = np.random.permutation(len(X))
            X, y = X[p], y[p]
            num_batches = X.shape[0] // self.batch_size
            X_batches, y_batches = np.array_split(X, num_batches), np.array_split(y, num_batches)
            for batch_num in range(num_batches):
                y_pred_batch = self.forwardprop(X_batches[batch_num])
                self.backprop(X_batches[batch_num], y_batches[batch_num])
            y_pred_full = self.forwardprop(X)
            print(f"Accuracy epoch {i+1}: " + str(np.mean((y_pred_full > 0.5).astype(int) == y)))
        
        self.fitted = True

    def predict_proba(self, X):
        X = self.scale_X(X)
        probs = self.forwardprop(X)
        return probs
    
    def predict(self, X):
        probs = self.predict_proba(X)
        
        return (probs > 0.5).astype(int)
        
    def forwardprop(self, X):
        self.activations = [X]
        
        for i in range(len(self.layers)):
            self.activations.append(self.layers[i].forwardprop(self.activations[-1]))
            
        return self.activations[-1]
    
    def backprop(self, X, y):
        for i in range(len(self.layers)):
            layer_index = len(self.layers) - i-1
            next_layer = None
            if i == 0:
                grad = (self.activations[-1] - y)
            else:
                curr_layer = self.layers[layer_index]
                if hasattr(curr_layer, 'weights'):
                    if hasattr(curr_layer, 'num_filters'):
                        activation_layer_index = layer_index+2
                    else:
                        activation_layer_index = layer_index+2
                    grad = self.layers[layer_index].backprop(grad, self.activations[activation_layer_index])
                else:
                    grad = self.layers[layer_index].backprop(grad)

def test_class(train_size, test_size):

    from keras.datasets import mnist
    
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    (X_train, y_train), (X_test, y_test) = (X_train[y_train <= 1], y_train[y_train <= 1]), (X_test[y_test <= 1], y_test[y_test <= 1])
    
    X_train = np.pad(X_train[:,:,:,None], pad_width = ((0,0), (2,2), (2,2), (0,0)), mode='constant', constant_values=0)[:train_size]
    X_test = np.pad(X_test[:,:,:,None], pad_width = ((0,0), (2,2), (2,2), (0,0)), mode='constant', constant_values=0)[:test_size]
    y_train = y_train[:train_size]
    y_test = y_test[:test_size]
    
    print("Train shape, test shape")
    print(X_train.shape, X_test.shape)
    
    cnn = ConvolutionalNeuralNetwork([3, 4])
    cnn.fit(X_train, y_train)
    y_pred = cnn.predict(X_test)
    y_pred_proba = cnn.predict_proba(X_test)
    y_pred_train = cnn.predict(X_train)
    y_pred_proba_train = cnn.predict_proba(X_train)
    
    print("Test Accuracy")
    print(np.mean(y_pred.flatten() == y_test))
    print("Train Accuracy")
    print(np.mean(y_pred_train.flatten() == y_train.flatten()))
    print("Baseline")
    print(np.mean(y_test.flatten()==1))
    
    print("Sample pred")
    print(y_pred.flatten())
    print("Sample probabilities")
    print(y_pred_proba.flatten())
    print("Sample y")
    print(y_test.flatten())
    
test_class(train_size=1024, test_size=64)
