import random
import numpy as np

class Network(object):

	# build the neural network structure
	def __init__(self, sizes):
		# neural network layer number
		self.num_layers = len(sizes)
		# neural network each layer size
		self.sizes = sizes
		# bias term 2 iterms
		self.biases = [np.random.randn(y, 1) 
						for y in sizes[1: ]]
		# weights 2 iterms
		self.weights = [np.random.randn(y, x) 
						for x, y in zip(sizes[:-1], sizes[1:])]

	# feedforward propagation
	# calculate cost throughout each layer from input layer to hidden layer
	# and the final output layer
	# compute the cost of the feedforward propagation result
	def feedforward(self, a):
		for b, w in zip(self.biases, self.weights):
			a = sigmoid(np.dot(w, a) + b)
		return a

	# stochasitic gradient descent
	# including mini batch gradient descent
	def SGD(self, training_data, epochs, mini_batch_size, eta, test_data=None):
		training_data = list(training_data)

		n = len(training_data)

		if test_data:
			test_data = list(test_data)
			n_test = len(test_data)
		
		# loop training
		for j in range(epochs):
			# randomly shuffle the training data set
			random.shuffle(training_data)
			
			# get mini batch data set
			# from the training data set
			# given the mini batch size
			# Note the pythonic pragramming code style
			mini_batches = [
				training_data[k:k+mini_batch_size]
				#            start, end, step size
				for k in range(0, n, mini_batch_size)]
			
			for mini_batch in mini_batches:
				self.update_mini_batch(mini_batch, eta)

			if test_data:
				print('Epoch {}: {} / {}'.format(j, self.evaluate(test_data), n_test))
			else:
				print('Epoch {} complete'.format(j))

	# update weights and biases using the mini batch data set
	def update_mini_batch(self, mini_batch, eta):
		# get the same shape of weights and biases
		nabla_b = [np.zeros(b.shape) for b in self.biases]
		nabla_w = [np.zeros(w.shape) for w in self.weights]

		# udpate weights and biases throught mini batch data set
		# note that the pythonic code below
		for x, y in mini_batch:
			delta_nabla_b, delta_nabla_w = self.backprop(x, y)
			# the pythonic programming code style
			nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
			nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]

		# !Important 
		# Your should update weights and biases simultaneously
		# Note: w and b are numpy.ndarray
		# so they can apply matrix operation like + , - , * , etc
		self.weights = [w - (eta / len(mini_batch)) * nw 
						for w, nw in zip(self.weights, nabla_w)]

		self.biases = [b - (eta / len(mini_batch)) * nb
						for b, nb in zip(self.biases,nabla_b)]

	# backpropagation implementation
	# backpropagation algorithm should be the most complex algortihm 
	# in neural network
	# the goal of it is to compute derivatives
	def backprop(self, x, y):
		# contructor the same shape of weights and biases
		nabla_b = [np.zeros(b.shape) for b in self.biases]
		nabla_w = [np.zeros(w.shape) for w in self.weights]
		
		# back propagation should be computed after 
		# feedforward propagation
		activation = x
		# collect all the activation (after sigmoid)
		activations = [x]
		# collect all the activation (before sigmoid)
		zs = []

		# running feedforward propagation
		# on each layer: from input, then hidden, final output
		for b, w in zip(self.biases, self.weights):
			z = np.dot(w, activation) + b
			zs.append(z)
			activation = sigmoid(z)
			activations.append(activation)

		# the |y - y'| from the output layer
		delta = self.cost_derivative(activations[-1], y) * sigmoid_prime(zs[-1])

		nabla_b[-1] = delta
		nabla_w[-1] = np.dot(delta, activations[-2].transpose())

		# running back propagation from the back second layer
		# from back (output layer) to front (input layer) via hidden layer
		# core code
		for l in range(2, self.num_layers):
			z = zs[-l]
			sp = sigmoid_prime(z)
			delta = np.dot(self.weights[-l + 1].transpose(), delta) * sp
			nabla_b[-l] = delta
			nabla_w[-l] = np.dot(delta, activations[-l - 1].transpose())

		return (nabla_b, nabla_w)

	# evaluate the training result
	# calculate the computed accuracy
	# the pythonic programming
	def evaluate(self, test_data):
		# the final feedforward contains a (10 * 1) column vector
		test_results = [(np.argmax(self.feedforward(x)), y)
						for (x, y) in test_data]
	
		# calculate the accuracy on the test data set
		return sum(int(x == y) for (x, y) in test_results)

	# compute cost derivatives
	# the output layer's derivative
	def cost_derivative(self, output_activations, y):
		return (output_activations - y)

# the helper function
# sigmoid function
def sigmoid(z):
	return 1.0 / (1.0 + np.exp(-z))

# sigmoid function's derivative
# the advantage of sigmoid function on compute derivatives
def sigmoid_prime(z):
	return sigmoid(z) * (1 - sigmoid(z))


if __name__ == '__main__':
	import mnist_loader
	
	# Note: there is not used the validation data set
	# because the learning rate and other hyper parameters 
	# are already set
	# so, when we need to find the opitmal value of learning rate
	# and other hyper parameters 
	# we need a test set to evaluate our model(including the parameters)
	# so we need the validation data set to do this!!!
	training_data, validation_data, test_data = mnist_loader.load_data_wrapper()

	# build the neural network structure
	# input layer 784
	# hidden layer 100
	# output layer 10
	# weights should be [(100 * 784), (10 * 100)]
	# biases should be [(100 * 1), (10 * 1)] just to contruct a column vector
	net = Network([784, 100, 10])

	# using Stochastic Gradient Descent to train the data
	#   training data set, epoch, mini batch size, learning rate, test data set
	net.SGD(training_data, 30, 10, 3.0, test_data=test_data)