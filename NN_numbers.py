import numpy
import scipy.special
import matplotlib.pyplot

class NeuralNetwork:
	def __init__(self, inputNodes, hiddenNodes, outputNodes, learningRate):
		# Neural Networks initialisation
		self.inpNodes = inputNodes
		self.hidNodes = hiddenNodes
		self.outNpdes = outputNodes

		self.w_ih = numpy.random.normal(0.0, pow(self.hidNodes, -0.5), (self.hidNodes, self.inpNodes))
		self.w_ho = numpy.random.normal(0.0, pow(self.outNpdes, -0.5), (self.outNpdes, self.hidNodes))

		self.learRate = learningRate

		self.activFunc = lambda x: scipy.special.expit(x)
		pass

	def train(self, inputList, targetList):
		# Here is training of neural network
		inputs = numpy.array(inputList, ndmin=2).T
		targets = numpy.array(targetList, ndmin=2).T

		hiddenInputs = numpy.dot(self.w_ih, inputs)
		hiddenOutputs = self.activFunc(hiddenInputs)

		finalInputs = numpy.dot(self.w_ho, hiddenOutputs)
		finalOutputs = self.activFunc(finalInputs)

		# Search for errors on each layer
		outputErrors = targets - finalOutputs
		hiddenErrors = numpy.dot(self.w_ho.T, outputErrors)

		# Weight correction
		self.w_ho += self.learRate * numpy.dot((outputErrors * finalOutputs * (1.0 - finalOutputs)), numpy.transpose(hiddenOutputs))
		self.w_ih += self.learRate * numpy.dot((hiddenErrors * hiddenOutputs * (1.0 - hiddenOutputs)), numpy.transpose(inputs))
		pass

	def query(self, inputList):
		# Here is quering of neural network
		inputs = numpy.array(inputList, ndmin=2).T

		hiddenInputs = numpy.dot(self.w_ih, inputs)
		hiddenOutputs = self.activFunc(hiddenInputs)

		finalInputs = numpy.dot(self.w_ho, hiddenOutputs)
		finalOutputs = self.activFunc(finalInputs)
		return finalOutputs

def main(learningRate, epochs):
	inputNodes = 784
	hiddenNodes = 200
	outputNodes = 10
	learningRate = 0.3
	epochs = 1

	n = NeuralNetwork(inputNodes, hiddenNodes, outputNodes, learningRate)

	# Here is a training on 60 000 examples
	training_data_file = open("mnist_train.csv", 'r')
	training_data_list = training_data_file.readlines()
	training_data_file.close()
	for e in range(epochs):
		for record in training_data_list:
			all_values = record.split(',')
			inputs = (numpy.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
			targets = numpy.zeros(outputNodes) + 0.01
			targets[int(all_values[0])] = 0.99
			n.train(inputs, targets)
			pass
		pass

	# Here is a testing on 10 000 examples
	test_data_file = open("mnist_test.csv", 'r')
	test_data_list = test_data_file.readlines()
	test_data_file.close()
	score = []
	for record in test_data_list:
		all_values = record.split(',')
		correct_label = int(all_values[0])
		inputs = (numpy.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
		outputs = n.query(inputs)
		label = numpy.argmax(outputs)

		if label == correct_label:
			score.append(1)
		else:
			score.append(0)
			pass
		pass

	#print(score)
	score_array = numpy.asarray(score)
	evaluation = score_array.sum()/score_array.size
	print("Evaluation:", evaluation)
	return evaluation

if __name__ == "__main__":
	# Here we are trying to set different the settings of learning rates and epochs
	learnRates = numpy.arange(0.0, 0.5, 0.1)
	epochs = [1, 3, 5, 8]
	results = []
	i = 0
	for lr in learnRates:
		results.append([])
		print("Learning Rate:", lr)
		for ep in epochs:
			print("Epochs:", ep)
			results[i].append(main(lr, ep))
			pass
		print(results[i], "\n")
		input()
		i += 1
		pass
	print(results)