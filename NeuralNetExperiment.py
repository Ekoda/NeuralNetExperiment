import numpy as np


def sigmoidActivator(x):
    return 1 / (1 + np.exp(-x))


class Neuron:
    def __init__(self, weights, bias):
        self.weights = weights
        self.bias = bias

    def feedForward(self, inputs):
        z = np.dot(self.weights, inputs) + self.bias
        return sigmoidActivator(z)


class NeuralNetwork:
    def __init__(self):
        HN1weights = np.array([-1.2714172259077934, 0.11481589722563311])
        HN2weights = np.array([2.996913372909818, -0.08033467436158576])
        ONweights = np.array([2.96889604350092, -3.5595815944375997])
        HN1bias = 0.3957225052931545
        HN2bias = -1.2097638079416635
        ONbias = 0

        self.hiddenNeuron1 = Neuron(HN1weights, HN1bias)
        self.hiddenNeuron2 = Neuron(HN2weights, HN2bias)
        self.outputNeuron = Neuron(ONweights, ONbias)

    def NeuralNetFeedForward(self, x):
        HiddenNeuron1output = self.hiddenNeuron1.feedForward(x)
        HiddenNeuron2output = self.hiddenNeuron2.feedForward(x)
        HiddenLayerInput = np.array(
            [HiddenNeuron1output, HiddenNeuron2output])
        return self.outputNeuron.feedForward(HiddenLayerInput)


# Input values of Neuron
trueCheck = np.array([-7, -3])
falseCheck = np.array([20, 2])

print(NeuralNetwork().NeuralNetFeedForward(trueCheck))  # Returns 1, true
print(NeuralNetwork().NeuralNetFeedForward(falseCheck))  # returns 0, false
