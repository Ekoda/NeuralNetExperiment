import numpy as np


# Sigmoid activator to convert any number to a number between 0 and 1
def sigmoidActivator(x):
    return 1 / (1 + np.exp(-x))


class Neuron:
    '''
    The basic neuron. 
    - Takes two weights in np.array and one value for bias.
    - Contains a feedforward function for processing two inputs in np.array form.
    '''

    def __init__(self, weights, bias):
        self.weights = weights
        self.bias = bias

    def feedForward(self, inputs):
        z = np.dot(self.weights, inputs) + self.bias
        return sigmoidActivator(z)


class NeuralNetwork:
    '''
    - A neural network that takes 2 inputs. and has 3 neurons
    - The value of the weights and biases are extracted from an already trained neural network
    - Network is trained to take input 1. Weights in lbs -135, and 2. Height in inches - 66
    - Outputs a number between 0 and 1. 0 = male, 1 = female
    '''

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
trueCheckFemale = np.array([-7, -3])
falseCheckMale = np.array([20, 2])

print(NeuralNetwork().NeuralNetFeedForward(trueCheckFemale))  # Returns 1, true
print(NeuralNetwork().NeuralNetFeedForward(falseCheckMale))  # returns 0, false

