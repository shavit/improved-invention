from numpy import array, asarray, dot, random, transpose
from scipy import special

class Network(object):

    def __init__(self, iNodes, hNodes, oNodes, lRate):
        self.__inputNodes = iNodes
        self.__hiddenNodes = hNodes
        self.__outputNodes = oNodes
        self.__learningRate = lRate

        self.__weightsInputHidden = random.normal(0.0,
            pow(self.__inputNodes, -0.5),
            (self.__hiddenNodes, self.__inputNodes))
        self.__weightsHiddenOutput = random.normal(0.0,
            pow(self.__hiddenNodes, -0.5),
            (self.__outputNodes, self.__hiddenNodes))

        self.__activationFunction = lambda x: special.expit(x)

    def train(self, inputsList, targetsList):
        inputs = array(asarray(inputsList), ndmin=2).T
        targets = array(asarray(targetsList), ndmin=2).T

        hiddenInputs = dot(self.__weightsInputHidden, inputs)
        hiddenOutputs = self.__activationFunction(hiddenInputs)

        finalInputs = dot(self.__weightsHiddenOutput, hiddenOutputs)
        finalOutputs = self.__activationFunction(finalInputs)

        outputsErrors = targets - finalOutputs
        hiddenErrors = dot(self.__weightsHiddenOutput.T, outputsErrors)
        
        self.__weightsHiddenOutput += self.__learningRate * dot(
            (outputsErrors * finalOutputs * (1.0 - finalOutputs)),
            transpose(hiddenOutputs))

    def query(self, inputsList):
        inputs = array(asarray(inputsList), ndmin=2).T
        hiddenInputs = dot(self.__weightsInputHidden, inputs)

        hiddenOutputs = self.__activationFunction(hiddenInputs)
        finalInputs = dot(self.__weightsHiddenOutput, hiddenOutputs)

        return self.__activationFunction(finalInputs)
