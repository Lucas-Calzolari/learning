import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class BinaryPerceptron():
    """ 
    
    First implementation of a perceptron, which will be able to perfectly divide a group of flowers in 2 labels. 
    
    """

    def __init__(self, X, Y, threehold=0, upperValue=1, lowerValue=-1,learningRate=0.001, maxIter=1000):
        if len(X) != len(Y):
            raise ValueError('Data samples and labels must have same length!')
        self.X = X
        self.Y = np.where(Y, upperValue, lowerValue) #The labels array is expected to be a boolean mask
        self.__W =  np.random.rand(X.shape[1])/100
        self.learningRate = learningRate
        self.threehold=threehold
        self.upperValue=upperValue
        self.lowerValue=lowerValue
        self.maxIter=maxIter

    def calibrate(self):
        amountMissclassification = 1
        i=0
        while amountMissclassification > 0 and i < self.maxIter:
            i += 1
            amountMissclassification = self.__iterateSamples()
            print(amountMissclassification)

        if (i == self.maxIter):
            print("Max amount of iterations reached!")


    def __iterateSamples(self):
        amountMissclassification = 0
        for i, sample in enumerate(self.X):
            netInput = sample.dot(self.__W)
            estimatedValue = self.upperValue if netInput > self.threehold else self.lowerValue
            targetValue = self.Y[i]
            delta = self.learningRate * (targetValue - estimatedValue) * sample
            self.__W += delta
            if (estimatedValue != targetValue):
                amountMissclassification += 1

        return amountMissclassification

# Separate data from classification label.
def extractDataAndLabels(df):
    columns = flowersDF.columns
    labels = flowersDF.loc[:, columns[-1:]].values.flatten()
    data = flowersDF.loc[:, columns[: -1]].values
    return data, labels 

# Download flower dataframe and parse with pandas 
flowersDF = pd.read_csv('../../data/iris.data',header=None)

classNameA = "Iris-setosa"
classNameB = "Iris-versicolor"
# classNameC = "Iris-virginica"

#Slice only 2 classes of dataset
groupA = flowersDF[4] == classNameA
groupB = flowersDF[4] == classNameB
dataFrame = flowersDF[groupA | groupB]
print(dataFrame)
data, labels = extractDataAndLabels(dataFrame)

# A mask for each classification
classAMask = labels == classNameA

#Create and calibrate the perceptron with flowers data
bp = BinaryPerceptron(data, classAMask)
bp.calibrate()

#Although we will be using 4 dimensions on the perceptron, we will plot only 2 on the graph bellow, since if a group is divisible
#by 2 dimensions only, it is also perfectly divisible by 4.
  
plt.scatter(flowersDF[groupA][0], flowersDF[groupA][1], color='red', marker='o', label=classNameA)
plt.scatter(flowersDF[groupB][0], flowersDF[groupB][1], color='blue', marker='x', label=classNameB)
plt.xlabel('sepal length [cm]')
plt.ylabel('petal length [cm]')
plt.legend(loc='upper left')
plt.show() 
