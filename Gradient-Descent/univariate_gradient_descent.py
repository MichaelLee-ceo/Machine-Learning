import numpy as np
import matplotlib.pyplot as plt

def lossFunction(w):
    return w ** 2

def derivativeL(w):
    return 2 * w

def gradientDescent(start, dfunction, learningRate, epochs):
    gradient = []
    gradient.append(start)
    prevWeight = start

    for i in range(epochs):
        newWeight = prevWeight - learningRate * dfunction(prevWeight)
        gradient.append(newWeight)
        prevWeight = newWeight
    return np.array(gradient)

start = 5
epochs = 20
lr = 0.1
gradient = gradientDescent(start, derivativeL, lr, epochs)
print(gradient)

x = np.arange(-5.5, 5.5, 0.01)
plt.plot(x, lossFunction(x), color='black')
plt.plot(gradient, lossFunction(gradient), label='lr{}'.format(lr))
plt.scatter(gradient, lossFunction(gradient))


for i in range(1, epochs+1):
    plt.annotate('', xy=(gradient[i], lossFunction(gradient)[i]),
                   xytext=(gradient[i-1], lossFunction(gradient)[i-1]),
                   arrowprops={'arrowstyle': '->', 'color': 'm', 'lw': 1},
                   va='center', ha='center')

plt.legend()
plt.show()