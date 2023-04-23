import random
import numpy as np
from pprint import pprint
import matplotlib.pyplot as plt

data_input = 4
data_output = 3
neurons = 20


def relu(t):
    return np.maximum(t, 0)


def softmax(x):
    return np.exp(x) / np.sum(np.exp(x))


def batch_softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=1, keepdims=True)


def sparse_cross_entropy(z, y):
    return -np.log(z[0, y])


def batch_sparse_cross_entropy(z, y):
    return -np.log(np.array([z[j, y[j]] for j in range(len(y))]))


def batch_to_full(y, number_classes):
    container = np.zeros((len(y), number_classes))
    for i, j in enumerate(y):
        container[i, j] = 1
    return container


def to_full(y, number_classes):
    container = np.zeros((1, number_classes))
    container[0, y] = 1
    return container


def relu_deriv(t):
    return (t >= 0).astype(float)


dataset = [(np.array([[6., 5., 4., 1.]]), 1),
           (np.array([[9., 5., 5., 1.]]), 0),
           (np.array([[1., 3., 3., 0.]]), 2),
           (np.array([[15., 2., 6., 1.]]), 2),
           (np.array([[11., 5., 4., 1.]]), 0),
           (np.array([[4., 4., 3., 1.]]), 1),
           (np.array([[8., 5., 10., 1.]]), 0),
           (np.array([[7., 4., 5., 1.]]), 1),
           (np.array([[1., 3., 9., 1.]]), 1),
           (np.array([[8., 2., 4., 1.]]), 2),
           (np.array([[12., 3., 6., 0.]]), 2),
           (np.array([[4., 4., 9., 0.]]), 1),
           (np.array([[15., 5., 9., 1.]]), 0),
           (np.array([[8., 3., 8., 1.]]), 1),
           (np.array([[3., 4., 5., 1.]]), 1),
           (np.array([[8., 3., 4., 1.]]), 2),
           (np.array([[3., 5., 4., 0.]]), 1),
           (np.array([[9., 4., 5., 0.]]), 1),
           (np.array([[9., 2., 4., 1.]]), 2),
           (np.array([[11., 4., 5., 1.]]), 0),
           (np.array([[5., 4., 8., 0.]]), 1),
           (np.array([[13., 3., 7., 1.]]), 2),
           (np.array([[4., 5., 8., 1.]]), 0),
           (np.array([[13., 3., 5., 0.]]), 2),
           (np.array([[14., 2., 3., 1.]]), 2),
           (np.array([[7., 4., 6., 0.]]), 1),
           (np.array([[6., 3., 9., 1.]]), 1),
           (np.array([[4., 4., 9., 0.]]), 1),
           (np.array([[14., 2., 10., 1.]]), 1),
           (np.array([[3., 4., 9., 0.]]), 1),
           (np.array([[8., 5., 7., 1.]]), 0),
           (np.array([[14., 5., 9., 0.]]), 0),
           (np.array([[13., 5., 9., 1.]]), 0),
           (np.array([[7., 3., 10., 1.]]), 1),
           (np.array([[2., 3., 10., 0.]]), 1),
           (np.array([[4., 2., 8., 0.]]), 1),
           (np.array([[10., 2., 8., 1.]]), 2),
           (np.array([[4., 2., 10., 0.]]), 1),
           (np.array([[11., 4., 4., 0.]]), 1),
           (np.array([[1., 3., 7., 0.]]), 1),
           (np.array([[4., 2., 6., 1.]]), 2),
           (np.array([[8., 2., 6., 0.]]), 2),
           (np.array([[4., 3., 6., 0.]]), 2),
           (np.array([[2., 2., 6., 0.]]), 2),
           (np.array([[11., 4., 7., 0.]]), 1),
           (np.array([[5., 2., 9., 0.]]), 1),
           (np.array([[7., 3., 4., 0.]]), 2),
           (np.array([[14., 4., 4., 0.]]), 1),
           (np.array([[8., 5., 10., 1.]]), 0),
           (np.array([[8., 4., 9., 1.]]), 1),
           (np.array([[10., 5., 10., 1.]]), 0),
           (np.array([[10., 2., 5., 0.]]), 2),
           (np.array([[11., 5., 4., 1.]]), 0),
           (np.array([[4., 4., 7., 0.]]), 1),
           (np.array([[7., 3., 4., 1.]]), 2),
           (np.array([[4., 5., 7., 0.]]), 0),
           (np.array([[10., 2., 9., 1.]]), 1),
           (np.array([[3., 2., 9., 0.]]), 2),
           (np.array([[11., 5., 6., 1.]]), 0),
           (np.array([[11., 2., 9., 1.]]), 1),
           (np.array([[12., 2., 10., 1.]]), 1),
           (np.array([[8., 2., 7., 0.]]), 1),
           (np.array([[6., 5., 8., 0.]]), 0),
           (np.array([[3., 5., 6., 1.]]), 0),
           (np.array([[10., 2., 3., 0.]]), 2),
           (np.array([[7., 4., 6., 0.]]), 1),
           (np.array([[12., 2., 8., 1.]]), 2),
           (np.array([[6., 3., 4., 0.]]), 2),
           (np.array([[15., 3., 8., 0.]]), 2),
           (np.array([[15., 2., 10., 1.]]), 1),
           (np.array([[1., 5., 4., 0.]]), 1),
           (np.array([[1., 4., 5., 1.]]), 1),
           (np.array([[6., 2., 9., 1.]]), 1),
           (np.array([[6., 3., 6., 1.]]), 2),
           (np.array([[11., 2., 3., 0.]]), 2),
           (np.array([[1., 5., 8., 1.]]), 1),
           (np.array([[1., 5., 10., 0.]]), 1),
           (np.array([[6., 4., 10., 1.]]), 1),
           (np.array([[13., 3., 3., 0.]]), 2),
           (np.array([[5., 5., 10., 1.]]), 0),
           (np.array([[1., 2., 5., 0.]]), 2),
           (np.array([[12., 3., 3., 1.]]), 2),
           (np.array([[14., 2., 7., 1.]]), 2),
           (np.array([[11., 3., 9., 1.]]), 1),
           (np.array([[1., 4., 4., 1.]]), 1),
           (np.array([[13., 5., 6., 1.]]), 0),
           (np.array([[1., 3., 8., 0.]]), 1),
           (np.array([[4., 5., 7., 0.]]), 0),
           (np.array([[14., 3., 9., 0.]]), 1),
           (np.array([[5., 3., 3., 1.]]), 2),
           (np.array([[2., 3., 4., 1.]]), 2),
           (np.array([[9., 2., 10., 0.]]), 1),
           (np.array([[14., 5., 4., 0.]]), 0),
           (np.array([[5., 4., 3., 0.]]), 1),
           (np.array([[9., 5., 10., 1.]]), 0),
           (np.array([[6., 2., 3., 0.]]), 2),
           (np.array([[6., 4., 6., 1.]]), 1),
           (np.array([[2., 3., 8., 0.]]), 1),
           (np.array([[5., 3., 10., 0.]]), 1),
           (np.array([[13., 2., 5., 0.]]), 2),
           (np.array([[7., 2., 7., 1.]]), 1),
           (np.array([[10., 5., 7., 1.]]), 0),
           (np.array([[6., 4., 10., 0.]]), 1),
           (np.array([[15., 4., 3., 0.]]), 1),
           (np.array([[15., 2., 8., 0.]]), 2),
           (np.array([[5., 2., 7., 1.]]), 1),
           (np.array([[7., 5., 9., 0.]]), 0),
           (np.array([[13., 4., 10., 0.]]), 1),
           (np.array([[2., 2., 3., 0.]]), 2),
           (np.array([[10., 4., 10., 1.]]), 1),
           (np.array([[3., 4., 6., 1.]]), 1),
           (np.array([[8., 3., 7., 1.]]), 1),
           (np.array([[3., 2., 8., 1.]]), 2),
           (np.array([[7., 2., 6., 0.]]), 2),
           (np.array([[8., 4., 3., 0.]]), 1),
           (np.array([[1., 3., 9., 1.]]), 1),
           (np.array([[5., 4., 7., 0.]]), 1),
           (np.array([[15., 2., 10., 1.]]), 1),
           (np.array([[6., 4., 6., 1.]]), 1),
           (np.array([[15., 3., 7., 0.]]), 2),
           (np.array([[10., 4., 8., 0.]]), 1),
           (np.array([[3., 5., 9., 1.]]), 0),
           (np.array([[12., 5., 8., 1.]]), 0),
           (np.array([[14., 2., 6., 1.]]), 2),
           (np.array([[12., 2., 3., 1.]]), 2),
           (np.array([[14., 2., 9., 0.]]), 1),
           (np.array([[7., 2., 5., 1.]]), 2),
           (np.array([[11., 3., 7., 1.]]), 2),
           (np.array([[6., 5., 9., 1.]]), 0),
           (np.array([[5., 5., 6., 0.]]), 0),
           (np.array([[10., 3., 3., 1.]]), 2),
           (np.array([[13., 2., 4., 1.]]), 2),
           (np.array([[2., 5., 4., 0.]]), 1),
           (np.array([[6., 2., 6., 0.]]), 2),
           (np.array([[15., 5., 8., 0.]]), 0),
           (np.array([[5., 5., 9., 1.]]), 0),
           (np.array([[2., 3., 3., 0.]]), 2),
           (np.array([[11., 3., 10., 0.]]), 1),
           (np.array([[14., 5., 3., 0.]]), 1),
           (np.array([[9., 4., 8., 1.]]), 1),
           (np.array([[6., 4., 4., 0.]]), 1),
           (np.array([[9., 5., 6., 1.]]), 0),
           (np.array([[2., 4., 7., 1.]]), 1),
           (np.array([[9., 5., 6., 0.]]), 0),
           (np.array([[6., 5., 3., 1.]]), 1),
           (np.array([[7., 4., 6., 1.]]), 1),
           (np.array([[2., 3., 10., 1.]]), 1),
           (np.array([[15., 3., 6., 0.]]), 2),
           (np.array([[6., 2., 5., 0.]]), 2),
           (np.array([[7., 4., 9., 1.]]), 1),
           (np.array([[1., 5., 3., 1.]]), 1),
           (np.array([[7., 2., 6., 1.]]), 2),
           (np.array([[12., 5., 9., 1.]]), 0),
           (np.array([[1., 2., 8., 0.]]), 2),
           (np.array([[5., 3., 4., 0.]]), 2),
           (np.array([[14., 5., 8., 0.]]), 0),
           (np.array([[4., 5., 7., 1.]]), 0),
           (np.array([[5., 2., 8., 0.]]), 1),
           (np.array([[9., 5., 6., 1.]]), 0),
           (np.array([[9., 5., 9., 1.]]), 0),
           (np.array([[7., 4., 5., 1.]]), 1),
           (np.array([[2., 4., 8., 0.]]), 1),
           (np.array([[13., 5., 5., 0.]]), 0),
           (np.array([[4., 4., 6., 0.]]), 1),
           (np.array([[7., 5., 4., 1.]]), 1),
           (np.array([[3., 5., 7., 1.]]), 0),
           (np.array([[8., 5., 6., 1.]]), 0),
           (np.array([[12., 3., 6., 1.]]), 2),
           (np.array([[4., 4., 10., 1.]]), 1),
           (np.array([[8., 2., 7., 0.]]), 1),
           (np.array([[8., 3., 10., 1.]]), 1),
           (np.array([[11., 3., 5., 0.]]), 2),
           (np.array([[3., 5., 3., 0.]]), 1),
           (np.array([[2., 4., 3., 1.]]), 1),
           (np.array([[6., 5., 8., 1.]]), 0),
           (np.array([[14., 2., 4., 1.]]), 2),
           (np.array([[2., 3., 6., 1.]]), 2),
           (np.array([[13., 5., 6., 0.]]), 0),
           (np.array([[5., 2., 4., 1.]]), 2),
           (np.array([[15., 4., 4., 0.]]), 1),
           (np.array([[8., 3., 3., 0.]]), 2),
           (np.array([[6., 3., 8., 0.]]), 1),
           (np.array([[3., 5., 8., 1.]]), 0),
           (np.array([[15., 5., 10., 0.]]), 0),
           (np.array([[6., 5., 5., 1.]]), 0),
           (np.array([[11., 5., 4., 1.]]), 0),
           (np.array([[1., 4., 8., 0.]]), 1),
           (np.array([[8., 5., 5., 0.]]), 0),
           (np.array([[5., 5., 6., 0.]]), 0),
           (np.array([[8., 5., 10., 0.]]), 0),
           (np.array([[4., 3., 4., 0.]]), 2),
           (np.array([[12., 2., 9., 1.]]), 1),
           (np.array([[11., 2., 4., 0.]]), 2),
           (np.array([[15., 4., 5., 0.]]), 1),
           (np.array([[2., 2., 4., 0.]]), 2),
           (np.array([[1., 3., 8., 0.]]), 1),
           (np.array([[9., 3., 3., 1.]]), 2),
           (np.array([[3., 2., 4., 1.]]), 2),
           (np.array([[14., 3., 6., 1.]]), 2),
           (np.array([[13., 3., 7., 1.]]), 2),
           (np.array([[11., 4., 7., 1.]]), 0),
           (np.array([[4., 4., 5., 1.]]), 1),
           (np.array([[6., 4., 5., 1.]]), 1),
           (np.array([[12., 2., 7., 1.]]), 2),
           (np.array([[13., 4., 7., 0.]]), 1),
           (np.array([[15., 5., 6., 1.]]), 0),
           (np.array([[13., 4., 9., 1.]]), 1),
           (np.array([[15., 4., 8., 1.]]), 0),
           (np.array([[9., 5., 3., 1.]]), 1),
           (np.array([[8., 5., 10., 1.]]), 0),
           (np.array([[11., 5., 8., 0.]]), 0),
           (np.array([[13., 2., 4., 1.]]), 2),
           (np.array([[6., 3., 3., 0.]]), 2),
           (np.array([[10., 2., 9., 1.]]), 1),
           (np.array([[5., 4., 4., 1.]]), 1),
           (np.array([[2., 4., 5., 0.]]), 1),
           (np.array([[3., 5., 5., 1.]]), 0),
           (np.array([[2., 5., 7., 1.]]), 1),
           (np.array([[1., 2., 7., 1.]]), 2),
           (np.array([[10., 2., 9., 1.]]), 1),
           (np.array([[15., 2., 8., 0.]]), 2),
           (np.array([[4., 4., 7., 1.]]), 1),
           (np.array([[14., 4., 10., 1.]]), 1),
           (np.array([[15., 2., 4., 0.]]), 2),
           (np.array([[4., 2., 5., 0.]]), 2),
           (np.array([[15., 5., 8., 0.]]), 0),
           (np.array([[4., 3., 8., 1.]]), 1),
           (np.array([[14., 4., 9., 1.]]), 1),
           (np.array([[12., 2., 10., 1.]]), 1),
           (np.array([[14., 2., 5., 0.]]), 2),
           (np.array([[1., 2., 3., 0.]]), 2),
           (np.array([[5., 3., 6., 1.]]), 2),
           (np.array([[12., 5., 10., 0.]]), 0),
           (np.array([[13., 3., 5., 1.]]), 2),
           (np.array([[7., 2., 6., 1.]]), 2),
           (np.array([[12., 5., 5., 1.]]), 0),
           (np.array([[4., 4., 9., 1.]]), 1),
           (np.array([[8., 5., 10., 1.]]), 0),
           (np.array([[5., 5., 6., 1.]]), 0),
           (np.array([[9., 5., 10., 1.]]), 0)]

weight1 = np.random.rand(data_input, neurons)
b1 = np.random.rand(1, neurons)
weight2 = np.random.rand(neurons, data_output)
b2 = np.random.rand(1, data_output)

weight1 = (weight1 - 0.5) * 2 * np.sqrt(1 / data_input)
b1 = (b1 - 0.5) * 2 * np.sqrt(1 / data_input)
weight2 = (weight2 - 0.5) * 2 * np.sqrt(1 / neurons)
b2 = (b2 - 0.5) * 2 * np.sqrt(1 / neurons)

ALPHA_SEED = 0.0002
N = 1500
BATCH = 50

loss_array = []

for ep in range(N):
    random.shuffle(dataset)
    for i in range(len(dataset) // BATCH):
        batch_x, batch_y = zip(*dataset[i * BATCH: i * BATCH + BATCH])
        x = np.concatenate(batch_x, axis=0)
        y = np.array(batch_y)

        # Forward1
        t1 = x @ weight1 + b1
        h1 = relu(t1)
        t2 = h1 @ weight2 + b2
        z = batch_softmax(t2)
        E = np.sum(batch_sparse_cross_entropy(z, y))

        # Backward
        y_full = batch_to_full(y, data_output)
        dError_dt2 = z - y_full
        dError_dW2 = h1.T @ dError_dt2
        dError_db2 = np.sum(dError_dt2, axis=0, keepdims=True)
        dError_dh1 = dError_dt2 @ weight2.T
        dError_dt1 = dError_dh1 * relu_deriv(t1)
        dError_dW1 = x.T @ dError_dt1
        dError_db1 = np.sum(dError_dt1, axis=0, keepdims=True)

        # Update
        weight1 = weight1 - ALPHA_SEED * dError_dW1
        b1 = b1 - ALPHA_SEED * dError_db1
        weight2 = weight2 - ALPHA_SEED * dError_dW2
        b2 = b2 - ALPHA_SEED * dError_db2

        loss_array.append(E)


def predict(x):
    return batch_softmax(relu(x @ weight1 + b1) @ weight2 + b2)


def calculate_accuracy():
    correct = 0
    for x, y in dataset:
        if np.argmax(predict(x)) == y:
            correct += 1
    return correct / len(dataset)


print("Точность:", calculate_accuracy())

plt.plot(loss_array)
plt.show()
pprint(weight1)
print()
pprint(b1)
print()
pprint(weight2)
print()
pprint(b2)