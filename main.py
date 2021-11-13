import numpy as np
import pandas as pd
import os
import argparse
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split

from utils.dense import Dense
from utils.activations import Tanh, Sigmoid
from utils.losses import mse, mse_prime


parser = argparse.ArgumentParser(description="Implementing neural network without any ML libraries")
# Paths
parser.add_argument('-d','--data',type=str,default='Dataset/Data.csv',
                    help="Dataset path")
parser.add_argument('-l','--label',type=str,default='Dataset/labels.csv',
                    help="Dataset path")
parser.add_argument('-p','--plot',type=str,default='Dataset',
                    help="Plot path")
parser.add_argument('-s','--save',type=str,default='Dataset',
                    help="Save path")

# Parameters
parser.add_argument('-ts','--test_size',type=int,default=1000,
                    help="Test size")
parser.add_argument('-e','--n_epoch',type=int,default=100,
                    help="Number of epochs")
parser.add_argument('-lr','--lr',type=float,default=0.01,
                    help="learning rate")
parser.add_argument('-hn','--hidden_neurones',type=int,default=100,
                   help="Neurones in the hidden layer")

args = parser.parse_args()

epochs = args.n_epoch
learning_rate = args.lr
num_hidd = args.hidden_neurones

data = pd.read_csv(args.data)
label = pd.read_csv(args.label)
plot_path = args.plot
save_path = args.save
model_name = 'neural_network'


def one_hot(y_train):
  integer_encoded = y_train
  onehot_encoded = np.zeros((len(y_train),10))
  for i in range(len(y_train)):
    onehot_encoded[i, y_train[i]] = 1
  return onehot_encoded


data = np.array(data)
label = np.array(label)
m, n = data.shape
x_train, x_test, y_train, y_test = train_test_split(data, label, test_size = args.test_size,
                                                     random_state = 4)




x_train = x_train.reshape(x_train.shape[0], 28 * 28, 1)
x_train = x_train.astype('float32')

y_train = one_hot(y_train)
y_train = y_train.reshape(y_train.shape[0], 10, 1)


x_test = x_test.reshape(x_test.shape[0], 28 * 28, 1)
x_test = x_test.astype('float32')

y_test = one_hot(y_test)
y_test = y_test.reshape(y_test.shape[0], 10, 1)

# neural network
network = [
    Dense(28 * 28, num_hidd),
    Tanh(),
    Dense(num_hidd, 10),
    Sigmoid(),
]
#print("x_train : {}, y_train : {}".format(x_train.shape, y_train.shape))


def plot(train_loss):
    plt.title("Loss after epoch: {}".format(len(train_loss)))
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.plot(list(range(len(train_loss))),train_loss,color="r",label="Train_loss")
    path = os.path.join(plot_path+'/',"loss_"+model_name+".png")
    plt.savefig(path)
    plt.close()

train_gph = []

# train
for e in range(epochs):
    error = 0
    # train on 1000 samples, since we're not training on GPU...
    for x, y in zip(x_train, y_train):
        # forward
        output = x
        for layer in network:
            output = layer.forward(output)

        # error
        error += mse(y, output)

        # backward
        grad = mse_prime(y, output)
        for layer in reversed(network):
            grad = layer.backward(grad, learning_rate)

    error /= 1000
    train_gph.append(error)
    plot(train_gph)
    print('%d/%d, error=%f' % (e + 1, epochs, error))

# test
total = 0
correct = 0
confusion_matrix = np.zeros((20,20))
for x, y in zip(x_test, y_test):
    output = x
    for layer in network:
        output = layer.forward(output)
    correct += np.argmax(output) == np.argmax(y)
    total += 1
    confusion_matrix[np.argmax(output), np.argmax(y)] += 1
    #print('pred:', np.argmax(output) , '\ttrue:', np.argmax(y))

print("Accuracy : ", correct*100/total)
confusion_matrix_df = pd.DataFrame(confusion_matrix)
confusion_matrix_df.to_csv(save_path+'/confusion_matrix.csv')