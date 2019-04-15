"""
This module implements training and evaluation of a Convolutional Neural Network in PyTorch.
You should fill in code into indicated sections.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np
import os
from convnet_pytorch import ConvNet
import cifar10_utils
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# Default constants
LEARNING_RATE_DEFAULT = 1e-4
BATCH_SIZE_DEFAULT = 32
MAX_STEPS_DEFAULT = 5000
EVAL_FREQ_DEFAULT = 500
OPTIMIZER_DEFAULT = 'ADAM'

# Directory in which cifar data is saved
DATA_DIR_DEFAULT = './cifar10/cifar-10-batches-py'

FLAGS = None

def accuracy(predictions, targets):
  """
  Computes the prediction accuracy, i.e. the average of correct predictions
  of the network.
  
  Args:
    predictions: 2D float array of size [batch_size, n_classes]
    labels: 2D int array of size [batch_size, n_classes]
            with one-hot encoding. Ground truth labels for
            each sample in the batch
  Returns:
    accuracy: scalar float, the accuracy of predictions,
              i.e. the average correct predictions over the whole batch
  
  TODO:
  Implement accuracy computation.
  """

  ########################
  # PUT YOUR CODE HERE  #
  #######################
  match = 0
  bSize = targets.shape[0]
  pred = predictions.argmax(dim=1)
  target = targets.argmax(dim=1)
  match += (pred == target).sum().item()
  accuracy = match / bSize
  ########################
  # END OF YOUR CODE    #
  #######################

  return accuracy

def train():
  """
  Performs training and evaluation of ConvNet model. 

  TODO:
  Implement training and evaluation of ConvNet model. Evaluate your model on the whole test set each eval_freq iterations.
  """

  ### DO NOT CHANGE SEEDS!
  # Set the random seeds for reproducibility
  np.random.seed(42)

  ########################
  # PUT YOUR CODE HERE  #
  #######################
  device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
  cifar10 = cifar10_utils.get_cifar10(FLAGS.data_dir)
  x, y = cifar10['train'].next_batch(FLAGS.batch_size)
  x = torch.from_numpy(x).float().to(device)
  y = torch.from_numpy(y).float().to(device)
  n_classes = y.shape[1]
  n_channels = x.shape[1]
  CNN = ConvNet(n_channels, n_classes)
  CNN.to(device)
  if OPTIMIZER_DEFAULT == 'SGD':
    optimizer = optim.SGD(CNN.parameters())
  elif OPTIMIZER_DEFAULT == 'ADAM':
    optimizer = optim.Adam(CNN.parameters())
  else:
    print('Try SGD or ADAM...')
  loss = nn.CrossEntropyLoss()
  l_list = list()
  t_list = list()
  train_acc = list()
  test_acc = list()
  print('\nTraining...')
  for i in range(FLAGS.max_steps):
    optimizer.zero_grad()
    s_pred = CNN(x)
    f_loss = loss(s_pred, y.argmax(dim=1))
    f_loss.backward()
    optimizer.step()
    if i % FLAGS.eval_freq == 0:
      l_list.append(round(f_loss.item(), 3))
      train_acc.append(accuracy(s_pred, y))
      test_size = cifar10['test'].labels.shape[0]
      iter_num = 10
      tmp_size = int(test_size / iter_num)
      tmp_correct = 0
      tmp_loss = 0
      for j in range(iter_num):
        t_x, t_y = cifar10['test'].next_batch(tmp_size)
        t_x = torch.from_numpy(t_x).float().to(device)
        t_y = torch.from_numpy(t_y).float().to(device)
        t_pred = CNN(t_x)
        tmp_loss += loss(t_pred, t_y.argmax(dim=1)).item()
        tmp_correct += accuracy(t_pred, t_y) * tmp_size
      test_acc.append(tmp_correct/test_size)
      t_list.append(round(tmp_loss/iter_num, 3))
    x, y = cifar10['train'].next_batch(FLAGS.batch_size)
    x = torch.from_numpy(x).float().to(device)
    y = torch.from_numpy(y).float().to(device)
  print('Done!\n')
  print('Training Losses:', l_list)
  print('Test Losses:', t_list)
  print('Training Accuracies:', train_acc)
  print('Test Accuracies:', test_acc)
  print('Best Test Accuracy:', max(test_acc))
  iterations = np.arange(1, len(l_list)+1)
  fig, axs = plt.subplots(1, 2, figsize=(10,5))
  axs[0].plot(iterations, train_acc, iterations, test_acc)
  axs[0].set_xlabel('Iteration')
  axs[0].set_ylabel('Accuracy')
  axs[0].legend(('train', 'test'))
  axs[1].plot(iterations, l_list, iterations, t_list)
  axs[1].set_xlabel('Iteration')
  axs[1].set_ylabel('Loss')
  axs[1].legend(('train', 'test'))
  fig.tight_layout()
  plt.show()
  ########################
  # END OF YOUR CODE    #
  #######################

def print_flags():
  """
  Prints all entries in FLAGS variable.
  """
  for key, value in vars(FLAGS).items():
    print(key + ' : ' + str(value))

def main():
  """
  Main function
  """
  # Print all Flags to confirm parameter settings
  print_flags()

  if not os.path.exists(FLAGS.data_dir):
    os.makedirs(FLAGS.data_dir)

  # Run the training operation
  train()

if __name__ == '__main__':
  # Command line arguments
  parser = argparse.ArgumentParser()
  parser.add_argument('--learning_rate', type = float, default = LEARNING_RATE_DEFAULT,
                      help='Learning rate')
  parser.add_argument('--max_steps', type = int, default = MAX_STEPS_DEFAULT,
                      help='Number of steps to run trainer.')
  parser.add_argument('--batch_size', type = int, default = BATCH_SIZE_DEFAULT,
                      help='Batch size to run trainer.')
  parser.add_argument('--eval_freq', type=int, default=EVAL_FREQ_DEFAULT,
                        help='Frequency of evaluation on the test set')
  parser.add_argument('--data_dir', type = str, default = DATA_DIR_DEFAULT,
                      help='Directory for storing input data')
  FLAGS, unparsed = parser.parse_known_args()

  main()