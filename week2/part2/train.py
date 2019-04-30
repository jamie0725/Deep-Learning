# MIT License
#
# Copyright (c) 2017 Tom Runia
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to conditions.
#
# Author: Deep Learning Course | Fall 2018
# Date Created: 2018-09-04
################################################################################

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import time
from datetime import datetime
import argparse

import numpy as np
import random

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from dataset import TextDataset
from model import TextGenerationModel

################################################################################

def compute_accuracy(predictions, targets):
  """
  Computes the prediction accuracy, i.e. the average of correct predictions
  of the network.
  
  Args:
    predictions: 2D float array of size [batch_size, n_classes]
    targets: Ground truth labels for each sample in the batch
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
  size = targets.shape[0] * targets.shape[1]
  pred = predictions.argmax(dim=2)
  match += (pred == targets).sum().item()
  accuracy = match / size
  ########################
  # END OF YOUR CODE    #
  #######################

  return accuracy

def train(config):

    # Print all configs to confirm parameter settings
    print_flags()
    assert config.sampling_method in ('greedy', 'random')

    # Initialize the device which to run the model on
    device = torch.device(config.device)

    # Initialize the dataset and data loader (note the +1)
    dataset = TextDataset(filename=config.txt_file,
                          seq_length=config.seq_length)
    data_loader = DataLoader(dataset, config.batch_size, num_workers=1)

    # Initialize the model that we are going to use
    model = TextGenerationModel(batch_size=config.batch_size,
                                seq_length=config.seq_length,
                                vocabulary_size=dataset.vocab_size,
                                dropout=1-config.dropout_keep_prob,
                                lstm_num_hidden=config.lstm_num_hidden,
                                lstm_num_layers=config.lstm_num_layers,
                                device=device)
    model.to(device)

    # Setup the loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    epoch = 10

    # Store some measures
    los = list()
    iteration = list()
    acc = list()
    max_step = 0

    for i in range(epoch):
      for step, (batch_inputs, batch_targets) in enumerate(data_loader):

          # Only for time measurement of step through network
          t1 = time.time()

          model.train()
          optimizer.zero_grad()

          batch_inputs = torch.stack(batch_inputs).to(device)
          batch_targets = torch.stack(batch_targets).to(device)

          h_0 = torch.zeros(config.lstm_num_layers, batch_inputs.shape[1], config.lstm_num_hidden).to(device)
          c_0 = torch.zeros(config.lstm_num_layers, batch_inputs.shape[1], config.lstm_num_hidden).to(device)

          pred, _, _ = model(batch_inputs, h_0, c_0)
          accuracy = compute_accuracy(pred, batch_targets)
          pred = pred.permute(1, 2, 0)
          batch_targets = batch_targets.permute(1, 0)
          loss = criterion(pred, batch_targets)
          loss.backward()
          torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.max_norm)
          optimizer.step()

          # Just for time measurement
          t2 = time.time()
          examples_per_second = config.batch_size/float(t2-t1)

          if (step + i * max_step) % config.print_every == 0:

              print("[{}] Train Step {:04d}/{:04d}, Batch Size = {}, Examples/Sec = {:.2f}, "
                    "Accuracy = {:.2f}, Loss = {:.3f}".format(
                      datetime.now().strftime("%Y-%m-%d %H:%M"), step + i * max_step,
                      int(config.train_steps), config.batch_size, examples_per_second,
                      accuracy, loss
              ))
              iteration.append(step + i * max_step)
              acc.append(accuracy)
              los.append(loss)
              if max_step < step:
                max_step = step

          if (step + i * max_step) % config.sample_every == 0:
              model.eval()
              batch_sample = 5
              generated = [dataset._char_to_ix[random.choice(dataset._chars)] for c in range(batch_sample)]
              generated = torch.LongTensor(generated).view(-1, batch_sample).to(device)
              for l in range(config.genreate_length - 1):
                if l == 0:
                  h_s = torch.zeros(config.lstm_num_layers, batch_sample, config.lstm_num_hidden).to(device)
                  c_s = torch.zeros(config.lstm_num_layers, batch_sample, config.lstm_num_hidden).to(device)
                  gen, h_s, c_s = model(generated, h_s, c_s)
                else:
                  gen, h_s, c_s = model(gen, h_s, c_s)
                if config.sampling_method == 'greedy':
                  gen = gen.argmax(dim=2)
                else:
                  gen = nn.functional.softmax(gen/config.temperature, dim=2)
                  dist = torch.distributions.categorical.Categorical(gen)
                  gen = dist.sample()
                generated = torch.cat((generated, gen))
              generated = generated.t()
              sentence = [dataset.convert_to_string(idx) for idx in generated.tolist()]
              if config.sampling_method == 'random':
                with open('{}/generate_{}_{}_{}.txt'.format(config.summary_path, datetime.now().strftime("%Y-%m-%d"), config.sampling_method, config.temperature), 'a', encoding='utf-8') as file:
                  file.write('--------------\n')
                  file.write('Training Step: {}\n'.format(step + i * max_step))
                  file.write('--------------\n')
                  for sen in sentence:
                    file.write('{}\n'.format(sen))
                  file.write('\n')
                  file.close()   
              else:
                with open('{}/generate_{}_{}.txt'.format(config.summary_path, datetime.now().strftime("%Y-%m-%d"), config.sampling_method), 'a', encoding='utf-8') as file:
                  file.write('--------------\n')
                  file.write('Training Step: {}\n'.format(step + i * max_step))
                  file.write('--------------\n')
                  for sen in sentence:
                    file.write('{}\n'.format(sen))
                  file.write('\n')
                  file.close()

          if (step + i * max_step) == config.train_steps:
              # If you receive a PyTorch data-loader error, check this bug report:
              # https://github.com/pytorch/pytorch/pull/9655
              break

      if (step + i * max_step) == config.train_steps:
        break

    print('Done training.')
    fig, axs = plt.subplots(1, 2, figsize=(10,5))
    axs[0].plot(iteration, acc)
    axs[0].set_xlabel('Iteration')
    axs[0].set_ylabel('Accuracy')
    axs[1].plot(iteration, los)
    axs[1].set_xlabel('Iteration')
    axs[1].set_ylabel('Loss')
    fig.tight_layout()
    plt.show()

def print_flags():
  """
  Prints all entries in config variable.
  """
  for key, value in vars(config).items():
    print(key + ' : ' + str(value))

 ################################################################################
 ################################################################################

if __name__ == "__main__":

    # Parse training configuration
    parser = argparse.ArgumentParser()

    # Model params
    parser.add_argument('--txt_file', type=str, required=True, help="Path to a .txt file to train on")
    parser.add_argument('--seq_length', type=int, default=30, help='Length of an input sequence')
    parser.add_argument('--lstm_num_hidden', type=int, default=128, help='Number of hidden units in the LSTM')
    parser.add_argument('--lstm_num_layers', type=int, default=2, help='Number of LSTM layers in the model')

    # Training params
    parser.add_argument('--batch_size', type=int, default=64, help='Number of examples to process in a batch')
    parser.add_argument('--learning_rate', type=float, default=2e-3, help='Learning rate')
    parser.add_argument('--generate_length', type=int, default=30, help='Length of genreated sentence')
    parser.add_argument('--sampling_method', type=str, default="greedy", help='Method of sampling')
    parser.add_argument('--temperature', type=float, default=1.0, help='Temperature of sampling')

    # It is not necessary to implement the following three params, but it may help training.
    parser.add_argument('--learning_rate_decay', type=float, default=0.96, help='Learning rate decay fraction')
    parser.add_argument('--learning_rate_step', type=int, default=5000, help='Learning rate step')
    parser.add_argument('--dropout_keep_prob', type=float, default=1.0, help='Dropout keep probability')

    parser.add_argument('--train_steps', type=int, default=1e6, help='Number of training steps')
    parser.add_argument('--max_norm', type=float, default=5.0, help='--')

    # Misc params
    parser.add_argument('--summary_path', type=str, default="./summaries/", help='Output path for summaries')
    parser.add_argument('--print_every', type=int, default=5, help='How often to print training progress')
    parser.add_argument('--sample_every', type=int, default=100, help='How often to sample from the model')
    parser.add_argument('--device', type=str, default="cuda:0", help="Training device 'cpu' or 'cuda:0'")

    config = parser.parse_args()

    # Train the model
    train(config)
