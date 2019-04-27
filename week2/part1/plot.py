import sys
import os
import matplotlib.pyplot as plt

model = input('model: ')
assert model in ('RNN', 'LSTM')
if model == 'RNN':
    path = './result/RNN_acc.txt'
else:
    path = './result/LSTM_acc.txt'
result = open('{}'.format(path)).read().splitlines()
length = list()
acc = list()
for i in range(len(result)):
    line = list(map(float, result[i].split()))
    length.append(line[0])
    acc.append(line[1])
fig = plt.figure(figsize=(10,5))
plt.scatter(length, acc)
plt.plot(length, acc, linestyle='--')
plt.xlabel('Length')
plt.ylabel('Accuracy')
fig.tight_layout()
plt.show()
fig.savefig('./result/{}_plot.eps'.format(model))