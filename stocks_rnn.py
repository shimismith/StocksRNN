import string

def read_data(filename):
  lines = open(filename).read().strip().lower()
  lines = lines.replace('-', ' ')
  lines = lines.split('\n')
  
  text, target = [], []

  for line in lines:
    line = line.split(',')

    # Remove punctuation and any non-alphanumeric "words"
    line_str = line[2].translate(str.maketrans('', '', string.punctuation))
    text.append([s for s in line[2].split() if s.isalpha()])
    target.append(int(line[0]))

  return text, target

def words_to_index_list(words, word_to_index):
  return [word_to_index[word] for word in words]

def load_data(filename):
  text, target = read_data(filename)
  vocab = set(sum(text, []))
  word_to_index = {word : index for (index, word) in enumerate(list(vocab))}
  vocab_size = len(word_to_index)

  data = list(zip(text, target))

  return data, word_to_index, vocab_size

import torch
import torch.nn as nn

class RNN(nn.Module):
  def __init__(self, vocab_size, hidden_size):
    super(RNN, self).__init__()

    self.hidden_size = hidden_size
    self.embedding = nn.Embedding(vocab_size, hidden_size)
    self.gru = nn.GRUCell(hidden_size, hidden_size)

    self.dense = nn.Linear(self.hidden_size, 2)

  def forward(self, inputs):
    batch_size, seq_len = inputs.size()
    hidden = torch.zeros(batch_size, self.hidden_size).cuda()  # initial hidden state

    encoded = self.embedding(inputs)  # batch_size x seq_len x hidden_size
    annotations = []

    for i in range(seq_len):
        x = encoded[:, i, :]  # Get the current time step, across the whole batch
        hidden = self.gru(x, hidden)

    output = self.dense(hidden)

    return output

import numpy as np
from tqdm import tqdm

def compute_loss(input_tensor, target_tensor, model, batch_size, criterion, optimizer):

  losses = []
  num_batches = int(np.ceil(len(data) / batch_size))
  for i in tqdm(range(num_batches)):
    start = i*batch_size
    end = start + batch_size

    # inputs = input_tensor[start:end]
    # targets = input_tensor[start:end]
    inputs = input_tensor[i].unsqueeze(0)
    targets = target_tensor[i]

    inputs = inputs.cuda()
    targets = targets.cuda()

    pred = model(inputs)
    loss = criterion(pred, targets)

    losses.append(loss.item())

    if optimizer:
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()

  mean_loss = np.mean(losses)
  return mean_loss


def train(lr, epochs, batch_size, data, word_to_index, vocab_size):

  # input_tensor = torch.stack([torch.LongTensor(words_to_index_list(pair[0], word_to_index)) for pair in data])
  input_tensor = [torch.LongTensor(words_to_index_list(pair[0], word_to_index)) for pair in data]
  target_tensor = torch.stack([torch.LongTensor([pair[1]]) for pair in data])

  num_train = int(0.8 * len(data))
  
  model = RNN(vocab_size, 30) 
  model.cuda()

  optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
  criterion = nn.CrossEntropyLoss()

  model.train()

  train_losses = []
  val_losses = []

  for epoch in range(epochs):

    train_loss = compute_loss(input_tensor[:num_train], target_tensor[:num_train], model, batch_size, criterion, optimizer)
    val_loss = compute_loss(input_tensor[num_train:], target_tensor[num_train:], model, batch_size, criterion, None)

    print("Epoch: {:3d} | Train loss: {:.3f} | Val loss: {:.3f}".format(epoch, train_loss, val_loss))
    
  return model

data, word_to_index, vocab_size = load_data('/content/drive/My Drive/stocks.csv')
train(0.001, 50, 1, data, word_to_index, vocab_size)