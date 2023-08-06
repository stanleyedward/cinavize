import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable
#TRAINING
nb_epoch = 50
for epoch in range(1, nb_epoch + 1):
  train_loss = 0
  s = 0.
  for id_user in range(nb_users):
    input = Variable(training_set[id_user]).unsqueeze(0)
    target = input.clone()
    if torch.sum(target.data > 0) > 0:
      output = sae(input)
      target.require_grad = False
      output[target == 0] = 0
      loss = criterion(output, target)
      mean_corrector = nb_movies/float(torch.sum(target.data > 0) + 1e-10)
      loss.backward()
      train_loss += np.sqrt(loss.data*mean_corrector)
      s += 1.
      optimizer.step()
  print('epoch: '+str(epoch)+' loss: '+ str(train_loss/s))
