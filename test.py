from utils import *
import torch

forecast_horizon = "1h"
epoch = 49
training_auxiliary_path = f"../checkpoint/alex_net_{forecast_horizon}/auxiliary_lstm_{epoch}epoch.pt"
auxiliary_lstm_checkpoint = torch.load(training_auxiliary_path)

train_loss = auxiliary_lstm_checkpoint["train_loss"]
val_loss = auxiliary_lstm_checkpoint["val_loss"]

min_epoch = np.argmin(val_loss)
print(min_epoch)
print(train_loss[min_epoch])
print(val_loss[min_epoch])
