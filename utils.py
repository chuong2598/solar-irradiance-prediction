import numpy as np
from matplotlib import pyplot as plt
import os
from tqdm.auto import tqdm
import torch
import torchvision


def save_model(epoch, model, optim, train_loss, val_loss, save_path):
    check_point = {
        "epoch": epoch, 
        "model_state_dict": model.state_dict(),
        "optim_state_dict": optim.state_dict(),
        "train_loss": train_loss,
        "val_loss": val_loss
    }
    torch.save(check_point, save_path)
    model_name = save_path.split("/")[-1]
    print("Save model {}".format(model_name))

def get_auxiliary_stats():
    data = np.loadtxt("../processed_dataset/training/auxiliary_training.txt", delimiter=",")[:,7:]
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    return mean, std


# !!!! Fix later !!!!
def save_image(sequence, ground_truth, prediction, save_path):
    grid = torchvision.utils.make_grid(sequence, nrow=4)
    grid = grid.permute(1,2,0).detach().cpu().numpy()
    plt.figure(figsize=(10, 10))
    plt.title("input")

    grid = torchvision.utils.make_grid(ground_truth, nrow=6)
    grid = grid.permute(1,2,0).detach().cpu().numpy()
    plt.figure(figsize=(15, 15))
    plt.title("ground_truth")

    grid = torchvision.utils.make_grid(prediction, nrow=6)
    grid = grid.permute(1,2,0).detach().cpu().numpy()
    plt.figure(figsize=(15, 15))
    plt.title("prediction")
    plt.savefig(save_path)
    plt.clf()


def save_learning_curve(train_loss, val_loss, save_path):
    epochs = np.arange(0, len(train_loss))
    plt.plot(epochs, train_loss)
    plt.plot(epochs, val_loss)
    plt.title("Learning curve at epoch {}".format(epochs[-1]))
    plt.legend(["Training", "Validation"])
    plt.xlabel("Epoch")
    plt.ylabel("l1 loss")
    plt.savefig(save_path)
    plt.clf()

    