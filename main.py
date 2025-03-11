import numpy as np
import torch
import time
import sys
import os
import argparse
import tqdm
import qpth
import pickle
import cvxpylayers
import cvxpy as cp
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter

from loss import *
from models import *
from data import *

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    args = parser.parse_args()

    input_dim   = 64
    output_dim  = 10
    num_samples = 2048

    train_loader, test_loader = genData(input_dim, output_dim, num_samples)

    model = MLP(input_dim, output_dim)
    learning_rate = 1e-4
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    num_epochs = 100
    loss_fn = torch.nn.MSELoss()
    writer = SummaryWriter()

    for epoch in range(num_epochs):
        train_ts_loss_list, test_ts_loss_list = [], []
        train_df_loss_list, test_df_loss_list = [], []
        train_food_loss_list, test_food_loss_list = [], []
        for i, (x, y) in enumerate(train_loader):
            y_pred = model(x)
            ts_loss = loss_fn(y_pred, y)
            df_loss = df_loss_fn(y_pred, y)
            food_loss = food_loss_fn(y_pred, y)

            loss = ts_loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            train_ts_loss_list.append(ts_loss.item())
            train_df_loss_list.append(df_loss.item())
            train_food_loss_list.append(food_loss.item())

        for i, (x, y) in enumerate(test_loader):
            y_pred = model(x)
            ts_loss = loss_fn(y_pred, y)
            df_loss = df_loss_fn(y_pred, y)
            food_loss = food_loss_fn(y_pred, y)

            test_ts_loss_list.append(ts_loss.item())
            test_df_loss_list.append(df_loss.item())
            test_food_loss_list.append(food_loss.item())

        train_ts_loss = np.mean(train_ts_loss_list)
        train_df_loss = np.mean(train_df_loss_list)
        train_food_loss = np.mean(train_food_loss_list)
        test_ts_loss = np.mean(test_ts_loss_list)
        test_df_loss = np.mean(test_df_loss_list)
        test_food_loss = np.mean(test_food_loss_list)
        print("Epoch {}, Train TS Loss {}, Test TS Loss {}, Train DF Loss {}, Test DF Loss {}, Train Food Loss {}, Test Food Loss {}".format(epoch, train_ts_loss, test_ts_loss, train_df_loss, test_df_loss, train_food_loss, test_food_loss))

        writer.add_scalar('Loss/TS/train', train_ts_loss, epoch)
        writer.add_scalar('Loss/TS/test', test_ts_loss, epoch)
        writer.add_scalar('Loss/DF/train', train_df_loss, epoch)
        writer.add_scalar('Loss/DF/test', test_df_loss, epoch)
        writer.add_scalar('Loss/Food/train', train_food_loss, epoch)
        writer.add_scalar('Loss/Food/test', test_food_loss, epoch)

    writer.flush()

