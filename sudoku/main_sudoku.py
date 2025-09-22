import numpy as np
import torch


import time
import os
import argparse

from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, TensorDataset, Subset

from models_sudoku import BLOSudoku, OptNetSudoku, SingleOptLayerSudoku
from constants import FFOCP_EQ, QPTH, LPGD, CVXPY_LAYER

        
        
    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', type=str, default='ffocp_eq', help='ffocp_eq, lpgd, qpth, cvxpylayer')
    parser.add_argument('--epochs', type=int, default=20, help='number of epochs')
    parser.add_argument('--seed', type=int, default=1, help='random seed')
    parser.add_argument('--lr', type=float, default=0.00001, help='learning rate')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size')
    
    args = parser.parse_args()
    


    method = args.method
    seed = args.seed
    num_epochs = args.epochs
    learning_rate = args.lr
    batch_size = args.batch_size
    
    n = 2
    board_side_len = n**2
    Qpenalty = 0.1
    
    device = torch.device('cpu') #torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True

    #################### PREPARE DATA ##################
    
    train_data_dir_path = f"sudoku/data/{n}"
    features = torch.load(os.path.join(train_data_dir_path, "features.pt"))
    labels = torch.load(os.path.join(train_data_dir_path, "labels.pt"))
    features = torch.tensor(features, dtype=torch.float32).to(device)[:50000]
    labels   = torch.tensor(labels, dtype=torch.float32).to(device)[:50000]
    print(features.shape)
    print(labels.shape)
    #assert(1==0)
    
    # Create TensorDataset
    dataset = TensorDataset(features, labels)   
    
   # Fixed split indices
    num_samples = len(dataset)
    train_split = 0.8
    train_size = int(num_samples * train_split)
    test_size = num_samples - train_size

    # Use first 80% for training, last 20% for testing
    train_indices = list(range(0, train_size))
    test_indices = list(range(train_size, num_samples))

    train_dataset = Subset(dataset, train_indices)
    test_dataset  = Subset(dataset, test_indices)

    # DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    print(f"Training samples: {len(train_dataset)}, Test samples: {len(test_dataset)}")
    
    ###############################################
    model = SingleOptLayerSudoku(n, learnable_parts=["eq"], layer_type=method, Qpenalty=0.1, alpha=1000)
    
    # if method==FFOCP_EQ:
    #     model = BLOSudoku(n, Qpenalty, alpha=100).to(device)
    # elif method==QPTH:
    #     model = OptNetSudoku(n, Qpenalty).to(device)
    # else:
    #     assert(1==0)
        
    
    
    directory = 'sudoku_results_{}/{}/'.format(args.batch_size, method)
    filename = '{}_n{}_lr{}_seed{}.csv'.format(method, n, learning_rate, seed)
    if os.path.exists(directory + filename):
        os.remove(directory + filename)

    if not os.path.exists(directory):
        os.makedirs(directory)

    file = open(directory + filename, 'w')
    file.write('epoch, train_loss, test_loss, forward_time, backward_time\n')
    
    writer = SummaryWriter(log_dir=f"runs/sudoku_n{n}_{method}_bs{batch_size}_lr{learning_rate}_seed{seed}")
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0)
    loss_fn = torch.nn.MSELoss()
    
    avg_train_loss = []
    avg_test_loss = []
    for epoch in range(num_epochs):
        print(f"##### epoch {epoch}: ")
        train_loss_list, test_loss_list = [], []
        forward_time = 0
        backward_time = 0

        model.train()
        for i, (x, y) in enumerate(train_loader):
            if i%1==0:
                print(f"\t\t train example: {i}/{len(train_loader)}")
            x = x.to(device)
            y = y.to(device)
            
            start_time = time.time()
            pred = model(x)
            loss = loss_fn(pred, y)
            
            forward_time += time.time() - start_time

            start_time = time.time()
            loss.backward()
            backward_time += time.time() - start_time

            #torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            if epoch > 0:
                optimizer.step()

            optimizer.zero_grad()

            train_loss_list.append(loss.item())

        if epoch%10==0:
                torch.save(model.state_dict(), os.path.join(directory, f"model_epoch{epoch}.pt"))
        print('Forward time {}, backward time {}'.format(forward_time, backward_time))

        model.eval()
        with torch.no_grad():
            for i, (x, y) in enumerate(test_loader):
                x = x.to(device)
                y = y.to(device)
                
                pred = model(x)
                loss = loss_fn(pred, y)

                test_loss_list.append(loss.item())

        train_loss = np.mean(train_loss_list)
        test_loss = np.mean(test_loss_list)
        avg_train_loss.append(train_loss)
        avg_test_loss.append(test_loss)
        print("Epoch {}, Train Loss {}, Test Loss {}".format(epoch, train_loss, test_loss))

        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/test', test_loss, epoch)

        file.write('{},{},{},{},{}\n'.format(epoch, train_loss, test_loss, forward_time, backward_time))

    writer.flush()
    file.close()
    
    # import matplotlib.pyplot as plt

    # # After training loop
    # epochs = range(1, num_epochs + 1)

    # plt.figure(figsize=(8,5))
    # plt.plot(epochs, avg_train_loss, label='Train Loss', marker='o')
    # plt.plot(epochs, avg_test_loss, label='Test Loss', marker='s')
    # plt.xlabel('Epoch')
    # plt.ylabel('MSE Loss')
    # plt.title('Training and Test Loss')
    # plt.legend()
    # plt.grid(True)
    # plt.tight_layout()
    # plt.show()