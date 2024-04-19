import numpy as np
import argparse
import torch
import os
from data.synthetic_dataset import create_synthetic_dataset, SyntheticDataset
from models.seq2seq import EncoderRNN, DecoderRNN, Net_GRU
from loss.dilate_loss import dilate_loss
from loss.tre_loss import tre_loss
from loss.tre_loss2 import tre_loss2
from loss.tre_loss2_ import tre_loss2_
from loss.tre_loss3 import tre_loss3
from torch.utils.data import DataLoader
import random
from tslearn.metrics import dtw, dtw_path
import matplotlib.pyplot as plt
import warnings
import warnings; warnings.simplefilter('ignore')

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
random.seed(0)
parser = argparse.ArgumentParser(description='tre_loss seting')
parser.add_argument('--device', type=str, default='cuda', help='')
parser.add_argument('--run_times', type=int, default=1, help='')#
parser.add_argument('--epoch', type=int, default=500, help='')
parser.add_argument('--batch_size', type=int, default=100, help='')
parser.add_argument('--N', type=int, default=500, help='sample_num')
parser.add_argument('--N_input', type=int, default=48, help='')
parser.add_argument('-N_output', type=int, default=48, help='')
parser.add_argument('--sigma', type=float, default=0.01, help='')
parser.add_argument('--gamma', type=float, default=0.01, help='')
parser.add_argument('--lr', type=float, default=0.001, help='lr')
parser.add_argument('--wd', type=float, default=0.001, help='weight decay')
parser.add_argument('--data_type', default="PEM_", help='')
parser.add_argument('--enable-cuda', default=True, help='Enable CUDA')
args = parser.parse_args()
# parameters

X_train_input=np.loadtxt("data/PEM_train_input.txt",delimiter=",") 
X_train_target=np.loadtxt("data/PEM_train_target.txt",delimiter=",") 
X_val_input=np.loadtxt("data/PEM_val_input.txt",delimiter=",") 
X_val_target=np.loadtxt("data/PEM_val_target.txt",delimiter=",") 

dataset_train = SyntheticDataset(X_train_input,X_train_target)
dataset_val  = SyntheticDataset(X_val_input,X_val_target)
trainloader = DataLoader(dataset_train, batch_size=args.batch_size,shuffle=True, num_workers=0,drop_last=True)
valloader  = DataLoader(dataset_val, batch_size=args.batch_size,shuffle=False, num_workers=0,drop_last=True)
val_loss_min = np.inf

def train_model(net, loss_type=args.loss_type, learning_rate=args.lr, epochs=args.epoch, gamma=0.001,
                print_every=50, eval_every=50, verbose=1, Lambda=1, alpha=3):
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)

    criterion = torch.nn.MSELoss(reduction='mean')

    for epoch in range(epochs):
        for i, data in enumerate(trainloader, 0):
            inputs, target = data
            inputs = torch.tensor(inputs, dtype=torch.float32).to(device)
            target = torch.tensor(target, dtype=torch.float32).to(device)
            batch_size, N_output = target.shape[0:2]

            # forward + backward + optimize
            outputs = net(inputs)
            loss_mse, loss_shape, loss_temporal = torch.tensor(0), torch.tensor(0), torch.tensor(0)
            loss_tre = tre_loss(target, outputs, alpha, device)
            loss = loss_tre

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


        if (verbose):
            if (epoch % print_every == 0):
                name = args.loss_type + "_"+str(epoch)+".ckpt"
                iter_name = os.path.join("./checkpoints", name)
                torch.save(net.state_dict(), iter_name)
                print('epoch ', epoch, ' loss ', loss.item(), ' loss shape ', loss_shape.item(), ' loss temporal ',
                      loss_temporal.item())

                eval_model(net, valloader, epoch, gamma, verbose=1)



def eval_model(net, loader,epoch, gamma, verbose=1):
    criterion = torch.nn.MSELoss(reduction='mean')
    losses_mse = []
    losses_dtw = []
    losses_tdi = []
    val_loss = 0

    for i, data in enumerate(loader, 0):
        loss_mse, loss_dtw, loss_tdi = torch.tensor(0), torch.tensor(0), torch.tensor(0)
        # get the inputs
        inputs, target = data
        inputs = torch.tensor(inputs, dtype=torch.float32).to(device)
        target = torch.tensor(target, dtype=torch.float32).to(device)
        batch_size, N_output = target.shape[0:2]
        outputs = net(inputs)

        # MSE
        loss_mse = criterion(target, outputs)
        loss_dtw, loss_tdi = 0, 0
        # DTW and TDI
        for k in range(batch_size):
            target_k_cpu = target[k, :, 0:1].view(-1).detach().cpu().numpy()
            output_k_cpu = outputs[k, :, 0:1].view(-1).detach().cpu().numpy()

            path, sim = dtw_path(target_k_cpu, output_k_cpu)
            loss_dtw += sim

            Dist = 0
            for i, j in path:
                Dist += (i - j) * (i - j)
            loss_tdi += Dist / (N_output * N_output)

        loss_dtw = loss_dtw / batch_size
        loss_tdi = loss_tdi / batch_size

        # print statistics
        losses_mse.append(loss_mse.item())
        losses_dtw.append(loss_dtw)
        losses_tdi.append(loss_tdi)
    val_loss = np.array(losses_mse).mean()
    global val_loss_min
    if val_loss < val_loss_min and epoch > (args.epoch * 0.01):
        name=args.data_type+args.loss_type+"_best.ckpt"
        pname = os.path.join("./checkpoints",name)
        torch.save(net.state_dict(), pname)
        val_loss_min = val_loss

    print(' Eval mse= ', val_loss, ' dtw= ', np.array(losses_dtw).mean(), ' tdi= ',
          np.array(losses_tdi).mean())

if __name__== "__main__":
    encoder = EncoderRNN(input_size=1, hidden_size=128, num_grulstm_layers=1, batch_size=args.batch_size).to(device)
    decoder = DecoderRNN(input_size=1, hidden_size=128, num_grulstm_layers=1, fc_units=16, output_size=1).to(device)
    net_gru = Net_GRU(encoder, decoder, args.N_output, device).to(device)
    train_model(net_gru, loss_type=args.loss_type, learning_rate=args.lr, epochs=args.epoch, gamma=args.gamma, print_every=50,
                eval_every=50, verbose=1)
    print("training is finishing!")

