import numpy as np
import argparse
import torch
import os
import math
from data.synthetic_dataset import create_synthetic_dataset, SyntheticDataset
from models.seq2seq import EncoderRNN, DecoderRNN, Net_GRU
from loss.dilate_loss import dilate_loss
from loss.tre_loss import tre_loss
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
parser.add_argument('--epoch', type=int, default=1000, help='')
parser.add_argument('--batch_size', type=int, default=100, help='')
parser.add_argument('--N', type=int, default=500, help='sample_num')
parser.add_argument('--N_input', type=int, default=48, help='')
parser.add_argument('-N_output', type=int, default=48, help='')
parser.add_argument('--sigma', type=float, default=0.01, help='')
parser.add_argument('--gamma', type=float, default=0.01, help='')
parser.add_argument('--lr', type=float, default=0.001, help='lr')
parser.add_argument('--wd', type=float, default=0.001, help='weight decay')
parser.add_argument('--loss_type', default="tre", help='')
parser.add_argument('--data_type', default="PEM_", help='')
parser.add_argument('--enable-cuda', default=True, help='Enable CUDA')
args = parser.parse_args()
# parameters

X_test_input=np.loadtxt("data/PEM_test_input.txt",delimiter=",") # 
X_test_target=np.loadtxt("data/PEM_test_target.txt",delimiter=",") # 

dataset_test = SyntheticDataset(X_test_input,X_test_target)
testloader = DataLoader(dataset_test, batch_size=args.batch_size,shuffle=True, num_workers=0,drop_last=True)
a=len(dataset_test)
b=len(testloader)

encoder = EncoderRNN(input_size=1, hidden_size=128, num_grulstm_layers=1, batch_size=args.batch_size).to(device)
decoder = DecoderRNN(input_size=1, hidden_size=128, num_grulstm_layers=1, fc_units=16, output_size=1).to(device)
net_gru = Net_GRU(encoder, decoder, args.N_output, device).to(device)

with torch.no_grad():
    name = args.data_type+args.loss_type + "_best.ckpt"
    pname = os.path.join("./checkpoints", name)
    net_gru.load_state_dict(torch.load(pname))
    criterion = torch.nn.MSELoss(reduction='mean')
    losses_mse = []
    losses_mae = []
    losses_smape = []
    losses_dtw = []
    losses_tdi = []
    step=0

    for m, data in enumerate(testloader, 0):
        inputs, target = data
        inputs = torch.tensor(inputs, dtype=torch.float32).to(device)
        target = torch.tensor(target, dtype=torch.float32).to(device)
        target =target #*(x_max-x_min)+x_min
        batch_size, N_output = target.shape[0:2]
        outputs = net_gru(inputs)
        outputs = outputs #* (x_max - x_min) + x_min
        # MAE
        loss_mae = torch.mean(torch.abs(outputs - target))
        loss_mse= criterion(outputs, target)
        loss_smape = torch.mean(2.0 * torch.abs((outputs - target) / (outputs + target)))
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
        losses_mae.append(loss_mae.item())
        losses_smape.append(loss_smape.item())
        losses_dtw.append(loss_dtw)
        losses_tdi.append(loss_tdi)

    losses_mse_mean = np.array(losses_mse).mean()
    losses_mae_mean = np.array(losses_mae).mean()
    losses_smape_mean = np.array(losses_smape).mean()
    losses_dtw_mean = np.array(losses_dtw).mean()
    losses_tdi_mean = np.array(losses_tdi).mean()
    print(' the mean MSE:', losses_mse_mean)
    print(' the mean MAE:', losses_mae_mean)
    print(' the mean SMAPE:', losses_smape_mean)
    print('the mean DTW:', losses_dtw_mean)
    print('the mean TDI:', losses_tdi_mean)







