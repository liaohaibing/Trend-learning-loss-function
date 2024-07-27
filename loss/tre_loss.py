import torch
import numpy as np
import numpy.fft as nf
from scipy.spatial.distance import cdist

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def tre_loss(targets,outputs,alpha,device):
	sq_error = w_mse(targets,outputs,device)
	error1=torch.mean(sq_error)
	x1 = derivatives(targets, device)
	x2 = derivatives(outputs, device)
	# 求皮尔逊相关系数
	corr = np.corrcoef(x1.cpu().detach().numpy().squeeze(), x2.cpu().detach().numpy().squeeze())
	batch_size, lens = x1.shape[0:2]
	p_corr = corr[0:batch_size, batch_size::]
	p_corr=np.nan_to_num(p_corr)
	w_corr = 1 - np.diag(p_corr, 0)
	distance = cdist(targets.cpu().detach().numpy().squeeze(), outputs.cpu().detach().numpy().squeeze(), 'euclidean')
	dd= np.diag(distance,0)
	nd=dd/lens
	d_error=w_corr*nd
	#d_error = torch.abs(derivatives(targets,device) - derivatives(outputs,device))
	d_error=np.mean(d_error)
	error2=torch.tensor(d_error,dtype=torch.float32,requires_grad=True).to(device)
	add_error = error1+alpha*error2
	loss= add_error
	return loss


def w_mse(targets,outputs,device):
	delta=0.00000001
	batch_size, len = targets.shape[0:2]
	target1= targets[:,1:len,:].to(device)
	target2= targets[:,0:len-1,:].to(device)
	output1 = outputs[:, 1:len, :].to(device)
	output2 = outputs[:, 0:len - 1, :].to(device)
	sigma_matrix=(target1-target2)*(output1-output2).to(device)
	sigma=torch.sign(sigma_matrix).to(device)
	newtargets=targets[:,1:len,:].to(device)
	newoutputs=outputs[:,1:len,:].to(device)
	W=(1.01+torch.abs(newoutputs-newtargets)/(torch.abs(newoutputs+newtargets)+0.00001))**(1-sigma).to(device)
	w_error=W*torch.abs(newoutputs-newtargets)
	return w_error
def derivatives(input,device):
	batch_size,len =input.shape[0:2]
	input2 = input[:,2:len,:].to(device)
	input1= input[:,0:len-2,:].to(device)
	D= input2-input1
	return D

if __name__ == "__main__":
	targets = np.array([[2, 3, 1, 4, 2, 5],[1,2,3,4,5,6]])
	targets=torch.tensor(targets)
	target = targets[:,:,np.newaxis]
	outputs = np.array([[3, 4, 3, 2, 1, 6],[3,3,2,4,5,6]])
	outputs=torch.tensor(outputs)
	output=outputs[:,:,np.newaxis]
	tre_loss2(output,target,1,device)
	print("finishing")

