# Caution: This simulation runs for 1 day on GPU

import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np


if torch.cuda.is_available():  
    dev = "cuda" 
else:  
    dev = "cpu"
device = torch.device(dev)

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
trainset = torchvision.datasets.MNIST(root = '.data/trainset', train = True, transform=transform, download=True)
testset = torchvision.datasets.MNIST(root = '.data/testset', train = False, transform=transform, download=True)


torch.manual_seed(0)
    
BATCHSIZE = 20
EPOCHS = 1
TEST_INSTANCES = 10000
TRAINING_INSTANCES = 60000
LAYERS = 1

#trainset = torch.load('.data/trainset')
#trainset, valset = torch.utils.data.random_split(trainset, [50000, 10000])

trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCHSIZE,
                                          shuffle=True, num_workers=0)

#testset = torch.load('.data/testset')
testloader = torch.utils.data.DataLoader(testset, batch_size=BATCHSIZE,
                                         shuffle=False, num_workers=0)


class NetLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        
        # Weight parameters
        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features, device=device,dtype=torch.double).uniform_(-0.1, 0.1))
        self.weight_phi = nn.Parameter((torch.full((out_features, in_features), torch.log(torch.exp((torch.tensor(1e-4, dtype=torch.double)))-1), device=device).double()))
        
        # Bias parameters
        self.bias_mu = nn.Parameter(torch.empty(out_features, device=device, dtype=torch.double).uniform_(-0.1, 0.1)) 
        self.bias_phi = nn.Parameter(torch.full((1, out_features), torch.log(torch.exp((torch.tensor(1e-4, dtype=torch.double)))-1), device=device).double()) 
        
    
    def forward(self, input, sample=False):
        weight_sig = F.softplus(self.weight_phi)
        bias_sig = F.softplus(self.bias_phi)
        
        if self.training:
            net.reg_loss += net.reg_cost_func(self.weight_mu).sum()
            net.reg_loss += net.reg_cost_func(self.bias_mu).sum()
            
            if sample:
                net.rel_loss += net.rel_cost_func(weight_sig)
                net.rel_loss += net.rel_cost_func(bias_sig)
                  
        weight_dist = torch.distributions.Normal(self.weight_mu, weight_sig)
        bias_dist = torch.distributions.Normal(self.bias_mu, bias_sig)
        
        weight = weight_dist.rsample()
        bias = bias_dist.rsample()
                
        return F.linear(input, weight, bias)
                   
class Net(nn.Module):
    def __init__(self, power=2, scale=1): 
        super().__init__()
        self.firingrate = []
        self.p = power
        self.s = torch.tensor(scale, device=device)
        self.linear1 = NetLayer(28*28, 100)
        self.linear2 = NetLayer(100, 100)
        self.linear3 = NetLayer(100, 10)
    
    def rel_cost_func(self, sig):
        return (BATCHSIZE/TRAINING_INSTANCES)*torch.sum((1/self.p) * (self.s/sig)**self.p)
    
    def reg_cost_func(self, mu):
        return (BATCHSIZE/TRAINING_INSTANCES)*0.01*torch.sum(torch.abs(mu))
        
    def forward(self, x, sample=False, batch_idx=False, epoch=False):
        self.rel_loss = 0
        self.reg_loss = 0
        
        x = x.view(-1, 784)
        x = self.linear1(x, sample)
        if epoch in np.arange(50, 60, 1):
            if len(self.firingrate) == 0:
                self.firingrate.append(x.detach().clone())
            else:
                self.firingrate[0] += x.detach().clone()
        x = F.relu(self.linear2(x, sample))
        if epoch in np.arange(50, 60, 1):
            if len(self.firingrate) == 1:
                self.firingrate.append(x.detach().clone())
            else:
                self.firingrate[1] += x.detach().clone()
        x = F.relu(self.linear3(x, sample))
        x = F.log_softmax(x, dim=1)
        return x
    
    @staticmethod
    def loss(pred_values, true_values):
        criterion = nn.NLLLoss(reduction='mean')
        loss = criterion(pred_values, true_values)*BATCHSIZE
        return loss


def train(sample=False):
    loss_list = []
    accs_list = []
    hessian = []
    lr = []
  
    for epoch in range(60):
        for batch_idx, (data, target) in enumerate(trainloader):
            if batch_idx <= len(trainloader):
                net.train()
                data = data.to(device)
                data = data.type(torch.double)
                target = target.to(device)
                target = target.type(torch.long)
                net.zero_grad()
                loglike = 0
                rel_loss = 0
                reg_loss = 0

              
                preds = net(data, sample=sample)
                loglike += net.loss(preds, target)
                
                loss = loglike + net.rel_loss + net.reg_loss
                loss.backward()
                if epoch in np.arange(50, 60, 1):
                    for name, p in net.named_parameters():
                        if name in ['linear1.weight_mu', 'linear2.weight_mu', 'linear3.weight_mu']:
                            g = p.grad.data
                            if len(hessian) <= LAYERS:
                                hessian.append((g.detach().clone()**2))
                                lr.append(torch.abs(g.detach().clone()))
                            elif name=='linear1.weight_mu':
                                hessian[0] += (g.detach().clone()**2)
                                lr[0] += torch.abs(g.detach().clone())
                            elif name=='linear2.weight_mu':
                                hessian[1] += (g.detach().clone()**2)
                                lr[1] += torch.abs(g.detach().clone())
                            elif name=='linear3.weight_mu':
                                hessian[1] += (g.detach().clone()**2)
                                lr[1] += torch.abs(g.detach().clone())
                            
                if epoch not in np.arange(50, 60, 1):
                    optimiser.step()
                    
    return hessian, lr


hessian_list = []
var_list = []
lr_list = []
firingrate_list = []

powers = [0.5, 2/3, 4/3, 2]
   
for p in powers:
    s = 0.001
    net = Net(power=p, scale=s)
    optimiser = optim.Adam(net.parameters(), lr=0.0001)
    hessian, lr = train(sample=True)
    
    for mod in net.modules():
        if isinstance(mod, NetLayer):
            var_list.append(torch.pow(F.softplus(mod.weight_phi.detach().clone()),2))
      
  
    hess = hessian[0]/(60000*10)
    hess2 = hessian[1]/(60000*10)
    hess3 = hessian[2]/(60000*10)
 
    hessian_list.append(hess)
    hessian_list.append(hess2)
    hessian_list.append(hess3)

    learn_rate = lr[0]/(60000*10)
    learn_rate2 = lr[1]/(60000*10)
    learn_rate3 = lr[2]/(60000*10)
  
    lr_list.append(learn_rate)
    lr_list.append(learn_rate2)
    lr_list.append(learn_rate3)
    

    firingrate_list.append(net.firingrate[0]/(60000*10))
    firingrate_list.append(net.firingrate[1]/(60000*10))
   
for p in range(len(powers)):
    for i in range(3):
        np.save('hessian_list_{}'.format(3*p+i), (hessian_list[3*p+i]).cpu().numpy())
        np.save('var_list_{}'.format(3*p+i), (var_list[3*p+i]).cpu().numpy())
        np.save('lr_list_{}'.format(3*p+i), (lr_list[3*p+i]).cpu().numpy())

for p in range(len(powers)):      
    for i in range(2):
        np.save('x_{}'.format(3*p+i+1), (firingrate_list[2*p+i]).cpu().numpy())
       
    
x = torch.zeros((784)).to(device).double()
for batch_idx, (data_target) in enumerate(trainloader):
    data = data_target.to(device).double()
    data = data_target[:,:-1].to(device)
    x += torch.abs(data).sum(dim=[0])
    
    
for p in range(len(powers)):
    np.save('x_{}'.format(3*p), ((x)/60000).cpu().numpy())

        
    
