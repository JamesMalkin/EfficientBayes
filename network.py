import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import math
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt



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
        self.bias_mu = nn.Parameter(torch.empty(out_features, device=device, dtype=torch.double).uniform_(-0.1, 0.1)) #(-0.2, 0.2) 
        self.bias_phi = nn.Parameter(torch.full((1, out_features), torch.log(torch.exp((torch.tensor(1e-4, dtype=torch.double)))-1), device=device).double()) #was 0.01 before january
        
    
    def forward(self, input, sample=False):
        weight_sig = F.softplus(self.weight_phi)
        bias_sig = F.softplus(self.bias_phi)
        weight_var  = torch.pow(weight_sig.detach().clone(),2)
        bias_var  = torch.pow(bias_sig.detach().clone(),2)
        
        if self.training:
            net.reg_loss += net.reg_cost_func(self.weight_mu).sum()
            net.reg_loss += net.reg_cost_func(self.bias_mu).sum()
            #net.prior_loss += net.prior_cost_func(self.weight_mu).sum()
            #net.prior_loss += net.prior_cost_func(self.bias_mu).sum()
            
            if sample:
                net.rel_loss += net.rel_cost_func(weight_sig)
                net.rel_loss += net.rel_cost_func(bias_sig)
                #net.ent_loss += net.ent_cost_func(self.weight_mu).sum()
                #net.ent_loss += net.ent_cost_func(self.bias_mu).sum()
                  
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
        self.linear1 = NetLayer(28*28, 10, self.p, self.s)
        #self.linear2 = NetLayer(100, 100, self.p, self.s)
        #self.linear3 = NetLayer(100, 10, self.p, self.s)
    
    def rel_cost_func(self, sig):
        return (BATCHSIZE/TRAINING_INSTANCES)*torch.sum((1/self.p) * (self.s/sig)**self.p)
    
    def reg_cost_func(self, mu):
        return (BATCHSIZE/TRAINING_INSTANCES)*0.1*torch.sum(torch.abs(mu))
    
    def prior_cost_func(self, mu):
        return (BATCHSIZE/TRAINING_INSTANCES)*1*torch.sum(mu**2)

        
    def forward(self, x, sample=False, biosample=False, lang=False, noise=False, s=False, batch_idx=False, epoch=False):
        self.rel_loss = 0
        self.reg_loss = 0
        
        x = x.view(-1, 784)
        x = self.linear1(x, sample)
        x = F.relu(self.linear2(x, sample, biosample, lang, noise))
        x = F.relu(self.linear3(x, sample))
        x = F.log_softmax(x, dim=1)
        return x
    
    @staticmethod
    def loss(pred_values, true_values):
        criterion = nn.NLLLoss(reduction='mean')
        loss = criterion(pred_values, true_values)*BATCHSIZE
        return loss

SAMPLES = 1
def train(sample=False, biosample=False, s=False, lang=False):
    loss_list = []
    accs_list = []
    hessian = []
    lr = []
    running_loss = 0
    running_rel_loss = 0
    running_reg_loss = 0
    running_mode_loss = 0
    for epoch in range(1000):
        for batch_idx, (data, target) in enumerate(trainloader):
            net.train()
            data = data.to(device)
            data = data.type(torch.double)
            target = target.to(device)
            target = target.type(torch.long)
            net.zero_grad()
            loglike = 0
            rel_loss = 0
            reg_loss = 0

            for j in range(SAMPLES):
                preds = net(data, sample=sample)
                loglike += net.loss(preds, target)
                rel_loss += net.rel_loss
                reg_loss += net.reg_loss

            loss = loglike + rel_loss + reg_loss
            loss.backward()
            optimiser.step()

    mode_performance, mode_loss = (0,0)
    sample_performance, sample_loss = (0,0)
    average_performance, average_loss = (0,0)
    net.eval()
    mode_performance, mode_loss = test(sample=False, exp_accuracy=False)
    sample_performance, sample_loss = test(sample=True, exp_accuracy=False)
    expected_performance, expected_loss = test(sample=True, exp_accuracy=True)
    
    return mode_performance, mode_loss, sample_performance, sample_loss, expected_performance, expected_loss

        

def test(sample=False, classes=10, exp_accuracy=False):
    correct = 0
    loss = 0
    net.eval()
    if exp_accuracy:
        with torch.no_grad():
            for data in testloader:
                preds = torch.zeros(BATCHSIZE, 10)
                class_preds = torch.zeros(BATCHSIZE, 1)
                for n in range(100):
                    images, labels = data
                    images = images.to(device)
                    images = images.type(torch.double) 
                    labels = labels.to(device)
                    labels = labels.type(torch.long)
                    preds += net(images, sample=sample, biosample=biosample, lang=lang, noise=noise)
                    
                preds /= 100
                loss += net.loss(preds, labels)
                class_preds = preds.max(1, keepdim=True)[1]
                correct += class_preds.eq(labels.view(-1, 1)).sum().item()
        accuracy = 100 * correct / (TEST_INSTANCES)
        loss /= TEST_INSTANCES
        print('Expected Accuracy', np.round(accuracy,3))
        print('Expected Loss', np.round(loss.item(),3))
    else:
        with torch.no_grad():
            testloss = [0,0]
            for data in testloader:
                data = data
                images, labels = data
                images = images.to(device)
                images = images 
                labels = labels.to(device)
                images = images.type(torch.double) 
                labels = labels.type(torch.long)
                preds = net(images, sample=sample, biosample=biosample, lang=lang, noise=noise)
                loss += net.loss(preds, target)
                class_preds = preds.max(1, keepdim=True)[1]
                correct += class_preds.eq(labels.view(-1, 1)).sum().item()
            
        loss = loss / TEST_INSTANCES
        accuracy = 100 * correct / TEST_INSTANCES
        print('Sample Accuracy', np.round(accuracy,3))
        print('Sample Loss', np.round(loss.item(),3))
    return accuracy, loss
