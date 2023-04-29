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
TEST_INSTANCES = 10000
TRAINING_INSTANCES = 60000
LAYERS = 3

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
        self.p = power
        self.s = torch.tensor(scale, device=device)
        # Weight parameters
        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features, device=device,dtype=torch.double).uniform_(-0.1, 0.1))
        self.weight_phi = nn.Parameter((torch.full((out_features, in_features), torch.log(torch.exp((torch.tensor(1e-4, dtype=torch.double)))-1), device=device).double()))
        
        # Bias parameters
        self.bias_mu = nn.Parameter(torch.empty(out_features, device=device, dtype=torch.double).uniform_(-0.1, 0.1)) 
        self.bias_phi = nn.Parameter(torch.full((1, out_features), torch.log(torch.exp((torch.tensor(1e-4, dtype=torch.double)))-1), device=device).double()) #was 0.01 before january
        
    
    def forward(self, input, sample=False, uniform=False):
        weight_sig = F.softplus(self.weight_phi)
        bias_sig = F.softplus(self.bias_phi)
        
        if self.training:
            net.reg_loss += net.reg_cost_func(self.weight_mu).sum()
            net.reg_loss += net.reg_cost_func(self.bias_mu).sum()
            
            if sample:
                net.rel_loss += net.rel_cost_func(weight_sig)
                net.rel_loss += net.rel_cost_func(bias_sig)
                
            
            if uniform==True:
                    weight_s = 0
                    bias_s = 0
                    l = 0
                    for mod in net.modules():
                        if isinstance(mod, NetLayer):
                            weight_s += ((F.softplus(mod.weight_phi))**2).sum()
                            weight_s += ((F.softplus(mod.bias_phi))**2).sum()
                            l += 1
                    weight_s = torch.sqrt(weight_s/(28*28*100+100+100*100+100+100*10+10))
                    #bias_s = torch.sqrt(bias_s/l)
                    net.rel_loss += (len(self.weight_mu.flatten()))*net.rel_cost_func(weight_s)
                    net.rel_loss += (len(self.bias_mu.flatten()))*net.rel_cost_func(weight_s)
                    weight_dist = torch.distributions.Normal(self.weight_mu, weight_s)
                    bias_dist = torch.distributions.Normal(self.bias_mu, weight_s)
                else:   
                    weight_dist = torch.distributions.Normal(self.weight_mu, weight_sig)
                    bias_dist = torch.distributions.Normal(self.bias_mu, bias_sig)
        
        weight = weight_dist.rsample()
        bias = bias_dist.rsample()
                
        return F.linear(input, weight, bias)
                   
class Net(nn.Module):
    def __init__(self, power=2, scale=1): 
        super().__init__()
        self.p = power
        self.s = torch.tensor(scale, device=device)
        self.firingrate = []
        self.linear1 = NetLayer(28*28, 10)
        self.linear2 = NetLayer(100, 100)
        self.linear3 = NetLayer(100, 10)
    
    def rel_cost_func(self, sig):
        return (BATCHSIZE/TRAINING_INSTANCES)*torch.sum((1/self.p) * (self.s/sig)**self.p)
    
    def reg_cost_func(self, mu):
        return (BATCHSIZE/TRAINING_INSTANCES)*0.01*torch.sum(torch.abs(mu))
        
    def forward(self, x, sample=False, biosample=False, lang=False, noise=False, s=False, batch_idx=False, epoch=False):
        self.rel_loss = 0
        self.reg_loss = 0
        
        x = x.view(-1, 784)
        x = self.linear1(x, sample, uniform)
        x = F.relu(self.linear2(x, sample, uniform))
        x = F.relu(self.linear3(x, sample, uniform))
        x = F.log_softmax(x, dim=1)
        return x
    
    @staticmethod
    def loss(pred_values, true_values):
        criterion = nn.NLLLoss(reduction='mean')
        loss = criterion(pred_values, true_values)*BATCHSIZE
        return loss


def train(sample=False, uniform=False):
    loss_list = []
    accs_list = []
    hessian = []
    lr = []
 
    for epoch in range(1100):
        if epoch in np.arange(1000, 1100, 1):
            sample = False
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

                preds = net(data, sample=sample, uniform=uniform)
                loglike += net.loss(preds, target)
                
                loss = loglike + net.rel_loss + net.reg_loss
             
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
                for n in range(20):
                    images, labels = data
                    images = images.to(device)
                    images = images.type(torch.double) 
                    labels = labels.to(device)
                    labels = labels.type(torch.long)
                    preds += net(images, sample=sample, uniform=uniform)
                    
                preds /= 20
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
                preds = net(images, sample=sample, uniform=uniform)
                loss += net.loss(preds, target)
                class_preds = preds.max(1, keepdim=True)[1]
                correct += class_preds.eq(labels.view(-1, 1)).sum().item()
            
        loss = loss / TEST_INSTANCES
        accuracy = 100 * correct / TEST_INSTANCES
        print('Sample Accuracy', np.round(accuracy,3))
        print('Sample Loss', np.round(loss.item(),3))
    return accuracy, loss

mode_performance_list = []
mode_loss_list = []
sample_performance_list = []
sample_loss_list = []
sampled_test_list = []
expected_performance_list = []
expected_loss_list = []

rel_list = []
hessian_list = []
sig_list = []
lr_list = []
firingrate_list = []

coeffs = 10**(np.arange(-2, 4.25, 0.25, dtype=float))
powers = [0.5, 2/3, 4/3, 2]
   
    
for p in powers:
    sig = []
    rel = []
    sampled_accs_arr = []
    sampled_loss_arr = []
    for c in coeffs:
        s = (c*p)**(1/p)
        net = Net(power=p, scale=s)
        optimiser = optim.Adam(net.parameters(), lr=0.0001)
        for epoch in range(1):
            mode_performance, mode_loss, sample_performance, sample_loss, expected_performance, expected_loss = train(sample=True, uniform=False)
            sampled_accs_arr.append(sample_performance)
            sampled_loss_arr.append(sample_loss)
        
        sig_list = []
        rel_list = []
        for mod in net.modules():
            if isinstance(mod, NetLayer):
                #var_list.append(torch.pow(F.softplus(mod.weight_phi.detach().clone()),2))
                sig_list.append(((F.softplus(mod.weight_phi.detach().clone()))**2).sum().item())
                relcost = lambda sig, p: (1/p)*sig**(-p) 
                rel_list.append(relcost(F.softplus(mod.weight_phi.detach().clone()),p).sum().item())
        sig.append(np.sqrt((np.sum(np.array(sig_list)))/(28*28*100+100*100+100*10)))
        rel.append(np.sum(np.array(rel_list)))
        
        net = Net(power=p, scale=s)
        optimiser = optim.Adam(net.parameters(), lr=0.0001)
        for epoch in range(1):   
            mode_performance, mode_loss, sample_performance, sample_loss, expected_performance, expected_loss = train(sample=True, uniform=True)
            sampled_accs_arr.append(sample_performance)
            sampled_loss_arr.append(sample_loss)
           
        
        sig_list = []
        for mod in net.modules():
            if isinstance(mod, NetLayer):
                sig_list.append(((F.softplus(mod.weight_phi.detach().clone()))**2).sum().item()) 
        sig.append(np.sqrt((np.sum(np.array(sig_list)))/(28*28*100+100*100+100*10)))
        relcost = lambda sig, p: (1/p)*sig**(-p)
        rel.append((28*28*100+100*100+100*10)*relcost(sig[-1], p).item())
        
    sig_arr.append(sig)
    rel_arr.append(rel)
    accs_arr.append(sampled_accs_arr)
    loss_arr.append(sampled_loss_arr)
  
    



