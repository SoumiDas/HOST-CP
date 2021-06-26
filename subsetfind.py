from __future__ import print_function
from torchvision.utils import save_image
import argparse
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torchvision import models
from torch.optim.lr_scheduler import StepLR
import torchvision.models as models
from torch.autograd import Variable
import numpy as np
import pickle
import os
import time
import faiss
import math
import copy
import cvxpy as cp
import torch.backends.cudnn as cudnn
from cvxpylayers.torch import CvxpyLayer
from arch import resnet


device = 'cuda' if torch.cuda.is_available() else 'cpu'
no_cuda = False
use_cuda = not no_cuda and torch.cuda.is_available()
batch_size = 40

torch.manual_seed(123)

kwargs = {'batch_size': 40}
if use_cuda:
  kwargs.update({'num_workers': 1,
                       'pin_memory': True,
                       'shuffle': False},
                     )
kwargstest = {'batch_size': 1000}
if use_cuda:
  kwargstest.update({'num_workers':1,
                       'pin_memory': True,
                       'shuffle': False},
                     )

train_transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
test_transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
dataset1 = datasets.CIFAR10('./', train=True, download=True,
                       transform=train_transform)
dataset2 = datasets.CIFAR10('./', train=False,
                       transform=test_transform)

#Fill in the test set indices
subset_indices = []

subset = torch.utils.data.Subset(dataset2, subset_indices)

train_loader = torch.utils.data.DataLoader(dataset1,**kwargs)

test_loader = torch.utils.data.DataLoader(subset, **kwargstest)
#test_wloader = torch.utils.data.DataLoader(dataset2, **kwargstest)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


D_out = 100 #feature dimension
D_in = 40 #Number of incoming instances
D_exin = 40 #Maximum number of existing instances
fraction = 0.20 #20%
m = 10   # number of subquantizers for FAISS
n_bits = 2 # bits for FAISS
d = 100 #feature dimension
k = 40 #Number of closest neighbours for each instance from FAISS

################### CVXPYLAYERS #######################
def subsetLayer(D_in, D_exin):
  
  finc = cp.Parameter((D_in, D_in))
  fex = cp.Parameter((D_in, D_exin))
 
  Zn = cp.Variable((D_in, D_in))
  Zo = cp.Variable((D_in, D_exin))
  
  obj = cp.sum(cp.multiply(fex, Zo))+cp.sum(cp.multiply(finc, Zn))
  
  M = cp.sum(Zn, axis=1) + cp.sum(Zo,axis=1)
  N_p = cp.norm(Zn, p=1, axis=0)  #sum over i, estimate of j's representativeness
  est_size = cp.norm(N_p, p=1)
  constraints = [0 <= Zo, Zo <= 1, 0 <= Zn, Zn <= 1, M == 1, est_size <= fraction*D_in]
  #constraints = [0 <= Zo, Zo <= 1, 0 <= Zn, Zn <= 1, M == 1]
 
  objective = cp.Minimize(obj)
  prob = cp.Problem(objective, constraints)
  print("Solving")
  layer = CvxpyLayer(prob, parameters=[finc, fex], variables=[Zn])

  return layer

############## Choose 40 neighbours for the current incoming set #################
def close_ind(I):
  ind = set()
  for i in range(I.shape[1]): #{For D_in x k, shape[1] is k}
    for j in range(I.shape[0]):
      ind.add(I[j,i])
      if len(ind)==40:
        break
    if len(ind)==40:
      break

  return ind

################## Extract the chosen neighbour indexed features from the existing set######
def extract_ind_tch(fxtch, chosenind):
  fex = []
  chosenind = sorted(chosenind)

  for i in range(len(chosenind)): #{number of chosen instances x 100}
    if i==0:
      fexv_tch = fxtch[chosenind[i]].unsqueeze(0)
    else:
      fexv_tch = torch.cat((fexv_tch,fxtch[chosenind[i]].unsqueeze(0)),0)
    
  return fexv_tch

############### Difference between the features of a square matrix########################

def find_feadiff(res1, res2):
  diffres = torch.zeros(res1.shape[0], res2.shape[0]).cuda()
  for i in range(res1.shape[0]):
    for j in range(i+1):
        diffres[i,j] = torch.dist(res1[i,:], res2[j,:],2)
        diffres[j,i] = diffres[i,j]
  return diffres

 ############### Difference between the features of a non-square matrix########################

def find_feadiff_nsq(res1, res2):
  diffres = torch.zeros(res1.shape[0], res2.shape[0]).cuda()
  for i in range(res1.shape[0]):
    for j in range(res2.shape[0]):
        diffres[i,j] = torch.dist(res1[i,:], res2[j,:],2)
  return diffres

############### Extract feature differences after extracting features ########################
def extract_features_diff(miniloader,resextract, extractor, red_dim):

  reslist = []

  batch,c,h,w = miniloader.size()
  mini = resextract(miniloader)
  mini = F.avg_pool2d(mini, 4)
  res = extractor(mini.squeeze(2).squeeze(2))
  resln = red_dim(res)
  feadiff = find_feadiff(resln,resln)

  return torch.Tensor.cpu(resln).detach().numpy(),feadiff, resln, mini
  #reduced feature dimension numpy, feature difference, reduced feature dimension torch

############################# Parameter filled model ##########################
def lincalc(inp,wt,bias):
   output = inp.matmul(wt.t())
   fop = output + bias
   ret = fop
   return ret

class lin(nn.Module):
  def __init__(self, wt, bs):
        super(lin, self).__init__()
        self.weight = wt
        self.bias = bs
  def forward(self, input):
        return lincalc(input, self.weight, self.bias)

class Net_test(nn.Module):
    def __init__(self,paramsel):
        super(Net_test, self).__init__()
        self.lin1 = lin(paramsel['linear1.weight'].clone(),paramsel['linear1.bias'].clone())
        self.lin2 = lin(paramsel['linear2.weight'].clone(),paramsel['linear2.bias'].clone())
        self.lin3 = lin(paramsel['linear3.weight'].clone(),paramsel['linear3.bias'].clone())


    def forward(self, x):
        out1 = self.lin1(x)
        out2 = F.relu(out1)
        out3 = self.lin2(out2)
        out4 = F.relu(out3)
        out5 = self.lin3(out4)

        return out5

##################### 3 layer network ####################
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.linear1 = nn.Linear(512, 384)
        self.linear2 = nn.Linear(384, 128)
        self.linear3 = nn.Linear(128, 10)
      
    def forward(self, x):
        out = self.linear1(x)
        out = F.relu(out)
        out = self.linear2(out)
        out = F.relu(out)
        out = self.linear3(out)
        
        return out

classf = Net()
classf = classf.cuda()

# Model
print('==> Building model..')
resf = resnet.ResNet18()
resf = resf.to(device)
########### Load a trained model or skip this step if no pretrained features are needed ############
checkpoint = torch.load('./resnet_checkp/ckpt.pth')
resf.load_state_dict(checkpoint['net'])
print("Model loaded")

labeldict = {0:'airplane',1:'automobile',2:'bird',3:'cat',4:'deer',5:'dog',6:'frog',7:'horse',8:'ship',9:'truck'}


def imshow(img):
    img = img / 2 + 0.5
    npimg = img.detach().cpu().numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

def subset_show(sind,epoch,batch,images, labels):

  lab = []
  labv = []
  labc={labeldict[0]:0,labeldict[1]:0,labeldict[2]:0,labeldict[3]:0,labeldict[4]:0,labeldict[5]:0,labeldict[6]:0,labeldict[7]:0,labeldict[8]:0,labeldict[9]:0}
  for ind in range(len(labels)):
    labv.append(labeldict[labels[ind].item()])
    labc[labeldict[labels[ind].item()]]+=1

  for ind in range(len(sind)):
    if ind==0:
      im = images[sind[ind]]
    else:
      im = torch.cat((im,images[sind[ind]]),0)
    lab.append(labeldict[labels[sind[ind]].item()])
    
  print("Epoch"+str(epoch)+",Batch"+str(batch))
  '''for si in range(im.shape[0]):
    save_image(im[si],'./epoch_images_20'+str(epoch)+'/im_'+str(batch)+'_'+str(si)+'.png')'''
  
  ########Use the selected indices to train on the subset from scratch after subset selection############
  
  with open('./selected_indices_20.txt','a') as fpsi:
    fpsi.write('\n')
    fpsi.write("Epoch "+str(epoch)+", Batch "+str(batch)+'\n')
    fpsi.write(','.join(str(si) for si in sind))


feature_map = list(classf.children())
feature_map.pop()

extractor = nn.Sequential(*feature_map)

res_map = list(resf.children())
res_map.pop()

resextract = nn.Sequential(*res_map)

red_dim = nn.Sequential(
          nn.Linear(128,100),
          nn.ReLU(),
          nn.BatchNorm1d(100)).to('cuda')

#optimizer = torch.optim.SGD(classf.parameters(), lr=0.001, momentum=0.9, weight_decay=5e-4)
D_out = 100 #feature dimension
D_in = 40 #Number of incoming instances
D_exin = 40 #Maximum number of existing instances
fraction = 0.20
m = 10   # number of subquantizers for FAISS
n_bits = 2 # bits for FAISS
d = 100 #feature dimension
k = 40 #Number of closest neighbours for each instance from FAISS

#optimizer1 = optim.SGD(red_dim.parameters(), lr=0.001, momentum=0.9, weight_decay=5e-4)
param_state={}
paraf ={}
lr = 0.001
weight_decay=5e-4
momentum=0.9
dampening = 0
paradictc = {}
paradicts = {}
paramsel = {}
pc = {}
ps = {}
pred = {}
fextemp = []
#Training epoch#######################
for epoch in range(5):
  
    print("Epoch "+str(epoch))
    fextemp = []
    ftime = 1
    running_loss = 0.0
    running_tloss = 0.0
    total = 0
    correct = 0
    for i, data in enumerate(train_loader, 0):
        inputs, labels = data
        batch,c,h,w = inputs.size()
        D_in = batch
        inputs, labels = inputs.cuda(), Variable(labels.cuda())
        
        fea_inc, fea_inc_diff, fea_inc_tch, features = extract_features_diff(inputs,resextract,extractor,red_dim)
        #fea_inc_diff = D^n

        if i>0: #Find neighbours
          ##########FAISS#############
          fextmp = np.float32(fextmp)
          fea_inc = np.float32(fea_inc)
          if fextmp.shape[0]==0:
            D_exin = 40
            fea_ex_diff = find_feadiff_nsq(fea_inc_tch, torch.zeros((40,100)).cuda())
          elif fextmp.shape[0]==1:
            fexv_tch = fxtch
            fea_ex_diff = find_feadiff_nsq(fea_inc_tch,fexv_tch) #D^o
            D_exin = fexv_tch.shape[0]
          else:
            if fextmp.shape[0]>1 and fextmp.shape[0]<=256:
              n_bits = int(torch.log2(torch.Tensor([fextmp.shape[0]])))
            else:
              n_bits = 8
            pq = faiss.IndexPQ(d, m, n_bits)                       # Training
            pq.train(fextmp)
            pq.add(fextmp)                          # Populate the index
            D, I = pq.search(fea_inc, k) #FAISS distance and index
            chosenind = close_ind(I)
            fexv_tch = extract_ind_tch(fxtch, chosenind)
            fea_ex_diff = find_feadiff_nsq(fea_inc_tch,fexv_tch) #D^o
            D_exin = fexv_tch.shape[0]
          
        else:
          D_exin = 40
          fea_ex_diff = find_feadiff_nsq(fea_inc_tch, torch.zeros((40,100)).cuda())
      
        if i== 0:
          layer = subsetLayer(D_in, D_exin)
        elif D_exin < 40:
          layer = subsetLayer(D_in, D_exin)
        elif ftime==1 and D_exin == 40 and i!=0:
          ftime = 2
          layer = subsetLayer(D_in, D_exin)
        
        sub_time = time.time()
        zv = layer(fea_inc_diff, fea_ex_diff)[0]
        print("Subset computation of 40 x 40 takes")
        print(time.time()-sub_time)
        
        repz = torch.norm(zv, p=1, dim=0)
        repzd = repz.double()
        repzf = repz.double()
        
        sv = torch.where(repz > 0.9, repz, torch.as_tensor(0.0).to(device='cuda'))
        
        svn = sv/0.9

        if torch.max(svn)!=0.0:
          svf = svn/torch.max(svn)
        else:
          svf = svn
      
        svfc = svf.to(device='cuda')
        #print("Selected vectors")
        #print(sv)
        #########For populating existing set with indices of selected instances########
        sc = svfc > torch.min(svfc)
        sind = list(sc.nonzero())
        #print(sind)
        if len(sind)!=0:
          subset_show(sind,epoch,i,inputs,labels)

        for ind in range(len(sind)):
            fextemp.append(fea_inc[sind[ind].item()])

        fextmp = np.asarray(fextemp)
        fxtch = torch.from_numpy(fextmp).cuda() #Existing set

      
        outputs = classf(features.squeeze(2).squeeze(2))
        loss = F.cross_entropy(outputs, labels, reduction='none')
        
        lossv = sv * loss
      
        #Similar to optimiser.zero_grad()

        for name,param in classf.named_parameters():
          #break
          if param.grad is not None:
            if param.grad.grad_fn is not None:
              param.grad.detach_()
            else:
              param.grad.requires_grad_(False)
            param.grad.zero_()
        #$\nabla_{\theta} Trloss(\phi,\theta_{t-1}) = \sum_{j\in training data} (s_j(\phi) \nabla_{\theta} Loss(x_j,y_j,\theta_{t-1}) $
        for lind in range(loss.shape[0]):
          loss[lind].backward(retain_graph=True)
          for name,param in classf.named_parameters():
            if lind == 0:
              pc[name]=torch.full_like(param.grad,0)
              ps[name]=torch.full_like(param.grad,0)

            pc[name] = pc[name].clone()+ svfc[lind].clone()*param.grad.clone()
            ps[name] = ps[name].clone()+ svfc[lind].clone()*param.grad.clone()
 
        for name, param in classf.named_parameters():
          paradicts[name] = ps[name].clone()
          paradictc[name] = pc[name].clone()

        #$\hat{\theta}_t(\phi,\theta_{t-1})$ = $\theta_{t-1} - \eta \nabla_{\theta} TrLoss(\phi,\theta_{t-1})$
        for name,param in classf.named_parameters():
          paramsel[name] = param.data - lr*paradicts[name]

        tloss = 0.0
        batchent = 0
        correct = 0
        total = 0
        classft = Net_test(paramsel) # \hat{\theta}_t(\phi,\theta_{t-1})
        classft = classft.cuda()
        for testi, data in enumerate(test_loader,0):       
              images, labels = data
              images, labels = images.cuda(), Variable(labels.cuda())
              intmfeatures = resextract(images)
              testfeatures = F.avg_pool2d(intmfeatures, 4).squeeze(2).squeeze(2)
              testop = classft(testfeatures)
              losstest = F.cross_entropy(testop, labels, reduction='mean')
              tloss = tloss + losstest.item()
              batchent = batchent + 1
              losstest.backward(retain_graph=True)

              for name,param in red_dim.named_parameters():
                if testi == 0:
                  pred[name]=torch.full_like(param.grad,0)
                pred[name] = pred[name].clone()+ param.grad.clone()

              #optimizer1.zero_grad()
              for name,param in red_dim.named_parameters():
                #print(name)
                if param.grad is not None:
                  if param.grad.grad_fn is not None:
                    param.grad.detach_()
                  else:
                    param.grad.requires_grad_(False)
                  param.grad.zero_()

              _, predicted = torch.max(testop.data, 1)
             
              total += labels.size(0)
              correct += (predicted == labels).sum()
              
        
        #$\phi_t$ = $\phi_{t-1} - \eta \nabla TsLoss(\phi_{t-1})$
        with torch.no_grad():
          for name,param in red_dim.named_parameters():
            pred[name] = pred[name].add_(param,alpha=weight_decay) #Weight decay
            param.add_(pred[name],alpha=-lr) #Learning rate

        # $\theta_t$ = $\theta_{t-1} - \eta \nabla TrLoss(\phi,\theta_{t-1})$
        with torch.no_grad():
          for name,param in classf.named_parameters():
            paradictc[name] = paradictc[name].add_(param,alpha=weight_decay) #Weight decay
            param.add_(paradictc[name],alpha=-lr) #Learning rate

        running_loss += lossv.mean().item()
        running_tloss += loss.mean().item()
        if i % 20 == 19:
            running_loss = 0.0
            running_tloss = 0.0

print('Done')