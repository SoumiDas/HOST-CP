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

#Test set indices
subset_indices = [3931,  787, 9037, 2093, 3403, 6198, 4153, 9401, 8555, 6480, 1477, 7734,
        2200, 9095, 3217, 6508, 6787, 7774, 8753, 3195, 5433, 2385,  298, 8668,
         966, 5049, 1802, 1169, 5202, 1597, 1130,  223,   70, 7052, 7875, 5275,
        5638,  898, 5441, 7044, 3046, 6796, 3661, 3740, 6143, 9969,  371, 1996,
        3927, 3891, 4853, 3223, 8155, 1681, 9751, 3520, 4147, 5324, 9301, 8977,
        2007, 1369, 3681, 9870, 2452, 9935, 5743, 6333, 6042, 2601, 6683, 7609,
         478, 8845, 5886, 9830, 2925, 1575, 8978, 4339, 6462, 2027, 5059, 8897,
        8953, 2667,  677, 8906, 8159, 1424, 2475, 6730, 7178, 1158, 5098, 9011,
        7533, 9598, 3911, 3407, 9219, 8374, 5824, 3010, 2607, 2190, 8337, 9028,
        8100, 3218, 7372, 9884, 3883, 6699, 5944, 7498, 7340,  751, 6619, 4148,
        8956,  233, 2421, 3714, 1915, 4187, 7695, 1393, 2678, 1995, 1161, 8882,
        6136, 2884, 5472, 2721,   84, 7900, 1114, 1828, 4580, 4769, 3009, 8649,
        2196, 3674, 9535, 4281, 5238, 6972, 1809, 1718, 3559, 2098, 5140, 2457,
        8450, 2302, 4542, 7337, 1038, 1432, 1540, 9919, 7239, 6291, 1101, 5324,
        5136,   35, 7403, 7126, 5975, 2897, 8563,  276, 8893, 3334, 8461,  558,
         858, 3840,  393, 6594,  973, 3030,  163, 1671, 8345,   89, 9470, 5926,
        4830, 7840, 4200, 9457, 7081, 5307, 5075, 7901, 9334, 2244, 5877,  401,
         175, 9469, 7387, 4376, 8069, 3356, 1887, 4454, 8290, 1860, 8342, 7624,
        2334, 4033, 8871, 1738, 9796,  699, 2105, 8681, 9676, 1458, 8954, 5419,
         400, 3312, 9167, 3859, 7466, 5339, 6708, 1373, 8989, 6925, 9961, 9404,
        6573, 1892, 5266, 5121, 6344, 4770, 2506, 1398, 4477, 9519,  164, 4329,
        3396, 4508, 2879, 1613, 1631, 3894, 3897, 8814, 6983, 8458, 5328, 2295,
        8674, 7034, 4870, 4434, 9201, 8114, 3510,  977,  350, 7326, 9358, 6057,
        4952, 4188, 3081, 7216, 2060, 8913, 2059,  277, 1988, 1324,  618, 6239,
        5164, 8070, 6947, 6424, 3111, 7730, 9072, 4285, 2721, 5364,  736, 9366,
        7434, 5246, 2920, 9535, 5407, 7768, 6199, 8621, 3312, 2945, 9461, 5093,
        3298, 8497,  186, 8724, 8406, 6843, 9769, 9309, 1239, 7198, 1505, 9542,
        4403,  253, 1579,  282, 6906, 7894, 7616, 7287,  870,  694, 8208, 2572,
        4075,  435, 8718, 9967, 2828, 9941, 9298, 4536, 8576, 8074, 8659, 6814,
        4040, 5515, 7980, 8782,  742, 7119, 2759, 8520, 1757, 2868, 5859, 7830,
        2395, 4589, 1653, 6908, 1475, 8094, 5729, 2776, 2195, 8454, 3194, 9282,
        2740, 2810, 2775, 1635, 1291, 1413, 6260, 4176, 5144, 3344, 2763, 4866,
        4191, 9116, 5569, 8005,  779, 8885, 8645, 1299, 5861, 6932, 7442, 5353,
        5058, 1421, 5829, 4625, 5082, 4344, 8629, 8088, 4833, 1251,  736,  772,
        7405, 7700, 5690, 2472, 3275, 3714, 5092, 8290, 5569, 3738, 5447, 3842,
         611, 4640, 7876, 9941, 5255, 8458, 9888, 9740, 7067, 2642, 2918, 5709,
        2632, 2222, 9127, 4917, 9142, 7335, 8823, 5234, 9745, 5302, 4943, 8512,
        2309, 3181, 9778, 5101, 9286,  945, 1877, 6426, 5042, 6693, 5866, 6204,
        4630, 3082, 6596, 3786, 7820, 7690, 9825, 3403, 5803, 5448,  399, 7655,
         474, 4439, 8000, 9025,  119, 5972, 8293, 1810,  689, 9312, 5991, 8546,
        4006, 6478, 8707, 9881, 5626, 5389, 9469, 7737,  666, 3640, 7726, 5614,
         654, 9194,  531, 2425, 7944, 5630, 2854, 7977, 4797,  753, 7536, 6869,
        8740, 4418, 3525, 2810, 2971, 9519, 9346, 5006, 9415, 2206, 1251, 9253,
        8076, 7949, 1197, 6741, 9082, 1213, 2294, 6322, 4984, 7998, 3446, 7075,
        3663, 6907, 1160, 6927, 1614, 7869, 6938, 4049, 6983, 1690, 1726,  629,
        5728, 9530, 3058, 5600, 5108,  420, 2279, 6550, 7239, 2445, 2358,  535,
        2365, 4424, 6462, 3691, 6874, 2546, 6750, 8276, 5555, 8699,  404, 1220,
         303,  794, 2214, 1375, 6221, 5957, 3869,  230, 7070, 1789, 6805, 1537,
        2936, 7005, 6558, 6722, 6133, 5618, 4246, 1770,  417, 2766,  672, 5461,
        5519, 1629, 8978, 4205, 7698, 8404, 2052, 6490, 8007, 8646, 9671, 1882,
        9555, 4484, 4123, 8334, 4384, 9801, 1734, 8727, 5523, 8376, 7851,  280,
          67,  697, 7792, 1018, 1187, 4226, 3311, 1752, 5996, 8384,  421, 4822,
        4815, 9868, 5374,  433, 8967, 2833, 5386, 5421, 8851, 9471, 6794, 6850,
        8408, 3916, 4862, 6920, 5018, 6625, 2855, 6983, 6792, 1430, 9160, 9523,
        6100, 7110, 5331,  606, 3323, 8398, 4095, 5960, 9491, 2632, 9266, 7723,
        2774,  377, 4547, 1080, 4013, 2576, 9600,  168, 8602, 1214, 5474,  266,
        2754, 6240,  917, 8777, 2573, 5457, 5642, 9331, 3585,  759, 4979,  659,
        2467, 9321, 2837, 2492,  959, 8353,  444, 2748, 6760, 3189, 4766, 4911,
        1779, 5781, 4971, 3567, 9518, 8181, 2893, 8746, 2800, 4340,  892, 3822,
        7755, 8167, 8793,  757,  870, 6267,  452, 2773, 1406, 1787, 8478, 2457,
        4518, 5107, 7134, 9588, 9042, 8279, 1740, 8187, 7242, 6972, 7201, 1921,
        2892, 3495,  417, 7954, 7568, 1751, 7951, 6964, 8515, 5557, 6844, 4709,
         207, 5140,  985, 5353, 9499,  714, 3527, 8650, 6180, 5081,  650, 8424,
        2025, 9194, 7185, 1890,  321, 1820, 5006, 9469, 9458, 3474, 5839, 8193,
        8316, 2302, 7837, 9436, 2443, 9611, 9269, 8885, 6630, 3293, 7183, 4341,
        3104, 5923, 7546, 8417, 2966, 8432, 7323, 2523, 2392, 1869, 8470, 5781,
        5454, 2902, 6327, 6871, 9299, 8627,  375, 9219, 5296, 8261, 6970, 6440,
        9416, 7475, 1754, 2137, 8365,  181, 2917, 5660, 5369, 5955, 8289,  206,
        3450, 6610, 2115, 5530, 9779,  407, 3387, 4520,  197, 3585, 4576, 8411,
        9924, 9884, 3611, 1220, 6801, 3264, 1345, 4822, 8530, 9536, 5469, 3198,
        2094, 4712, 4345, 5896, 1133, 6915, 8677, 9905, 9037, 3460, 8398,  625,
         485, 8901, 6767,  660, 3465, 4707, 8034, 9951, 9979, 7635, 8221, 3198,
        4690, 4695, 4108, 5411, 1986, 3557, 3106, 3553,  462, 5581, 5048, 4177,
         119, 9608,  861, 9203, 5022, 9921, 4891, 5698,  285, 3909,    1, 3327,
        8060, 1754, 3228, 7108, 3632, 5388, 1165, 2799, 3193, 7903, 1819,  324,
        2760, 8348, 3843, 9110, 9310, 8593, 9014, 4412, 4615, 4290, 1803, 7076,
         630, 2054, 8183, 7630, 4213, 8118, 8109, 5619,   51, 7331, 8763, 8005,
        7934, 5920, 9288,  688, 9459, 7734, 4209,  802, 7661, 3584, 2262, 7451,
        6968, 1370, 7544, 6422, 7199, 8445, 3293, 8395, 2624, 2985, 2811, 8654,
        5169, 5724,  460,  356, 8258, 7928, 9368,  355, 4905, 2073,  267,  231,
        3340, 1558, 5377, 1992, 4370, 5017, 9984, 1916, 9055, 7130, 2212, 6744,
        4372, 6677, 1939, 8008, 7356, 1515, 7606, 5127,  563, 4740, 5212, 8827,
        4992, 8039, 2608, 9717, 2196, 7930,    1, 9271, 5828, 8779, 4722, 9631, 7058, 6197, 8390, 4385]

subset = torch.utils.data.Subset(dataset2, subset_indices)

train_loader = torch.utils.data.DataLoader(dataset1,**kwargs)

test_loader = torch.utils.data.DataLoader(subset, **kwargstest)
#test_wloader = torch.utils.data.DataLoader(dataset2, **kwargstest)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


D_out = 100 #feature dimension
D_in = 40 #Number of incoming instances
D_exin = 40 #Maximum number of existing instances
fraction = 0.05
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
    save_image(im[si],'./epoch_images_5_15ep'+str(epoch)+'/im_'+str(batch)+'_'+str(si)+'.png')'''
  
  ########Use the selected indices to train on the subset from scratch after subset selection############
  
  with open('./selected_indices_5.txt','a') as fpsi:
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
fraction = 0.05
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
begin = time.time()
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
        if i % 20 == 19:    # print every 20 mini-batches
            running_loss = 0.0
            running_tloss = 0.0

print('Finished Training')
print(time.time()-begin)