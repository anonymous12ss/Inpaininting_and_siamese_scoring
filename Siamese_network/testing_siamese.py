# %matplotlib inline
import torchvision
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader,Dataset
import matplotlib.pyplot as plt
import torchvision.utils
import numpy as np
import random
from PIL import Image
import torch
from torch.autograd import Variable
import PIL.ImageOps    
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import os
from torch.utils import data
import pandas as pd
from random import randint

#torch.cuda.set_device(1)
torch.cuda.empty_cache()

#os.environ['CUDA_VISIBLE_DEVICES']='0' # pick GPU to use
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")

def imshow(img,text=None,should_save=False):
    npimg = img.numpy()
    plt.axis("off")
    if text:
        plt.text(75, 8, text, style='italic',fontweight='bold',
            bbox={'facecolor':'white', 'alpha':0.8, 'pad':10})
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()    

def show_plot(iteration,loss):
    plt.plot(iteration,loss)
    plt.show()
    
class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
        
    def forward(self, x):
        return x




class PretrainedNet(nn.Module):
    """
    Siamese neural network
    Modified from: https://hackernoon.com/facial-similarity-with-siamese-networks-in-pytorch-9642aa9db2f7
    Siamese ResNet-101 from Pytorch library
    """
    def __init__(self):
        super(PretrainedNet, self).__init__()
        self.cnn = torchvision.models.resnet18()
        #self.cnn.fc = Identity()
        #for name, para in self.cnn1.named_parameters():
        #    if para.requires_grad:
        #        print (name)
        #alex_load.classifier[4] = nn.Linear(in_features = alex_load.classifier[1].out_features, out_features = 1000, bias = True)
        #self.cnn1.classifier[6] = nn.Linear(in_features = self.cnn1.classifier[4].out_features, out_features = 2, bias = True)
        
        
    def forward_once(self, x):
        output = self.cnn(x)
        return output

    def forward(self, input1, input2):
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)
        return output1, output2



class ContrastiveLoss(torch.nn.Module):
    """
    Contrastive loss function.
    Based on: http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    """

    def __init__(self, margin=2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        euclidean_distance = F.pairwise_distance(output1, output2, keepdim = True)
        loss_contrastive = torch.mean((1-label) * torch.pow(euclidean_distance, 2) +
                                      (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))
        return loss_contrastive
    

'''
class TestConfig():
    testing_table = pd.read_csv('/mnt/recsys/daniel/simase_network/dataset_256_0_pct_all/csv_files/all_images.csv')
    image_dir = '/mnt/recsys/daniel/simase_network/dataset_256_0_pct_all/data'
    scores_csv_file = pd.read_csv('/mnt/recsys/daniel/datasets/gan_masks_classify_cleft/cleft_lip_scores.csv')
'''
class TestConfig():
    testing_table = pd.read_csv('/mnt/recsys/daniel/simase_network/dataset_256_0_pct_all/csv_files/all_images_cropped.csv')
    image_dir = '/mnt/recsys/daniel/simase_network/dataset_256_0_pct_all/cropped/'
    scores_csv_file = pd.read_csv('/mnt/recsys/daniel/datasets/gan_masks_classify_cleft/cleft_lip_scores.csv')
    
class siamese_test_Dataset(torch.utils.data.Dataset):

    def __init__(self, patient_table, image_dir,scores_csv_file,  transform=None ):
        
        self.patient_table = patient_table
        self.image_dir = image_dir
        self.transform = transform
        self.epoch_size = len(self.patient_table)
        self.scroes_file = scores_csv_file
        if self.transform is None:
            self.transform = transforms.Compose([transforms.ToTensor()])

    def __len__(self):
        return self.epoch_size

    def __getitem__(self, index):
        name_list = self.patient_table
        num_entries = len(name_list)
        label = 1

        while True:
            #random_image = random.choice(name_list).split('.')[0]+'.png' # note that processed images are all .png type, while patient_table has different types
            #img0 = None; img1 = None
            random_num = index
            
            random_row = name_list.iloc[random_num]
            patient_id = random_row['File_name']
            
            truth_image_name = patient_id + "_truth.png"
            fake_image_name = patient_id + '_out.png'
            #print (fake_image_name, truth_image_name)
            #print (patient_id[:-1])
            if (truth_image_name in os.listdir(os.path.join(self.image_dir, "truth"))) and (fake_image_name in os.listdir(os.path.join(self.image_dir,  "out"))):
                break
            elif fake_image_name in os.listdir(os.path.join(self.image_dir, "out")):
                print('attempted to get following image, but missing: ' + truth_image_name)
            elif truth_image_name in os.listdir(os.path.join(self.image_dir,  "truth")):
                print('attempted to get following image, but missing: ' + fake_image_name)
            else :
                print (fake_image_name)
        #print (truth_image_name, fake_image_name)
        #print (truth_image_name, fake_image_name)
        doctor_score = self.scroes_file.loc[self.scroes_file['Patient number'].isin([patient_id])]['Lip Score'].item()
        truth_img_dir = os.path.join(self.image_dir,  "truth")
        fake_img_dir = os.path.join(self.image_dir,  "out")
        img0 = Image.open(truth_img_dir +'/' +truth_image_name).convert("RGB")
        img1 = Image.open(fake_img_dir +'/' +fake_image_name).convert("RGB")
        #print (truth_image_name, fake_image_name)
        k1, k2 = img0.size, img1.size
        if self.transform is not None:
            img0 = self.transform(img0)
            img1 = self.transform(img1)
        #print (img1.shape)
        return img0, img1, patient_id, doctor_score, label



checkpoints_dir = "./ckpt/resnet_final"
test_epoch = 13

'''
checkpoints_dir = "./ckpt/resnet_same"
test_epoch = 25
'''
#test_model = SiameseNetworkbasic().cuda()
test_model = PretrainedNet().cuda()
test_model.load_state_dict(torch.load(checkpoints_dir + "/base_model_epoch{}.pth".format(test_epoch)))
test_model.eval()

test_transforms = transforms.Compose([ # pixel crop
    transforms.ToTensor()
])



testing_siamese_dataset = siamese_test_Dataset(patient_table = TestConfig.testing_table,
                                          image_dir = TestConfig.image_dir,
                                          scores_csv_file = TestConfig.scores_csv_file,
                                          transform = test_transforms)

scores_list = []
dissimilarity_score = []
patient_id = []
labels = [] 
x_axis = [x for x in range(138)]

#print(len(test_dataloader))

for test_cnt in range(138):
    x_0, x_1, p_id , score, label= testing_siamese_dataset[test_cnt]
    #print (x_0.shape)
    concatenated = torch.cat((x_0,x_1),0)
    output1,output2 = test_model(Variable(x_0).unsqueeze(0).cuda(),Variable(x_1).unsqueeze(0).cuda())
    euclidean_distance = F.pairwise_distance(output1, output2)


    scores_list.append(score)
    dissimilarity_score.append(euclidean_distance.item())
    #print (type(p_id))
    patient_id.append(p_id)
    labels.append(label)
    #imshow(torchvision.utils.make_grid(concatenated),'Dissimilarity: {:.2f}_Score{:.2f}'.format(euclidean_distance.item()*1000,score.item() ))
    #print (euclidean_distance.item())
    #print (p_id, euclidean_distance.item())
dissimilarity_score = np.array(dissimilarity_score).astype(np.float64)
scores_list = np.array(scores_list).astype(np.float64)

plt.scatter(scores_list, dissimilarity_score)
a, b = np.polyfit(scores_list, dissimilarity_score,  1)
plt.plot(scores_list, a*scores_list+b, 'r-') 
plt.show()