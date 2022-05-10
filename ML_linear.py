import padercontrib as pc
from paderbox.notebook import *
import paderbox as pb
import padertorch as pt

import torch
import torchvision
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torchvision import transforms
from torchvision.datasets import MNIST
from torch.autograd import Variable
from matplotlib import pyplot as plt
import numpy as np
import random
import sys

from torch.distributions import Normal, Bernoulli

from torch.utils.tensorboard import SummaryWriter



writer = SummaryWriter("runs/ml_vae_mnist_fc_change")


def save_checkpoint(state, filename="ml_vae_mnist_fc_change.pth.tar"):
    torch.save(state, filename)

cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if cuda else "cpu")

batch_size = 64
data = pc.database.mnist.MnistDatabase()
train = data.get_dataset('train')
train_loader = DataLoader(dataset=train, batch_size=batch_size, shuffle=True, num_workers=8)
n_steps = len(train_loader)


"""Model"""
class ML_VAE_encoder(nn.Module):
    def __init__(self, style_dim, content_dim):
        super(ML_VAE_encoder, self).__init__()

        self.encoder = nn.Sequential(
                            nn.Linear(28*28, 500, bias=True),
                            nn.Tanh(),
                            )
        
        
        """Style Content"""
        self.style = nn.Linear(500, style_dim*2, bias=True)
        
        """Content_Content"""
        self.content = nn.Linear(500, content_dim*2, bias=True)                                 

        
    def encode(self, x):
        mu_logvar = self.encoder(x.view(-1,784))
        """Style"""
        style= self.style(mu_logvar).view(-1,2, style_dim)
        
        style_mu = style[:,0,:]
        style_logvar = style[:,1,:]
        
        
        """Content"""
        content = self.content(mu_logvar).view(-1, 2, content_dim)
        content_mu = content[:,0,:]
        content_logvar = content[:,1,:]
        
        return style_mu, style_logvar, content_mu, content_logvar
    
   

    def forward(self, x):
        style_mu, style_logvar, content_mu, content_logvar = self.encode(x)
        
        return style_mu, style_logvar, content_mu, content_logvar 
    

    
"""Decoder"""

class ML_VAE_decoder(nn.Module):
    def __init__(self, style_dim, content_dim):
        super(ML_VAE_decoder, self).__init__()
        
        self.decoder = nn.Linear(style_dim+content_dim, 500)
        self.decode_mu = nn.Linear(500, 784, bias=True)
        self.decode_var = nn.Linear(500, 784,  bias=True)
        
        
    def forward(self, style_dim, content_dim):
        batch_size = style_dim.size(0)
        latent_code = torch.cat((style_dim,content_dim), dim=1)
        z = torch.tanh(self.decoder(latent_code))
        mu = self.decode_mu(z).view(batch_size, 1,28,28)
        
        return mu



    
    
def accumulate_evidence(mu,logvar, digit_batch):
    """ This method is adapted from https://github.com/DianeBouchacourt/multi-level-vae""
    content_mu = []
    content_logvar = []
    batch_labels = []
    """Convert the batch of tensor digits to list"""
    group_labels = digit_batch.tolist()
    """Number of groups in the batch"""
    labels = set(digit_batch.tolist())
    l = 0
    
    for label in labels:
        group = digit_batch.eq(label).nonzero().squeeze()
        group_length = []
        
        for i in range(len(group_labels)):
            """Sort the batch group in the ordered list"""
            if l==0:
                batch_labels.append(group_labels[i])
                l=1   
            """For if condition we calculate the group_length"""
            if label == group_labels[i]:
                group_length.append(group_labels[i])


        if len(group_length) > 0:
            """Calculating the group_mu and group variane"""
            group_var = -logvar[group,:]
            inv_group_var = torch.exp(group_var)
            group_mu = mu[group,:]*inv_group_var
            
            
        if len(group_length) > 1:
            """Sum the group_mu and group_variance"""
            group_mu = torch.sum(group_mu, dim=0)
        
            inv_group_var = torch.logsumexp(inv_group_var, dim=0)

        content_mu.append(group_mu)       
        content_logvar.append(inv_group_var) 
        
    content_mu = torch.stack(content_mu, dim=0)
    content_logvar = torch.stack(content_logvar, dim=0)

    """Calculate the sum of the content_mu and Content_varaiance"""
    content_logvar = - content_logvar
    content_mu = content_mu * torch.exp(content_logvar)
    
    return content_mu, content_logvar, labels



def group_wise_reparameterise(training, content_mu, content_logvar, digit_batch):
    
    if training:
        std = content_logvar.mul(0.5).exp_()
    else:
        std =torch.zeros_like(content_logvar)
    eps = {}
    content_latent_space = []
    labels = set(digit_batch.tolist())
    group_labels = digit_batch.tolist()

    for label in labels:
        eps[label] = torch.FloatTensor(1, std.size(1)).normal_(mean=0,std=0.1)

    for i in group_labels:
        for j,label in enumerate(labels):
            if label == i:
                reparameterise = std[j]*eps[label] + content_mu[j]
                content_latent_space.append(reparameterise)

    content_latent_space = torch.cat(content_latent_space, dim=0)


    return content_latent_space
        
    
def reparameterise(training, mu, logvar):
    
    if training:
            std = logvar.mul(0.5).exp_()
            eps = torch.zeros_like(std).normal_()
            return eps.mul(std).add_(mu)
    else:
        return mu
            
        
        

style_dim = 10
content_dim = 10
encoder = ML_VAE_encoder(style_dim, content_dim)
decoder = ML_VAE_decoder(style_dim, content_dim)
    
    
learning_rate = 1e-3
flag = True

optimizer = torch.optim.Adam(
   list(encoder.parameters()) + list(decoder.parameters()),
    lr=learning_rate,
)

loss = nn.MSELoss(reduction='sum')




epochs = 2000

for k,epoch in enumerate(range(1,epochs+1)):
    content_loss = 0.0
    style_loss = 0.0
    mse_loss = 0.0
    elbo_loss = 0.0
    for i,training in enumerate(train_loader):
        image = training['image']
        
        images = torch.unsqueeze(image,1)

        digit_batch = training['digit']
        
        optimizer.zero_grad()
        
        """Encoder"""
        style_mu, style_logvar, content_mu, content_logvar = encoder(images)

        """Accumulating group evidence"""
        group_mu, group_var, labels = accumulate_evidence(content_mu, content_logvar, digit_batch)

        """Style reaparameterisation"""
        style_latent_space = reparameterise(flag, style_mu, style_logvar)

        """Content reaparameterisation"""
        content_latent_space = group_wise_reparameterise(flag, group_mu, group_var,digit_batch)

        """Reconstruct the samples"""
        decode = decoder(style_latent_space, content_latent_space)
        
        """KL divergence loss for style latent space"""
        style_kl_loss = 0.5 * torch.sum(style_logvar[:].exp() - style_logvar[:] - 1 + style_mu[:].pow(2), dim=1)
        
        style_kl_loss = torch.sum(style_kl_loss)
        
        """KL divergence loss for content latent space"""  
        content_kl_loss = 0.5 * torch.sum(group_var.exp() - group_var - 1 + group_mu.pow(2), dim=1)
        
        content_kl_loss = torch.sum(content_kl_loss)

        """MSE Loss"""
        mse = loss(decode, images)
        
        
        """ELBO"""
        ELBO = (mse + style_kl_loss + content_kl_loss)/len(labels) #
        ELBO.backward()
        
        
        optimizer.step()
        
        elbo_loss +=ELBO.item()
        mse_loss += (mse/len(labels)).item()
        style_loss += (style_kl_loss/len(labels)).item()
        content_loss += (content_kl_loss/len(labels)).item()
    

    writer.add_scalar("ELBO loss", elbo_loss/n_steps, epoch*n_steps)
    writer.add_scalar("MSE loss", mse_loss/n_steps, epoch*n_steps)
    writer.add_scalar("Style loss", style_loss/n_steps, epoch*n_steps)
    writer.add_scalar("Content loss", content_loss/n_steps, epoch*n_steps)
    
    checkpoint = {'encoder': encoder.state_dict(), 'decoder': decoder.state_dict(), 'optimizer': optimizer.state_dict()}
    save_checkpoint(checkpoint)
    if k%100 == 0:
        original = torchvision.utils.make_grid(images)
        reconstruction = torchvision.utils.make_grid(decode)
        writer.add_image(f'Original_{k}:', original)
        writer.add_image(f"Reconstruction_{k}:", reconstruction)
    
        
    
    print(f'Epoch: {epoch} || ELBO: {ELBO: .4f} || MSE: {mse/len(labels): .3f} || Style_loss: {style_kl_loss/len(labels):.7f} || Content_loss: {content_kl_loss/len(labels):.3f}')
