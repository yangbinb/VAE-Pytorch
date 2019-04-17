import torch
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
import torchvision
from torchvision import transforms
import torch.optim as optim
from torch import nn
import matplotlib.pyplot as plt
from tensorboardX import SummaryWriter
from torchvision.utils import make_grid
import os
import utils
from datetime import datetime
import socket

class Network(nn.Module):
    # The basic module used by Encoder and Decoder
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(Network, self).__init__()
        modules = [nn.Linear(input_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, output_dim),
        ]
        self.net = nn.Sequential(*modules)
    def forward(self, x):
        return self.net(x)


class VAE(torch.nn.Module):
    def __init__(self, input_dim=784, hidden_dim=256, output_dim=256, latent_dim=128):
        super(VAE, self).__init__()
        self.encoder = Network(input_dim, hidden_dim, output_dim)
        self.decoder = Network(latent_dim, hidden_dim, input_dim)

        self.encode_mu = nn.Linear(output_dim, latent_dim)
        self.encode_log_sigma = nn.Linear(output_dim, latent_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, inputs):
        x = self.encoder(inputs)
        mu = self.encode_mu(x)
        log_sigma = self.encode_log_sigma(x)
        sigma = torch.exp(log_sigma)
        z = mu + sigma * torch.normal(torch.zeros(sigma.size()), torch.ones(sigma.size())).cuda()
        outputs = self.decoder(z)
        return self.sigmoid(outputs), mu, sigma


def latent_loss(z_mean, z_stddev):
    mean_sq = z_mean ** 2
    stddev_sq = z_stddev ** 2
    return 0.5 * torch.sum(mean_sq + stddev_sq - torch.log(stddev_sq) - 1)


if __name__ == '__main__':
    save_dir = './log'
    logdir = os.path.join(save_dir, datetime.now().strftime('%b%d_%H-%M-%S') + '_' + socket.gethostname())
    writer = SummaryWriter(log_dir=logdir)
    img_dim = 28
    input_dim = img_dim ** 2
    batch_size = 40

    transform = transforms.Compose(
        [transforms.ToTensor()])
    mnist = torchvision.datasets.MNIST('./', download=True, transform=transform)

    dataloader = torch.utils.data.DataLoader(mnist, batch_size=batch_size,
                                             shuffle=True, num_workers=20)

    print('Number of samples: ', len(mnist))

    vae = VAE().cuda()

    criterion = nn.MSELoss(size_average=False)

    optimizer = optim.SGD(vae.parameters(), lr=0.0001)
    idx = 0
    for epoch in range(100):
        print('Epoch: ', epoch)
        for i, data in enumerate(dataloader):
            inputs, classes = data
            inputs, classes = Variable(inputs.resize_(batch_size, input_dim)), Variable(classes)
            inputs, classes = inputs.cuda(), classes.cuda()
            optimizer.zero_grad()
            dec, z_mean, z_sigma = vae(inputs)
            ll = latent_loss(z_mean, z_sigma)
            reconstruction_error = criterion(dec, inputs)
            r_e = reconstruction_error.detach().data
            loss = reconstruction_error + ll
            loss.backward()
            optimizer.step()
            outputs = dec.clone().cpu().data.resize_(batch_size, img_dim, img_dim)
            if i % 100 == 0:
                temp = utils.decode_seg_map_sequence(outputs.numpy())
                grid_image = make_grid(temp[:,:,:], 8, normalize=False, range=(0,255))
                writer.add_image('Image', grid_image, idx)
                temp = utils.decode_seg_map_sequence(inputs.clone().cpu().data.resize_(batch_size, img_dim, img_dim).numpy())
                grid_image = make_grid(temp[:,:,:], 8, normalize=False, range=(0,255))
                writer.add_image('GT', grid_image, idx) 
                idx += 1
                print("reconstruction error: ", r_e)
