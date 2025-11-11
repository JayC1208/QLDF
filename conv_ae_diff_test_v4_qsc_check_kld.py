# %% [markdown]
# # Code for Quantum Scrambling Circuit
# 
# * Softmax inserted between encoder and decoder
# * Use squared root of encoder output (amplitude), and use amplitude embedding and add noise using QSC
# 

# %% [markdown]
# # Convolutional Autoencoder
# 
# Sticking with the MNIST dataset, let's improve our autoencoder's performance using convolutional layers. We'll build a convolutional autoencoder to compress the MNIST dataset. 
# 
# >The encoder portion will be made of convolutional and pooling layers and the decoder will be made of **transpose convolutional layers** that learn to "upsample" a compressed representation.
# 
# <img src='notebook_ims/autoencoder_1.png' />
# 
# ### Compressed Representation
# 
# A compressed representation can be great for saving and sharing any kind of data in a way that is more efficient than storing raw data. In practice, the compressed representation often holds key information about an input image and we can use it for denoising images or oher kinds of reconstruction and transformation!
# 
# <img src='notebook_ims/denoising.png' width=60%/>
# 
# Let's get started by importing our libraries and getting the dataset.

# %%
import torch
import numpy as np
from torchvision import datasets
import torchvision.transforms as transforms

from inspect import isfunction
from functools import partial
from tqdm import tqdm
from torch.optim.lr_scheduler import ExponentialLR, ReduceLROnPlateau

transform=transforms.Compose([
transforms.Resize(64), # Reszie from 28x28 to 64x64
transforms.ToTensor(),
#transforms.Normalize((0.1307,), (0.3081,)) # If you do normalization, please denormalize when you visualize your generated data
])
# load the training and test datasetsnvidia
train_data = datasets.MNIST(root='mnist_data', train=True,
                                   download=False, transform=transform)
test_data = datasets.MNIST(root='./mnist_data', train=False,
                                  download=False, transform=transform)

# %%
# Create training and test dataloaders

num_workers = 0
# how many samples per batch to load
batch_size = 64
n_qubit = 4  # 4

# prepare data loaders
train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, num_workers=num_workers)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, num_workers=num_workers)

# %% [markdown]
# ### Visualize the Data

# %%
import matplotlib.pyplot as plt
torch.manual_seed(42)
fig = plt.figure(figsize=(4,4))
rows,cols = 4,4

for i in range(1,cols*rows+1):
    random_idx = torch.randint(0,len(train_data),size=[1]).item()
    img,label = train_data[random_idx]
    fig.add_subplot(rows,cols,i)
    plt.imshow(img.squeeze(),cmap='gray')
    # plt.title(classes[label])
    plt.axis(False);

# %% [markdown]
# ---
# ## Convolutional  Autoencoder
# 
# #### Encoder
# The encoder part of the network will be a typical convolutional pyramid. Each convolutional layer will be followed by a max-pooling layer to reduce the dimensions of the layers. 
# 
# #### Decoder
# 
# The decoder though might be something new to you. The decoder needs to convert from a narrow representation to a wide, reconstructed image. For example, the representation could be a 7x7x4 max-pool layer. This is the output of the encoder, but also the input to the decoder. We want to get a 28x28x1 image out from the decoder so we need to work our way back up from the compressed representation. A schematic of the network is shown below.
# 
# <img src='notebook_ims/conv_enc_1.png' width=640px>
# 
# Here our final encoder layer has size 7x7x4 = 196. The original images have size 28x28 = 784, so the encoded vector is 25% the size of the original image. These are just suggested sizes for each of the layers. Feel free to change the depths and sizes, in fact, you're encouraged to add additional layers to make this representation even smaller! Remember our goal here is to find a small representation of the input data.
# 
# ### Transpose Convolutions, Decoder
# 
# This decoder uses **transposed convolutional** layers to increase the width and height of the input layers. They work almost exactly the same as convolutional layers, but in reverse. A stride in the input layer results in a larger stride in the transposed convolution layer. For example, if you have a 3x3 kernel, a 3x3 patch in the input layer will be reduced to one unit in a convolutional layer. Comparatively, one unit in the input layer will be expanded to a 3x3 path in a transposed convolution layer. PyTorch provides us with an easy way to create the layers, [`nn.ConvTranspose2d`](https://pytorch.org/docs/stable/nn.html#convtranspose2d). 
# 
# It is important to note that transpose convolution layers can lead to artifacts in the final images, such as checkerboard patterns. This is due to overlap in the kernels which can be avoided by setting the stride and kernel size equal. In [this Distill article](http://distill.pub/2016/deconv-checkerboard/) from Augustus Odena, *et al*, the authors show that these checkerboard artifacts can be avoided by resizing the layers using nearest neighbor or bilinear interpolation (upsampling) followed by a convolutional layer. 
# 
# > We'll show this approach in another notebook, so you can experiment with it and see the difference.
# 
# 
# #### TODO: Build the network shown above. 
# > Build the encoder out of a series of convolutional and pooling layers. 
# > When building the decoder, recall that transpose convolutional layers can upsample an input by a factor of 2 using a stride and kernel_size of 2. 

# %%
import torch.nn as nn
import torch.nn.functional as F

class Flatten(nn.Module):
    def forward(self, input):
        
        return input.view(input.size()[0], -1)#.to(DEVICE) # for connecting conv layer and linear layer

    
class UnFlatten(nn.Module):
    def forward(self, input):
        
        return input.view(input.size()[0], 64, 2, 2)#.to(DEVICE) # for connecting linear layer and conv layer

class ConvAutoencoder(nn.Module):
    def __init__(self, image_channels= 1, output_channels= 4, h_dim=256, z_dim=pow(2,n_qubit)): # h_dim : last hidden dimension, z_dim : latent dimension
        super(ConvAutoencoder, self).__init__()
        self.z_dim = z_dim
        self.encoder = nn.Sequential(
            nn.Conv2d(image_channels, output_channels, kernel_size=3, stride=2, padding = 1),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(),
            nn.Conv2d(output_channels, output_channels*2, kernel_size=3, stride=2, padding = 1),
            nn.BatchNorm2d(output_channels*2),
            nn.ReLU(),
            nn.Conv2d(output_channels*2, output_channels*4, kernel_size=3, stride=2, padding = 1),
            nn.BatchNorm2d(output_channels*4),
            nn.ReLU(),
            nn.Conv2d(output_channels*4, output_channels*8, kernel_size=3, stride=2, padding = 1),
            nn.BatchNorm2d(output_channels*8),
            nn.ReLU(),
            nn.Conv2d(output_channels*8, output_channels*16, kernel_size=3, stride=2, padding = 1),
            nn.BatchNorm2d(output_channels*16),
            nn.ReLU(),
            nn.Dropout(0.8),
            nn.Flatten()
        )

        
        self.fc1 = nn.Linear(h_dim, z_dim)#.to(DEVICE) # for mu right before reparameterization
        self.fc3 = nn.Linear(z_dim, h_dim)#.to(DEVICE) # right before decoding starts
        
        self.decoder = nn.Sequential(
            UnFlatten(),
            nn.ConvTranspose2d(output_channels*16, output_channels*8, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(output_channels*8),
            nn.ReLU(),
            nn.ConvTranspose2d(output_channels*8, output_channels*4, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(output_channels*4),
            nn.ReLU(),
            nn.ConvTranspose2d(output_channels*4, output_channels*2, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(output_channels*2),            
            nn.ReLU(),
            nn.ConvTranspose2d(output_channels*2, output_channels, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(output_channels),            
            nn.ReLU(),
            nn.ConvTranspose2d(output_channels, image_channels, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(image_channels),
            nn.Sigmoid() # so that make the range of values 0~1
        )
        

    def encode(self, x):
        h = self.encoder(x)
        h = self.fc1(h)
        h = nn.Softmax()(h)
        return h
        
    def decode(self, z):
        z = F.relu(self.fc3(z))
        z = self.decoder(z)
        return z

    def forward(self, x):
        z = self.encode(x) # save mu and logvar
        z = self.decode(z) # decode reparameterized z
        return z

# initialize the NN
model = ConvAutoencoder()
print(model)

# %% [markdown]
# ---
# ## Training
# 
# Here I'll write a bit of code to train the network. I'm not too interested in validation here, so I'll just monitor the training loss and the test loss afterwards. 
# 
# We are not concerned with labels in this case, just images, which we can get from the `train_loader`. Because we're comparing pixel values in input and output images, it will be best to use a loss that is meant for a regression task. Regression is all about comparing quantities rather than probabilistic values. So, in this case, I'll use `MSELoss`. And compare output images and input images as follows:
# ```
# loss = criterion(outputs, images)
# ```
# 
# Otherwise, this is pretty straightfoward training with PyTorch. Since this is a convlutional autoencoder, our images _do not_ need to be flattened before being passed in an input to our model.

# %%
# specify loss function
DEVICE = 'cuda:1' if torch.cuda.is_available() else 'cpu'
DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'
model = model.to(DEVICE)

criterion = nn.MSELoss()

# specify loss function
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# %%
# number of epochs to train the model
n_epochs = 1

for epoch in range(1, n_epochs+1):
    # monitor training loss
    train_loss = 0.0
    
    ###################
    # train the model #
    ###################
    for data in train_loader:
        # _ stands in for labels, here
        # no need to flatten images
        images, _ = data
        images = images.to(DEVICE)
        # clear the gradients of all optimized variables
        optimizer.zero_grad()
        # forward pass: compute predicted outputs by passing inputs to the model
        outputs = model(images).to(DEVICE)
        # calculate the loss
        loss = criterion(outputs, images)
        # backward pass: compute gradient of the loss with respect to model parameters
        loss.backward()
        # perform a single optimization step (parameter update)
        optimizer.step()
        # update running training loss
        train_loss += loss.item()*images.size(0)
            
    # print avg training statistics 
    train_loss = train_loss/len(train_loader)
    print('Epoch: {} \tTraining Loss: {:.6f}'.format(
        epoch, 
        train_loss
        ))

# %%

resize_transform = transforms.Compose([
    transforms.Resize((28, 28)), # Convert to 28x28 again
])
# obtain one batch of test images
images, labels = next(iter(test_loader))
# images, labels = dataiter.next()

# get sample outputs
output = model(images.to(DEVICE))
# prep images for display

ground_truth_images = torch.stack([resize_transform(image) for image in images]) 
generated_images = torch.stack([resize_transform(image) for image in output])
# output is resized into a batch of iages
print(generated_images.shape)
# output = output.view(batch_size, 1, 28, 28)
# use detach when it's an output that requires_grad
ground_truth_images = ground_truth_images.numpy()
output = generated_images.cpu().detach().numpy()

# plot the first ten input images and then reconstructed images
fig, axes = plt.subplots(nrows=2, ncols=20, sharex=True, sharey=True, figsize=(25,4))

# input images on top row, reconstructions on bottom
for gt, row in zip([ground_truth_images, output], axes):
    for img, ax in zip(gt, row):
        ax.imshow(np.squeeze(img), cmap='gray')
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

# %%
import pennylane as qml
from pennylane import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from math import pi
from datetime import datetime
from scipy.stats import unitary_group
from itertools import combinations

# %% [markdown]
# ### Quantum Scrambling Circuit
# 

# %%

class QSC(nn.Module):
    def __init__(self, n, dev):
        super().__init__()
        self.n = n  # number of qubits
        self.dev = dev

    def GenerateHaarSample(self, NData, seed):
        np.random.seed(seed)
        states_T = unitary_group.rvs(dim=2**self.n, size=NData)[:,:,0]
        return torch.from_numpy(states_T).cfloat()
    
    def step_diffusion(self, ttdx, inputs, diff_hs, seed):

        # use root of inputs to generate latent+noise value
        assert (inputs > 0).all()   # shape of input: [batch_size, latent_size]
        # batch_size = inputs.shape[0]
        # print(inputs.shape)
        batch_size = 1

        np.random.seed(seed)
        phis = torch.rand(batch_size, 3*self.n*ttdx)*np.pi/4. - np.pi/8.
        phis = phis*(diff_hs.repeat(3*self.n))

        gs = torch.rand(batch_size, ttdx)*0.2 + 0.4
        gs *= diff_hs

        probs = torch.zeros((batch_size, 2**self.n))

        @qml.qnode(self.dev, interface="torch")
        def ScrambleCircuit(in_val):
            t = in_val['t']
            inputs = in_val['inputs']
            # print(inputs.shape)
            phis = in_val['phis']
            gs = in_val['gs']

            qml.AmplitudeEmbedding(features=inputs, wires=range(self.n), normalize=True)
            
            for tt in range(t):
                for idx in range(self.n):
                    qml.RZ(phi=phis[3*self.n*tt+idx], wires=idx)
                    qml.RY(phi=phis[3*self.n*tt+self.n+idx], wires=idx)
                    qml.RZ(phi=phis[3*self.n*tt+2*self.n+idx], wires=idx)
                for idx, jdx in combinations(range(self.n), 2):
                    qml.MultiRZ(theta = gs[tt]/(self.n*np.sqrt(self.n)), wires=[idx, jdx])

            return qml.probs()
            
        for i in range(batch_size):
            in_val = dict()
            in_val['t'] = ttdx
            if batch_size == 1:
                in_val['inputs'] = inputs
            else:
                in_val['inputs'] = inputs[i]
            in_val['phis'] = phis[i]
            in_val['gs'] = gs[i]

            probs[i] = ScrambleCircuit(in_val)
            
        return probs
    

# %% [markdown]
# ### Quantum Circuit Class

# %% [markdown]
# * consider checking out this webpage and functions to utilize non-linear transform and ancillary subsystem measurement
# : https://pennylane.ai/qml/demos/tutorial_quantum_gans/
# 

# %%
import pennylane as qml
from pennylane import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from math import pi
from datetime import datetime
# use torch.nn.sequential and batch training for training quantum circuit
# refer to github code https://github.com/salcc/QuantumTransformers/tree/main

def statepreparation(x,num_qubits):
    qml.AmplitudeEmbedding(features=x, wires=num_qubits, normalize=True, pad_with=0.)

def statepreparation_time(x, T_max, target_qubit):
    x_ang = x / T_max *np.pi*2
    qml.AngleEmbedding(features=x_ang, wires=[target_qubit], rotation="X")

class denoise_qc_model(torch.nn.Module):
    def __init__(self, ql_depth, num_qubits, dev, device ='cpu', layer="SIMPLE", init_weight=pi, T_max=1000) -> None:
        super().__init__()

        self.depth = ql_depth
        self.num_qubits = num_qubits
        self.dev = dev

        self.tran = nn.Linear(pow(2, num_qubits+1), pow(2,num_qubits))
        self.tran = self.tran.to(device)

        self.T_max = T_max
        self.device = device
        self.latent_qubits = [i for i in range(num_qubits)]
        if layer=="SIMPLE":
            self.initial_weights = [init_weight for _ in range(num_qubits+1)]
            weight_size = (ql_depth, num_qubits, 2)
        elif layer=="STRONG":
            weight_size = (ql_depth, num_qubits+1, 3)


        def circuit(weights, inputs):   # initial weights only for simplified two design
            print("repeat!")
            print("weights size:", weights.shape)
            
            latent = inputs[:,:-1]
            t = inputs[:,-1].reshape((-1,1))
            # print("t shape:", t.shape)
            # print("latent shape:", latent.shape)
            statepreparation(latent, self.latent_qubits)
            statepreparation_time(t, self.T_max, num_qubits)

            if layer=="SIMPLE":
                 qml.SimplifiedTwoDesign(self.initial_weights, weights, wires=range(num_qubits+1))
            elif layer=="STRONG":
                qml.StronglyEntanglingLayers(weights, wires=range(num_qubits+1))

            # temp = [qml.expval(qml.PauliZ(q)) for q in range(1, num_qubits+1)]
            # temp = qml.probs(wires=list(range(num_qubits+1)))
            temp = qml.probs(wires=self.latent_qubits)
            return temp
        
        self.circuit = qml.QNode(circuit, self.dev, interface="torch", diff_method="backprop") #"parameter-shift")

        self.linear = qml.qnn.TorchLayer(self.circuit, {"weights":weight_size})#, "t":1})

    def set_train(self, train=False):
        if train==False:
            for param in self.tran.parameters():
                param.requires_grad = False
        else:
            for param in self.tran.parameters():
                param.requires_grad = True

    def forward(self, inputs): #, t):
        # input_dict = {"inputs":inputs,"t":t}
        # print(inputs.shape)
        print("forward!")
        out1 = self.linear(inputs)
        
        # out1 = self.tran(out1)
        return out1

        # probsgiven0 = out1[:,:(2 ** (len(self.latent_qubits)))]
        # probsgiven0 /= torch.sum(probsgiven0,dim=-1, keepdim=True)

        # # Post-Processing
        # probsgiven = probsgiven0 / torch.max(probsgiven0)
        
        # # return torch.sqrt(probsgiven)
        # return probsgiven

    def save(self, save_path):
        torch.save(self.state_dict(), save_path)

    def load(self, load_path):
        ckpt = torch.load(load_path)
        self.load_state_dict(ckpt)



# %%
num_qubits = n_qubit
num_layer = 32   # 16
layer_type="SIMPLE"
T_max = 10
optim_name = "Adam"
learning_rate = 5e-3 #5e-3

device = "cpu"
# device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
print(device)
# q_device = qml.device("lightning.gpu", wires=num_qubits+1, batch_obs=True)        
# q_device = qml.device("lightning.qubit", wires=num_qubits+1, batch_obs=True)        
q_device = qml.device("default.qubit.torch", wires=num_qubits+1) #, batch_obs=True)        
# qc = denoise_qc_modelv2(num_layer, num_qubits, dev=q_device, layer=layer_type, T_max=T_max).to(device)
# qc = denoise_qc_model(num_layer, num_qubits, dev=q_device, layer=layer_type, T_max=T_max).to(device)
qc = denoise_qc_model(num_layer, num_qubits, dev=q_device, layer=layer_type, T_max=T_max).to(device)

qsc_device = qml.device("default.qubit.torch", wires=num_qubits) #, batch_obs=True)        
qsc_model = QSC(n=num_qubits, dev=qsc_device)
# temp_input = torch.randn((10,4))
# temp_input += abs(temp_input.min()) + 0.1
diff_hs = np.linspace(0.5, 4., T_max)  #10 should be replaced to T_max

if optim_name == "Adam":
    opt = torch.optim.Adam(qc.parameters(), lr=learning_rate)
if optim_name == "AdamW":
    opt = torch.optim.Adam(qc.parameters(), lr=learning_rate)
elif optim_name == "SGD":
    opt = torch.optim.SGD(qc.parameters(), lr=learning_rate, momentum=0.9)
elif optim_name == "RMSProp":
    opt = torch.optim.RMSprop(qc.parameters(), lr=learning_rate)

scheduler = ReduceLROnPlateau(opt)#, gamma=0.9)

# %%
# qsc_device = qml.device("default.qubit.torch", wires=num_qubits) #, batch_obs=True)        
# qsc_model = QSC(n=num_qubits, dev=qsc_device)

# %% [markdown]
# ### Utility Functions

# %%

__conditioning_keys__ = {'concat': 'c_concat',
                         'crossattn': 'c_crossattn', 
                         'adm': 'y'}


def make_ddim_timesteps(ddim_discr_method, num_ddim_timesteps, num_ddpm_timesteps, verbose=True):
    if ddim_discr_method == 'uniform':
        c = num_ddpm_timesteps // num_ddim_timesteps
        ddim_timesteps = np.asarray(list(range(0, num_ddpm_timesteps, c)))
    elif ddim_discr_method == 'quad':
        ddim_timesteps = ((np.linspace(0, np.sqrt(num_ddpm_timesteps * .8), num_ddim_timesteps)) ** 2).astype(int)
    else:
        raise NotImplementedError(f'There is no ddim discretization method called "{ddim_discr_method}"')

    # assert ddim_timesteps.shape[0] == num_ddim_timesteps
    # add one to get the final alpha values right (the ones from first scale to data during sampling)
    steps_out = ddim_timesteps + 1
    if verbose:
        print(f'Selected timesteps for ddim sampler: {steps_out}')
    return steps_out


def make_ddim_sampling_parameters(alphacums, ddim_timesteps, eta, verbose=True):
    # select alphas for computing the variance schedule
    alphas = alphacums[ddim_timesteps]
    alphas_prev = np.asarray([alphacums[0]] + alphacums[ddim_timesteps[:-1]].tolist())

    # according the the formula provided in https://arxiv.org/abs/2010.02502
    sigmas = eta * np.sqrt((1 - alphas_prev) / (1 - alphas) * (1 - alphas / alphas_prev))
    if verbose:
        print(f'Selected alphas for ddim sampler: a_t: {alphas}; a_(t-1): {alphas_prev}')
        print(f'For the chosen value of eta, which is {eta}, '
              f'this results in the following sigma_t schedule for ddim sampler {sigmas}')
    return sigmas, alphas, alphas_prev


def extract_into_tensor(a, t, x_shape):
    b, *_ = t.shape
    # print(t)
    a = a.to(device)
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))

def make_beta_schedule(schedule, n_timestep, linear_start=1e-4, linear_end=2e-2, cosine_s=8e-3):
    if schedule == "linear":
        betas = (
                torch.linspace(linear_start ** 0.5, linear_end ** 0.5, n_timestep, dtype=torch.float64) ** 2
        )

    elif schedule == "cosine":
        timesteps = (
                torch.arange(n_timestep + 1, dtype=torch.float64) / n_timestep + cosine_s
        )
        alphas = timesteps / (1 + cosine_s) * np.pi / 2
        alphas = torch.cos(alphas).pow(2)
        alphas = alphas / alphas[0]
        betas = 1 - alphas[1:] / alphas[:-1]
        betas = np.clip(betas, a_min=0, a_max=0.999)

    elif schedule == "sqrt_linear":
        betas = torch.linspace(linear_start, linear_end, n_timestep, dtype=torch.float64)
    elif schedule == "sqrt":
        betas = torch.linspace(linear_start, linear_end, n_timestep, dtype=torch.float64) ** 0.5
    else:
        raise ValueError(f"schedule '{schedule}' unknown.")
    return betas.numpy()


def count_params(model, verbose=False):
    total_params = sum(p.numel() for p in model.parameters())
    if verbose:
        print(f"{model.__class__.__name__} has {total_params}  params.")
    return total_params

def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


def noise_like(shape, device, repeat=False):
    repeat_noise = lambda: torch.randn((1, *shape[1:]), device=device).repeat(shape[0], *((1,) * (len(shape) - 1)))
    noise = lambda: torch.randn(shape, device=device)
    return repeat_noise() if repeat else noise()

# %%

class LatentDiffusion(nn.Module):
    """main class"""
    def __init__(self,
                 model,
                 qsc_model,
                 T_max,
                 device, 
                 training,
                 loss_type = "l2",
                 ):
        super().__init__()
        self.model = model
        self.qsc_model = qsc_model
        self.diff_hs = np.linspace(0.5, 4, T_max)
        self.num_timesteps = T_max
        self.device = device
        self.training = training
        self.loss_type = loss_type
        if self.loss_type == "kld" or self.loss_type == "kld_recon":
            self.criterion = nn.KLDivLoss(reduction="batchmean", log_target=True)


    def apply_model(self, x_noisy, t, return_ids=False):
        concat_x_t = torch.hstack((x_noisy,torch.unsqueeze(t,-1)))
        x_recon = self.model(concat_x_t) #, **cond)
        if isinstance(x_recon, tuple) and not return_ids:
            return x_recon[0]
        else:
            return x_recon

    def get_loss(self, pred, target, mean=True):
        if self.loss_type == 'l1':
            loss = (target - pred).abs()
            if mean:
                loss = loss.mean()
        elif self.loss_type == 'l2':
            if mean:
                loss = torch.nn.functional.mse_loss(target, pred)
            else:
                loss = torch.nn.functional.mse_loss(target, pred, reduction='none')
        elif self.loss_type == "kld":
            assert not (torch.isnan(target.log())).all()
            # return 0
            loss = self.criterion(pred.log(), target.log())
        elif self.loss_type == "kld_recon":
            assert not (torch.isnan(target.log())).all()
            loss1 = self.criterion(pred.log(), target.log())
            loss2 = torch.nn.functional.mse_loss(target, pred)*10
            loss = loss1 + loss2
        else:
            raise NotImplementedError("unknown loss type '{loss_type}'")

        return loss
    
    def forward(self, x): #, c):#, *args, **kwargs):
        # generate time with given batch size 
        t = torch.randint(0, self.num_timesteps, (x.shape[0],), device=self.device).long()
        # print(t.shape)
        # print(concat_x_t.shape)
        return self.p_losses(x, t)#, *args, **kwargs)

    def gen_noisy(self, x_start, tmax=None):
        x_noisy = torch.zeros_like(x_start)
        x_shape = x_start.shape[0]
        
        if tmax is None or tmax == self.num_timesteps:
            tmax = self.num_timesteps
            diff_hs = self.diff_hs
        else:
            diff_hs = np.linspace(0.5, 4, tmax)

        with torch.no_grad():
            for tdx in range(x_shape):
                x_noisy[tdx,:] = self.qsc_model.step_diffusion(tmax, x_start[tdx,:], diff_hs[:tmax], tmax)

        return x_noisy
    
    def from_noisy(self, x_start):
        x_noisy = torch.zeros_like(x_start)
        x_shape = x_start.shape[0]
        t = torch.tensor([self.num_timesteps]).repeat(x_shape)
        with torch.no_grad():
            for tdx in range(x_shape):
                x_noisy[tdx,:] = self.qsc_model.step_diffusion(self.num_timesteps, x_start[tdx,:], self.diff_hs[:self.num_timesteps], self.num_timesteps)
        model_output = self.apply_model(x_noisy, t)
        return model_output
    
    def generate_sample(self, x_start, target_t):
        x_noisy = torch.zeros_like(x_start)
        x_shape = x_start.shape[0]
        t = torch.tensor([target_t]).repeat(x_shape)
        with torch.no_grad():
            for tdx in range(x_shape):
                x_noisy[tdx,:] = self.qsc_model.step_diffusion(target_t, x_start[tdx,:], self.diff_hs[:target_t], target_t)
        model_output = self.apply_model(x_noisy, t)
        return model_output

    def p_losses(self, x_start, t, noise=None):
        assert(x_start >0).all()
        # x_start = torch.sqrt(x_start)
        # print(x_start[0,:].shape)
        x_noisy = torch.zeros_like(x_start)
        with torch.no_grad():
            for tdx, t_val in enumerate(t):
                x_noisy[tdx,:] = self.qsc_model.step_diffusion(t_val, x_start[tdx,:], self.diff_hs[:t_val], t_val)

        # print(x_noisy.shape)

        model_output = self.apply_model(x_noisy, t)
        

        loss_dict = {}
        prefix = 'train' if self.training else 'val'

        target = x_start

        if self.loss_type == "kld" or self.loss_type == "kld_recon":
            loss = self.get_loss(model_output, target, mean=False)#.mean(dim=(1)) #.mean([1, 2, 3])
        else:
            loss = self.get_loss(model_output, target, mean=False).mean(dim=(1)) #.mean([1, 2, 3])

        loss_dict.update({f'{prefix}/loss_simple': loss.mean()})

        loss = loss.mean()

        return loss, loss_dict



# %%
stab_diff_test = LatentDiffusion(model=qc, qsc_model=qsc_model, T_max=T_max, device="cpu", training=True)

# %% [markdown]
# #### Train quantum circuit for denoising

# %%
# number of epochs to train the model
n_epochs = 20

# DEVICE = "cpu"
model.to(device)
model.eval()

loss_track = []
for epoch in range(1, n_epochs+1):
    # monitor training loss
    train_loss = 0.0
    
    ###################
    # train the model #
    ###################
    batch_loss = 0.0
    for bdx, data in enumerate(train_loader, 0):
        # _ stands in for labels, here
        # no need to flatten images
        images, _ = data
        images = images.to(device)
        # clear the gradients of all optimized variables
        opt.zero_grad()
        # forward pass: compute predicted outputs by passing inputs to the model
        latent = model.encode(images).to(device)
        # calculate the loss
        loss, loss_dict = stab_diff_test.forward(latent)#, None)

        # backward pass: compute gradient of the loss with respect to model parameters
        loss.backward()
        # perform a single optimization step (parameter update)
        opt.step()
        batch_loss += loss.item()
        print(f"bach loss batch:{bdx}, loss: {loss.item()}")
        # break
        # update running training loss
    epoch_loss = batch_loss / len(train_loader)
    loss_track.append(epoch_loss)
    scheduler.step(epoch_loss)
    print(f"Epoch {epoch} Loss:", batch_loss/len(train_loader))
        
        
print("epoch loss:", loss_track)
# %%
# random_data = qsc_model.GenerateHaarSample(10, 10)

# %%

resize_transform = transforms.Compose([
    transforms.Resize((28, 28)), # Convert to 28x28 again
])
# obtain one batch of test images
images, labels = next(iter(test_loader))
# images, labels = dataiter.next()

# get sample outputs
# forward pass: compute predicted outputs by passing inputs to the model
gt_latent = model.encode(images).to(device)
# prep images for display
one_shot_gen_latent = stab_diff_test.from_noisy(gt_latent)

noisy_latent = stab_diff_test.gen_noisy(gt_latent, tmax=T_max)

gen_latent = gt_latent
for idx in range(T_max, 0, -1):
    gen_latent = stab_diff_test.generate_sample(gen_latent, target_t = idx)


output = model.decode(gt_latent)
gen_output = model.decode(one_shot_gen_latent)
noisy_output = model.decode(noisy_latent)
prog_gen_output = model.decode(gen_latent)

ground_truth_images = torch.stack([resize_transform(image) for image in images]) 
ae_gen_image = torch.stack([resize_transform(image) for image in output])
generated_images = torch.stack([resize_transform(image) for image in gen_output])
nosiy_gen_images = torch.stack([resize_transform(image) for image in noisy_output])
prog_gen_images = torch.stack([resize_transform(image) for image in prog_gen_output])
# output is resized into a batch of iages
print(generated_images.shape)
# output = output.view(batch_size, 1, 28, 28)
# use detach when it's an output that requires_grad
ground_truth_images = ground_truth_images.numpy()
output1 =  ae_gen_image.cpu().detach().numpy()
output2 = generated_images.cpu().detach().numpy()
output3 = prog_gen_images.cpu().detach().numpy()
output4 = nosiy_gen_images.cpu().detach().numpy()

# plot the first ten input images and then reconstructed images
fig, axes = plt.subplots(nrows=5, ncols=20, sharex=True, sharey=True, figsize=(25,4))

# input images on top row, reconstructions on bottom
for gt, row in zip([ground_truth_images, output1, output2, output3, output4], axes):
    for img, ax in zip(gt, row):
        ax.imshow(np.squeeze(img), cmap='gray')
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

import datetime
now = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
plt.savefig(f"results/{now}_{num_layer}_{T_max}_{num_qubits}_l2_progressive.png",format="png")
