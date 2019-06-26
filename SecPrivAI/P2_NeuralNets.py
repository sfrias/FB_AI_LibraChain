#!/usr/bin/env python
# coding: utf-8

# # Neural networks with PyTorch
# 
# Deep learning networks tend to be massive with dozens or hundreds of layers, that's where the term "deep" comes from. You can build one of these deep networks using only weight matrices as we did in the previous notebook, but in general it's very cumbersome and difficult to implement. PyTorch has a nice module `nn` that provides a nice way to efficiently build large neural networks.

# In[1]:


# Import necessary packages
import math
import numpy as np  # numpy numeric package

import matplotlib.pyplot as plt # Plot utilities lib

import helper # helper.py functions


# In[2]:


import torch  # torch package

from torchvision import datasets, transforms  # Import torchvision utilities


# In[3]:


torch.__version__


# In[4]:


from IPython.core.display import display, HTML
display(HTML("<style>.container { width:100% !important; }</style>")) # Width 100% of notebook.


# In[5]:


torch.set_printoptions(linewidth=155, precision=4) # Display options to tensor values


# In[6]:


np.set_printoptions(linewidth= 155, precision=3)


# In[7]:


get_ipython().run_line_magic('matplotlib', 'inline # When it is called once, all figures in the notebook will be inline')


# In[8]:


get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'retina' # Definition of displayed plot is a bit better: retina quality")


# Sigmoid Function
# $$
# \Large S (x) =  \frac{\mathrm{1} }{\mathrm{1} + e^{- x }}
# $$ 

# In[9]:


# Activation function for torch.nn
def sigmoid_tch(x):
    return 1/(1 + torch.exp(-x))



def sigmoid_np(x):
    return 1/(1 + np.exp(-x))

# Example of sigmoid function squeeze values
rango = range(-60,61,9)
for gr in rango:
    print(gr/10,'\t->',sigmoid_np(gr/10))

t1 = np.arange(-60, 61, 9)
plt.plot(t1/10, 1/(1+ np.exp(-t1/10)), '-')
print(rango)
print(t1/10)
print(sigmoid_np(t1/10))


# Softmax function
# $$
# \Large \sigma(x_i) = \cfrac{e^{x_i}}{\sum_k^K{e^{x_k}}}
# $$

# In[10]:


z = [1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0]
print('z    = ', z)
z_exp = [math.exp(i) for i in z]
print('z_exp=',[round(i, 1) for i in z_exp])
#[2.72, 7.39, 20.09, 54.6, 2.72, 7.39, 20.09]
sum_z_exp = sum(z_exp)
print('sum_z_exp=',round(sum_z_exp, 2))
#114.98
softmax = [round(i / sum_z_exp, 3) for i in z_exp]
print('softmax = z_exp/sum_z_exp =',softmax)
#[0.024, 0.064, 0.175, 0.475, 0.024, 0.064, 0.175]


# 
# Now we're going to build a larger network that can solve a (formerly) difficult problem, identifying text in an image. Here we'll use the MNIST dataset which consists of greyscale handwritten digits. Each image is 28x28 pixels, you can see a sample below
# 
# <img src='assets/mnist.png'>
# 
# <H3>Our goal is to build a neural network that can take one of these images and predict the digit in the image</H3>

# <h2>
# <font color="#3333FF">FILE FORMATS FOR THE MNIST DATABASE</font></h2>
# The data is stored in a very simple file format designed for storing vectors
# and multidimensional matrices. General info on this format is given at
# the end of this page, but you don't need to read that to use the data files.
# 
# <p>All the integers in the files are stored in the MSB first (high endian)
# format used by most non-Intel processors. Users of Intel processors and
# other low-endian machines must flip the bytes of the header.
# 
# </p><p>There are 4 files:<tt></tt>
# 
# </p><p><tt>train-images-idx3-ubyte: training set images</tt>
# <br><tt>train-labels-idx1-ubyte: training set labels</tt>
# <br><tt>t10k-images-idx3-ubyte:&nbsp; test set images</tt>
# <br><tt>t10k-labels-idx1-ubyte:&nbsp; test set labels</tt>
# 
# </p><p>The training set contains 60000 examples, and the test set 10000 examples.
# 
# </p><p>The first 5000 examples of the test set are taken from the original
# NIST training set. The last 5000 are taken from the original NIST test
# set. The first 5000 are cleaner and easier than the last 5000.
# </p><h3>
# TRAINING SET LABEL FILE (train-labels-idx1-ubyte):</h3>
# <tt>[offset] [type]&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
# [value]&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; [description]</tt>
# <br><tt>0000&nbsp;&nbsp;&nbsp;&nbsp; 32 bit integer&nbsp; 0x00000801(2049)
# magic number (MSB first)</tt>
# <br><tt>0004&nbsp;&nbsp;&nbsp;&nbsp; 32 bit integer&nbsp; 60000&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
# number of items</tt>
# <br><tt>0008&nbsp;&nbsp;&nbsp;&nbsp; unsigned byte&nbsp;&nbsp; ??&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
# label</tt>
# <br><tt>0009&nbsp;&nbsp;&nbsp;&nbsp; unsigned byte&nbsp;&nbsp; ??&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
# label</tt>
# <br><tt>........</tt>
# <br><tt>xxxx&nbsp;&nbsp;&nbsp;&nbsp; unsigned byte&nbsp;&nbsp; ??&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
# label</tt>
# 
# <p><tt>The labels values are 0 to 9.</tt>
# </p><h3>
# TRAINING SET IMAGE FILE (train-images-idx3-ubyte):</h3>
# <tt>[offset] [type]&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
# [value]&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; [description]</tt>
# <br><tt>0000&nbsp;&nbsp;&nbsp;&nbsp; 32 bit integer&nbsp; 0x00000803(2051)
# magic number</tt>
# <br><tt>0004&nbsp;&nbsp;&nbsp;&nbsp; 32 bit integer&nbsp; 60000&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
# number of images</tt>
# <br><tt>0008&nbsp;&nbsp;&nbsp;&nbsp; 32 bit integer&nbsp; 28&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
# number of rows</tt>
# <br><tt>0012&nbsp;&nbsp;&nbsp;&nbsp; 32 bit integer&nbsp; 28&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
# number of columns</tt>
# <br><tt>0016&nbsp;&nbsp;&nbsp;&nbsp; unsigned byte&nbsp;&nbsp; ??&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
# pixel</tt>
# <br><tt>0017&nbsp;&nbsp;&nbsp;&nbsp; unsigned byte&nbsp;&nbsp; ??&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
# pixel</tt>
# <br><tt>........</tt>
# <br><tt>xxxx&nbsp;&nbsp;&nbsp;&nbsp; unsigned byte&nbsp;&nbsp; ??&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
# pixel</tt>
# 
# <p>Pixels are organized row-wise. Pixel values are 0 to 255. 0 means background
# (white), 255 means foreground (black).
# </p><h3>
# TEST SET LABEL FILE (t10k-labels-idx1-ubyte):</h3>
# <tt>[offset] [type]&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
# [value]&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; [description]</tt>
# <br><tt>0000&nbsp;&nbsp;&nbsp;&nbsp; 32 bit integer&nbsp; 0x00000801(2049)
# magic number (MSB first)</tt>
# <br><tt>0004&nbsp;&nbsp;&nbsp;&nbsp; 32 bit integer&nbsp; 10000&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
# number of items</tt>
# <br><tt>0008&nbsp;&nbsp;&nbsp;&nbsp; unsigned byte&nbsp;&nbsp; ??&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
# label</tt>
# <br><tt>0009&nbsp;&nbsp;&nbsp;&nbsp; unsigned byte&nbsp;&nbsp; ??&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
# label</tt>
# <br><tt>........</tt>
# <br><tt>xxxx&nbsp;&nbsp;&nbsp;&nbsp; unsigned byte&nbsp;&nbsp; ??&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
# label</tt>
# 
# <p>The labels values are 0 to 9.
# </p><h3>
# TEST SET IMAGE FILE (t10k-images-idx3-ubyte):</h3>
# <tt>[offset] [type]&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
# [value]&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; [description]</tt>
# <br><tt>0000&nbsp;&nbsp;&nbsp;&nbsp; 32 bit integer&nbsp; 0x00000803(2051)
# magic number</tt>
# <br><tt>0004&nbsp;&nbsp;&nbsp;&nbsp; 32 bit integer&nbsp; 10000&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
# number of images</tt>
# <br><tt>0008&nbsp;&nbsp;&nbsp;&nbsp; 32 bit integer&nbsp; 28&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
# number of rows</tt>
# <br><tt>0012&nbsp;&nbsp;&nbsp;&nbsp; 32 bit integer&nbsp; 28&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
# number of columns</tt>
# <br><tt>0016&nbsp;&nbsp;&nbsp;&nbsp; unsigned byte&nbsp;&nbsp; ??&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
# pixel</tt>
# <br><tt>0017&nbsp;&nbsp;&nbsp;&nbsp; unsigned byte&nbsp;&nbsp; ??&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
# pixel</tt>
# <br><tt>........</tt>
# <br><tt>xxxx&nbsp;&nbsp;&nbsp;&nbsp; unsigned byte&nbsp;&nbsp; ??&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
# pixel</tt>
# 
# <p>Pixels are organized row-wise. Pixel values are 0 to 255. 0 means background
# (white), 255 means foreground (black).
# <br>&nbsp;
# <br>
# </p><hr width="100%">
# <h2>
# <font color="#3333FF">THE IDX FILE FORMAT</font></h2>
# the IDX file format is a simple format for vectors and multidimensional
# matrices of various numerical types.
# 
# <p>The basic format is
# 
# </p><p><tt>magic number</tt>
# <br><tt>size in dimension 0</tt>
# <br><tt>size in dimension 1</tt>
# <br><tt>size in dimension 2</tt>
# <br><tt>.....</tt>
# <br><tt>size in dimension N</tt>
# <br><tt>data</tt>
# 
# </p><p>The magic number is an integer (MSB first). The first 2 bytes are always
# 0.
# 
# </p><p>The third byte codes the type of the data:
# <br>0x08: unsigned byte
# <br>0x09: signed byte
# <br>0x0B: short (2 bytes)
# <br>0x0C: int (4 bytes)
# <br>0x0D: float (4 bytes)
# <br>0x0E: double (8 bytes)
# 
# </p><p>The 4-th byte codes the number of dimensions of the vector/matrix: 1
# for vectors, 2 for matrices....
# 
# </p><p>The sizes in each dimension are 4-byte integers (MSB first, high endian,
# like in most non-Intel processors).
# 
# </p><p>The data is stored like in a C array, i.e. the index in the last dimension
# changes the fastest.
# </p><p>

# First up, we need to get our dataset. This is provided through the `torchvision` package. The code below will download the MNIST dataset, then create training and test datasets for us. Don't worry too much about the details here, you'll learn more about this later.

# In[11]:


### Run this cell to import torchvision utilities. Previous call at start.
#from torchvision import datasets, transforms


# The dataset is solely responsible for the abstraction of the data.
# Only one piece of information or sample is returned at any time.
# 
# In Pytorch, the dataset is abstracted in the torch.utils.data.Dataset class, which should be inherent to all datasets, and overwrite __len__ (Len (obj) is equivalent to __len__ ()) that returns the number of samples, __getitem__ (Obj [index] is equivalent to obj .__ getitem__) that returns a sample. This function will be called in parallel when we have multi-process acceleration. That is why the dataset should only include read-only objects.

# In[12]:


# Define a transform to normalize the data (probability),(distribution)
transform = transforms.Compose([transforms.ToTensor(),
                              transforms.Normalize((0.5,), (0.5,)),
                              ])


#  Las transformaciones de imágenes comunes se pueden encadenar con torchvision.transforms.Compose, desde una lista de transformaciones.<hr>
#  Una de las mas comunes es la normalización, donde se pasa una lista de medias y de desviaciones estándar, donde el canal de entrada se asocia al valor del canal menos la media, todo ello dividido por la desvíacion estándar de los valores del canal.<hr>
#  Restando la media, centramos los datos alrededor de cero, y dividiendo por la desviacion estandar aplastamos los valores para que estén entre -1 y +1. Esta operación de normalización nos ayuda a mantener los pesos de trabajo en la red cerca del cero, manteniendo la propagación hacia atrás mas estable. Sin normalización las redes tienden a fallar en  el proceso de aprendizaje.</pre>

# In[13]:


transform


# In[14]:


# Download the training data. Carga dataset en clase abstracta, descargándolos, generando carpeta de train,
# aplicando transformación.
trainset = datasets.MNIST('~/.pytorch/MNIST_data/', download=True, train=True, transform=transform)


# In[15]:


trainset


# ### Section kernel commands

# ### Trainloader

# In[16]:


# load the training data. Usa lotes de 64 imágenes
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)


# ### Section kernel commands

# We have the training data loaded into `trainloader` and we make that an iterator with `iter(trainloader)`. Later, we'll use this to loop through the dataset for training, like
# 
# ```python
# for image, label in trainloader:
#     ## do things with images and labels
# ```
# 
# You'll notice I created the `trainloader` with a batch size of 64, and `shuffle=True`. The batch size is the number of images we get in one iteration from the data loader and pass through our network, often called a *batch*. And `shuffle=True` tells it to shuffle the dataset every time we start going through the data loader again. But here I'm just grabbing the first batch so we can check out the data. We can see below that `images` is just a tensor with size `(64, 1, 28, 28)`. So, 64 images per batch, 1 color channel, and 28x28 images.

# In[17]:


dataiter = iter(trainloader)


# In[18]:


images, labels = dataiter.next()


# In[19]:


print(type(images))
print(images.shape)
print(labels.shape)
print(labels)
print((images[0,0,:,0:24]).int())


# In[20]:


print(images.shape)


# In[21]:


print(type(images))
print(images.shape)
a = images[0].numpy().squeeze()
print(a.shape, a.dtype)
a[:,7:21]


# This is whatthe images looks like.
# <br>First Images and last images.

# In[22]:


plt.imshow(images[0].numpy().squeeze(), cmap='Greys_r');


# In[23]:


plt.imshow(images[1].numpy().squeeze(), cmap='Greys_r');


# In[24]:


plt.imshow(images[2].numpy().squeeze(), cmap='Greys_r');


# In[25]:


plt.imshow(images[3].numpy().squeeze(), cmap='Greys_r');


# In[26]:


plt.imshow(images[4].numpy().squeeze(), cmap='Greys_r');


# In[27]:


plt.imshow(images[5].numpy().squeeze(), cmap='Greys_r');


# In[28]:


plt.imshow(images[6].numpy().squeeze(), cmap='Greys_r');


# In[29]:


plt.imshow(images[7].numpy().squeeze(), cmap='Greys_r');


# In[30]:


plt.imshow(images[8].numpy().squeeze(), cmap='Greys_r');


# In[31]:


plt.imshow(images[9].numpy().squeeze(), cmap='Greys_r');


# Last Images

# In[32]:


plt.imshow(images[61].numpy().squeeze(), cmap='Greys_r');


# In[33]:


plt.imshow(images[62].numpy().squeeze(), cmap='Greys_r');


# In[34]:


plt.imshow(images[63].numpy().squeeze(), cmap='Greys_r');


# In[35]:


# Error if Batch has 64 images: from 0 to 63. 64 is out of bounds.
plt.imshow(images[64].numpy().squeeze(), cmap='Greys_r');


# First, let's try to build a simple network for this dataset using weight matrices and matrix multiplications. Then, we'll see how to do it using PyTorch's `nn` module which provides a much more convenient and powerful method for defining network architectures.
# 
# The networks you've seen so far are called *fully-connected* or *dense* networks. Each unit in one layer is connected to each unit in the next layer. In fully-connected networks, the input to each layer must be a one-dimensional vector (which can be stacked into a 2D tensor as a batch of multiple examples). However, our images are 28x28 2D tensors, so we need to convert them into 1D vectors. Thinking about sizes, we need to convert the batch of images with shape `(64, 1, 28, 28)` to a have a shape of `(64, 784)`, 784 is 28 times 28. This is typically called *flattening*, we flattened the 2D images into 1D vectors.
# 
# Previously you built a network with one output unit. Here we need 10 output units, one for each digit. We want our network to predict the digit shown in an image, so what we'll do is calculate probabilities that the image is of any one digit or class. This ends up being a discrete probability distribution over the classes (digits) that tells us the most likely class for the image. That means we need 10 output units for the 10 classes (digits). We'll see how to convert the network output into a probability distribution next.
# 
# > **Exercise:** Flatten the batch of images `images`. Then build a multi-layer network with 784 input units, 256 hidden units, and 10 output units using random tensors for the weights and biases. For now, use a sigmoid activation for the hidden layer. Leave the output layer without an activation, we'll add one that gives us a probability distribution next.

# In[36]:


print(images.shape)
# Flat images from 2D(64,1,28,28) to 1D(64,784)
pixels = images.view(images.shape[0],-1)
print(pixels.shape)


# In[37]:


# Build a multi-layer network with 784 input units, 256 hidden units,
# and 10 output units using random tensors for the weights and biases.

# Input image(L0) is a matrix of 1 row, 784 pixel columns

# (L1)Perceptron weight matrix parameters, 784 pixel rows from L0, for each 256 hidden units columns
W1 = torch.randn(784, 256) 
print(W1.shape)
W1


# In[38]:


# (L1)Perceptron bias matrix,256 values, one each L1 units
B1 = torch.randn(256)
print(B1.shape)
B1


# In[39]:


# (L2)Perceptron weight matrix parameters, 256 rows:outputs of L1 layer, , for each 10 columns: output units
W2 = torch.randn(256, 10)
print(W2.shape)
W2


# In[40]:


# (L2)Perceptron bias matrix,10 values, one each L2 units
B2 = torch.randn(10)
print(B2.shape)
B2


# In[41]:


# Use a sigmoid activation for the L1 hidden layer.
h1 = sigmoid_tch(torch.mm(pixels, W1) + B1)
print(h1.shape)
h1


# In[42]:


# L2 output layer without an activation
h2 = torch.mm(h1, W2) + B2
print(h2.shape)
print(h2)


# Now we have 10 outputs for our network. We want to pass in an image to our network and get out a probability distribution over the classes that tells us the likely class(es) the image belongs to. Something that looks like this:
# <img src='assets/image_distribution.png' width=500px>
# 
# Here we see that the probability for each class is roughly the same. This is representing an untrained network, it hasn't seen any data yet so it just returns a uniform distribution with equal probabilities for each class.
# 
# To calculate this probability distribution, we often use the [**softmax** function](https://en.wikipedia.org/wiki/Softmax_function). Mathematically this looks like
# 
# $$
# \Large \sigma(x_i) = \cfrac{e^{x_i}}{\sum_k^K{e^{x_k}}}
# $$
# 
# What this does is squish each input $x_i$ between 0 and 1 and normalizes the values to give you a proper probability distribution where the probabilites sum up to one.
# 
# > **Exercise:** Implement a function `softmax` that performs the softmax calculation and returns probability distributions for each example in the batch. Note that you'll need to pay attention to the shapes when doing this. If you have a tensor `a` with shape `(64, 10)` and a tensor `b` with shape `(64,)`, doing `a/b` will give you an error because PyTorch will try to do the division across the columns (called broadcasting) but you'll get a size mismatch. The way to think about this is for each of the 64 examples, you only want to divide by one value, the sum in the denominator. So you need `b` to have a shape of `(64, 1)`. This way PyTorch will divide the 10 values in each row of `a` by the one value in each row of `b`. Pay attention to how you take the sum as well. You'll need to define the `dim` keyword in `torch.sum`. Setting `dim=0` takes the sum across the rows while `dim=1` takes the sum across the columns.

# In[43]:


def softmax(x):
    #print('************** SOFTMAX INITIAL ********************')
    z_exp = torch.exp(x) # Matrix with exponentials of values for x matrix values
    #print('***z_exp***', z_exp.shape)
    #print(z_exp)
    # Calculate sum of row values->dim =1, on z_exp matrix
    # Adjust from tensor of 64 elements to tensor 64 columns x 1 row ->.view(-1,1)
    sum_z_exp = torch.sum(z_exp, dim=1).view(-1,1) 
    #print('***sum_z_exp***',sum_z_exp.shape)
    #print(sum_z_exp)
    sftmx = z_exp / sum_z_exp
    #print('***softmax = z_exp/sum_z_exp***', sftmx.shape)
    #print(sftmx)
    #print('***Suma softmax***')
    #print(torch.sum(sftmx, dim=1))
    #print('**************** SOFTMAX END ***********************')
    return sftmx


# In[44]:


# Here, out should be the output of the network in the previous excercise with shape (64,10)
probabilities = softmax(h2)


# In[45]:


# Does it have the right shape? Should be (64, 10)
print('softmax',probabilities.shape)
#print(probabilities)


# In[46]:


# Does it sum to 1?
print(probabilities.sum(dim=1))


# ## Building networks with PyTorch
# 
# PyTorch provides a module `nn` that makes building networks much simpler. Here I'll show you how to build the same one as above with 784 inputs, 256 hidden units, 10 output units and a softmax output.

# In[47]:


from torch import nn


# In[48]:


class Network(nn.Module):
    def __init__(self):
        super().__init__()
        
        # Inputs to hidden layer linear transformation
        self.hidden = nn.Linear(784, 256)
        # Output layer, 10 units - one for each digit
        self.output = nn.Linear(256, 10)
        
        # Define sigmoid activation and softmax output 
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, x):
        # Pass the input tensor through each of our operations
        x = self.hidden(x)
        x = self.sigmoid(x)
        x = self.output(x)
        x = self.softmax(x)
        
        return x


# Let's go through this bit by bit.
# 
# ```python
# class Network(nn.Module):
# ```
# 
# Here we're inheriting from `nn.Module`. Combined with `super().__init__()` this creates a class that tracks the architecture and provides a lot of useful methods and attributes. It is mandatory to inherit from `nn.Module` when you're creating a class for your network. The name of the class itself can be anything.
# 
# ```python
# self.hidden = nn.Linear(784, 256)
# ```
# 
# This line creates a module for a linear transformation, $x\mathbf{W} + b$, with 784 inputs and 256 outputs and assigns it to `self.hidden`. The module automatically creates the weight and bias tensors which we'll use in the `forward` method. You can access the weight and bias tensors once the network (`net`) is created with `net.hidden.weight` and `net.hidden.bias`.
# 
# ```python
# self.output = nn.Linear(256, 10)
# ```
# 
# Similarly, this creates another linear transformation with 256 inputs and 10 outputs.
# 
# ```python
# self.sigmoid = nn.Sigmoid()
# self.softmax = nn.Softmax(dim=1)
# ```
# 
# Here I defined operations for the sigmoid activation and softmax output. Setting `dim=1` in `nn.Softmax(dim=1)` calculates softmax across the columns.
# 
# ```python
# def forward(self, x):
# ```
# 
# PyTorch networks created with `nn.Module` must have a `forward` method defined. It takes in a tensor `x` and passes it through the operations you defined in the `__init__` method.
# 
# ```python
# x = self.hidden(x)
# x = self.sigmoid(x)
# x = self.output(x)
# x = self.softmax(x)
# ```
# 
# Here the input tensor `x` is passed through each operation a reassigned to `x`. We can see that the input tensor goes through the hidden layer, then a sigmoid function, then the output layer, and finally the softmax function. It doesn't matter what you name the variables here, as long as the inputs and outputs of the operations match the network architecture you want to build. The order in which you define things in the `__init__` method doesn't matter, but you'll need to sequence the operations correctly in the `forward` method.
# 
# Now we can create a `Network` object.

# In[49]:


# Create the network and look at it's text representation
model = Network()
model


# In[63]:


model.forward


# You can define the network somewhat more concisely and clearly using the `torch.nn.functional` module. This is the most common way you'll see networks defined as many operations are simple element-wise functions. We normally import this module as `F`, `import torch.nn.functional as F`.

# In[50]:


import torch.nn.functional as F

class Network(nn.Module):
    def __init__(self):
        super().__init__()
        # Inputs to hidden layer linear transformation
        self.hidden = nn.Linear(784, 256)
        # Output layer, 10 units - one for each digit
        self.output = nn.Linear(256, 10)
        
    def forward(self, x):
        # Hidden layer with sigmoid activation
        x = F.sigmoid(self.hidden(x))
        # Output layer with softmax activation
        x = F.softmax(self.output(x), dim=1)
        
        return x


# ### Activation functions
# 
# So far we've only been looking at the softmax activation, but in general any function can be used as an activation function. The only requirement is that for a network to approximate a non-linear function, the activation functions must be non-linear. Here are a few more examples of common activation functions: Tanh (hyperbolic tangent), and ReLU (rectified linear unit).
# 
# <img src="assets/activation.png" width=700px>
# 
# In practice, the ReLU function is used almost exclusively as the activation function for hidden layers.

# ### Your Turn to Build a Network
# 
# <img src="assets/mlp_mnist.png" width=600px>
# 
# > **Exercise:** Create a network with 784 input units, a hidden layer with 128 units and a ReLU activation, then a hidden layer with 64 units and a ReLU activation, and finally an output layer with a softmax activation as shown above. You can use a ReLU activation with the `nn.ReLU` module or `F.relu` function.

# In[64]:


## Your nn.Module solution here
class DeepNetMd(nn.Module):
    def __init__(self):
        super().__init__()
        
        # Inputs to L1 linear transformation
        self.L1 = nn.Linear(784, 128)
        # L1 to L2 linear transformation
        self.L2 = nn.Linear(128, 64)
        # L2 to L3 linear transformation, 10 units - one for each digit
        self.L3 = nn.Linear(64, 10)
        
        # Define ReLu activation and softmax output 
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, x):
        # Pass the input tensor through each of our operations
        x = self.L1(x)
        x = self.relu(x)
        x = self.L2(x)
        x = self.relu(x)
        x = self.L3(x)
        x = self.softmax(x)
        
        return x


# In[65]:


# Create the network from nn.Module and look at it's text representation
model = DeepNetMd()
model


# In[66]:


model.forward


# In[70]:


import torch.nn.functional as F

class DeepNetFc(nn.Module):
    def __init__(self):
        super().__init__()
        
        # Inputs to L1 linear transformation
        self.L1 = nn.Linear(784, 128)
        # L1 to L2 linear transformation
        self.L2 = nn.Linear(128, 64)
        # L2 to L3 linear transformation, 10 units - one for each digit
        self.L3 = nn.Linear(64, 10)
        
    def forward(self, x):
        
        # Pass the input tensor through L1, L2 layer functions and operations
        x = F.ReLU(self.L1(x))
        x = F.ReLU(self.L2(x))
        # L3 layer with softmax activation
        x = F.softmax(self.output(x), dim=1)
        
        return x


# In[71]:


# Create the network from nn.functional and look at it's text representation
model = DeepNetFc()
model


# model.forward

# In[89]:


model = DeepNetMd()


# ### Initializing weights and biases
# 
# The weights and such are automatically initialized for you, but it's possible to customize how they are initialized. The weights and biases are tensors attached to the layer you defined, you can get them with `model.fc1.weight` for instance.

# In[90]:


print(model.L1.weight)
print(model.L1.bias)


# For custom initialization, we want to modify these tensors in place. These are actually autograd *Variables*, so we need to get back the actual tensors with `model.fc1.weight.data`. Once we have the tensors, we can fill them with zeros (for biases) or random normal values.

# In[91]:


model.L1.weight.data


# In[92]:


model.L1.bias.data


# In[93]:


# Set biases to all zeros
model.L1.bias.data.fill_(0)


# In[94]:


# sample from random normal with standard dev = 0.01
model.L1.weight.data.normal_(std=0.01)


# ### Forward pass
# 
# Now that we have a network, let's see what happens when we pass in an image.

# In[95]:


# Grab some data 
dataiter = iter(trainloader)
images, labels = dataiter.next()

# Resize images into a 1D vector, new shape is (batch size, color channels, image pixels) 
images.resize_(64, 1, 784)
# or images.resize_(images.shape[0], 1, 784) to automatically get batch size

# Forward pass through the network
img_idx = 0
ps = model.forward(images[img_idx,:])

img = images[img_idx]
helper.view_classify(img.view(1, 28, 28), ps)


# As you can see above, our network has basically no idea what this digit is. It's because we haven't trained it yet, all the weights are random!
# 
# ### Using `nn.Sequential`
# 
# PyTorch provides a convenient way to build networks like this where a tensor is passed sequentially through operations, `nn.Sequential` ([documentation](https://pytorch.org/docs/master/nn.html#torch.nn.Sequential)). Using this to build the equivalent network:

# In[96]:


# Hyperparameters for our network
input_size = 784
hidden_sizes = [128, 64]
output_size = 10

# Build a feed-forward network
model = nn.Sequential(nn.Linear(input_size, hidden_sizes[0]),
                      nn.ReLU(),
                      nn.Linear(hidden_sizes[0], hidden_sizes[1]),
                      nn.ReLU(),
                      nn.Linear(hidden_sizes[1], output_size),
                      nn.Softmax(dim=1))
print(model)

# Forward pass through the network and display output
images, labels = next(iter(trainloader))
images.resize_(images.shape[0], 1, 784)
ps = model.forward(images[0,:])
helper.view_classify(images[0].view(1, 28, 28), ps)


# Here our model is the same as before: 784 input units, a hidden layer with 128 units, ReLU activation, 64 unit hidden layer, another ReLU, then the output layer with 10 units, and the softmax output.
# 
# The operations are availble by passing in the appropriate index. For example, if you want to get first Linear operation and look at the weights, you'd use `model[0]`.

# In[97]:


print(model[0])
model[0].weight


# You can also pass in an `OrderedDict` to name the individual layers and operations, instead of using incremental integers. Note that dictionary keys must be unique, so _each operation must have a different name_.

# In[98]:


from collections import OrderedDict
model = nn.Sequential(OrderedDict([
                      ('fc1', nn.Linear(input_size, hidden_sizes[0])),
                      ('relu1', nn.ReLU()),
                      ('fc2', nn.Linear(hidden_sizes[0], hidden_sizes[1])),
                      ('relu2', nn.ReLU()),
                      ('output', nn.Linear(hidden_sizes[1], output_size)),
                      ('softmax', nn.Softmax(dim=1))]))
model


# Now you can access layers either by integer or the name

# In[99]:


print(model[0])
print(model.fc1)


# In the next notebook, we'll see how we can train a neural network to accuractly predict the numbers appearing in the MNIST images.
