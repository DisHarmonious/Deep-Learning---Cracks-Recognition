import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
import tensorflow as tf
from sklearn.utils import shuffle

### Read data
os.chdir("C:\\Users\\User\\Desktop\\python\\datasets\\cracks\\Negative")
all_files=os.listdir()
training_data=[]
img_pixel_x, img_pixel_y=128, 128
for x in range(1000):
    a=cv2.imread(all_files[x], cv2.IMREAD_GRAYSCALE)
    b=cv2.resize(a, (img_pixel_x,img_pixel_y))
    b=np.array(b)/255
    training_data.append([b, np.eye(2)[0]])
os.chdir("C:\\Users\\User\\Desktop\\python\\datasets\\cracks\\Positive")
all_files=os.listdir()
x=0
for x in range(1000):
    a=cv2.imread(all_files[x], cv2.IMREAD_GRAYSCALE)
    b=cv2.resize(a, (img_pixel_x,img_pixel_y))
    b=np.array(b)/255
    training_data.append([b, np.eye(2)[1]])

###shuffle
training_data = shuffle(training_data, random_state=0)

###build model
class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3)
        self.conv2 = nn.Conv2d(32, 128, 3)
        self.fc1 = nn.Linear(115200, 128) 
        self.fc2 = nn.Linear(128, 64) #nn.Linear / nn.Bilinear / nn.Identity
        self.fc3 = nn.Linear(64, 2)

    def forward(self, x):
        # Max pool (2, 2) 
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2)) #F.relu / torch.tanh / torch.sigmoid
        x = F.max_pool2d(F.relu(self.conv2(x)), 2) 
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

net = Net()

#epochs, optimizer and loss
learning_rate=0.001
optimizer=optim.Adam(net.parameters(), lr=learning_rate)
epochs=50
loss_function = nn.MSELoss()
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=learning_rate*.1)
batch_size=32

def batch_X(number):
    #batch_X to tensor
    a=torch.tensor(training_data[number][0])
    b=torch.randn(1,1,img_pixel_x,img_pixel_y)
    b[0][0]=a[0]
    k=b
    for i in range(number+1, number+batch_size):
        a=torch.tensor(training_data[i][0])
        c=torch.randn(1,1,img_pixel_x,img_pixel_y)
        c[0][0]=a[0]
        k=torch.cat((k,c))
    return k

def batch_y(number):
    #batch_y to tensor
    k=torch.tensor(training_data[number][1])
    for i in range(number+1, number+batch_size):
        a=torch.tensor(training_data[i][1])
        k=torch.cat((k,a))
    k=k.reshape(batch_size,2)
    return k
        

###train
split=1000
accuracy=0
for j in tqdm(range(epochs)):
    for i in range(0, split, batch_size):        
        f=batch_X(i)
        optimizer.zero_grad()
        output=net(f)
        target = batch_y(i)
        loss = loss_function(output.double(), target)
        loss.backward()
        optimizer.step()
    ###evaluate
    correct=0
    for i in range(split+1,len(training_data)):
        a=torch.tensor(training_data[i][0])
        b=torch.randn(1,1,img_pixel_x,img_pixel_y)
        b[0][0]=a[0]
        output=net(b)
        if torch.argmax(output)==torch.argmax(torch.tensor(training_data[i][1])): correct+=1
        previous_accuracy=100*correct/(len(training_data)-split)
    print(", Accuracy:", previous_accuracy, "%, loss: ", loss)
    if previous_accuracy > accuracy: 
        scheduler.step()
        accuracy=previous_accuracy




'''    IMPOVEMENT IDEAS:
    -change batch size
    -change learning rate
    -change optimizer
    -change activation function
    -adjust neurons
    -increase epochs
    -increase convolutional layers
    -change pooling
    -change loss function
    -change image resolution (32, 32) -> (50,50), or (100,100), or full (227,227)
    -(un)normalize pixels
    
'''





    




