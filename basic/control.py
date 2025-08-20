import torch
from torch import nn
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib
import matplotlib.pyplot as plt
import pickle
import torch.optim
from torch.utils.data.dataloader import DataLoader
from torchvision.utils import make_grid
from tqdm import tqdm
import random
import numpy as np
import torch.nn.functional as F
import torch
import torch.nn as nn
import torchvision
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from tqdm import tqdm
import pickle

device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)
random.seed(42)
torch.manual_seed(42)
np.random.seed(42)


num_epochs = 40
learning_rate = 0.0002


#####################################################################################################################
import os
import sys
output_dir = "./outputs_optimized/basic/control/"
os.makedirs(output_dir, exist_ok=True)
sys.stdout = open(os.path.join(output_dir, "log.txt"), "w")
sys.stderr = sys.stdout

# === Save paths ===
model_save_path = os.path.join(output_dir, "vanilla_model.pt")
train_loss_path = os.path.join(output_dir, 'train_loss.pkl')
test_loss_path = os.path.join(output_dir, 'test_loss.pkl')
train_accuracy_path = os.path.join(output_dir, 'train_accuracy.pkl')
test_accuracy_path = os.path.join(output_dir, 'test_accuracy.pkl')

#####################################################################################################################


train_total = [[0.0 for i in range(250)] for j in range(num_epochs)]
test_total = [[0.0 for i in range(250)] for j in range(num_epochs)]


train_losses = []
train_accuracy = []
test_losses = []
test_accuracy = []


train_data = torchvision.datasets.CIFAR10(
    root="data/train",
    train=True,
    download=True,
    transform=ToTensor(),
    target_transform=None
)

test_data = torchvision.datasets.CIFAR10(
    root="data/test",
    train=False,
    download=True,
    transform=ToTensor(),
    target_transform=None
)

train_dataloader = DataLoader(train_data, batch_size=1)
test_dataloader = DataLoader(test_data, batch_size=1)


def accuracy_fn(y_true, y_pred):
    correct = torch.eq(y_true, y_pred).sum().item()
    acc = (correct / len(y_pred)) * 100
    return acc

class VanillaModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.convBlock1 = nn.Sequential(
            nn.Conv2d(3, 16, 3, 1, 1), nn.ReLU(),
            nn.Conv2d(16, 16, 3, 1, 1), nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.convBlock2 = nn.Sequential(
            nn.Conv2d(16, 32, 3, 1, 1), nn.ReLU(),
            nn.Conv2d(32, 32, 3, 1, 1), nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.convBlock3 = nn.Sequential(
            nn.Conv2d(32, 64, 3, 1, 1), nn.ReLU(),
            nn.Conv2d(64, 64, 3, 1, 1), nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.flat = nn.Sequential(
            self.convBlock1, self.convBlock2, self.convBlock3,
            nn.AvgPool2d(1), nn.Flatten()
        )
        self.fc1 = nn.Linear(64 * 4 * 4, 250)
        self.fc2 = nn.Linear(250, 250)
        self.fc3 = nn.Linear(250, 250)
        self.fc4 = nn.Linear(250, 10)
        self.relu = nn.ReLU()
        self.epoch = 0



    def forward(self, x):
        x = self.flat(x)       # [1, 1024]
        x = self.fc1(x)        # [1, 250]
        x = F.relu(x)
        x = self.fc2(x)

        x = F.relu(x)
        # Log total activity
        if self.training:
            train_total[self.epoch - 1] = x.detach().cpu().tolist()
        else:
            test_total[self.epoch - 1] = x.detach().cpu().tolist()

        x = self.fc3(x)
        x = F.relu(x)
        x = self.fc4(x)        # [10]

        return x   # restore batch dim → [1, 10]


model = VanillaModel().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(params=model.parameters(),lr = learning_rate)

for epoch in tqdm(range(1,num_epochs+1)):
    print(f"Epoch: {epoch}\n-------")
    ### training
    train_loss, train_acc = 0, 0
    model.epoch = epoch
    # Add a loop to loop through training batches
    for batch, (X, y) in enumerate(train_dataloader):
        optimizer.zero_grad()
        X, y = X.to(device), y.to(device)
        model.train()
        # 1. Forward pass
        y_pred = model(X)

        # 2. Calculate loss (per batch)
        loss = criterion(y_pred, y)
        train_loss += loss.item() # accumulatively add up the loss per epoch
        train_acc += accuracy_fn(y, y_pred.argmax(dim=1))

        # 4. loss backward
        loss.backward()

        # 5. Optimizer step
        optimizer.step()

    # Divide total train loss by length of train dataloader (average loss per batch per epoch)
    train_loss /= len(train_dataloader)
    train_acc /= len(train_dataloader)

    ### testing
    # Setup variables for accumulatively adding up loss and accuracy
    test_loss, test_acc = 0, 0
    model.eval()
    with torch.no_grad():
        for X, y in test_dataloader:
            X, y = X.to(device), y.to(device)
            test_pred = model(X) # 1. Forward pass
            test_loss += criterion(test_pred, y).item() # 2. accumulatively add up the loss per epoch
            test_acc += accuracy_fn(y_true=y, y_pred=test_pred.argmax(dim=1)) # 3. Calculate accuracy (preds need to be same as y_true)

            # Calculations on test metrics need to happen inside torch.inference_mode()

        test_loss /= len(test_dataloader)
        test_acc /= len(test_dataloader)


    fc2_weights = model.fc2.weight.data.clone().detach().cpu().numpy().flatten()
    
    plt.figure(figsize=(10, 6))
    plt.hist(fc2_weights, bins=50, color='blue',)
    plt.title(f'FC2 Weights Distribution - Epoch {epoch}')
    plt.xlabel('Weight Value')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, f'fc2_weights_histogram_epoch_{epoch}.png'))
    plt.close()
    

    ## Print out what's happening
    print(f"\ntrain loss: {train_loss:.5f}, train acc: {train_acc:.2f} | test loss: {test_loss:.5f}, test acc: {test_acc:.2f}%\n")

    train_losses.append(train_loss)
    test_losses.append(test_loss)
    train_accuracy.append(train_acc)
    test_accuracy.append(test_acc)

    train_total[epoch-1] = [(i/(len(train_dataloader)*1)) for i in train_total[epoch-1]]
    #train_synaptic[epoch-1] = [i/(len(train_dataloader)*1) for i in train_synaptic[epoch-1]]
    test_total[epoch-1] = [i/(len(test_dataloader)*1) for i in test_total[epoch-1]]
    #test_synaptic[epoch-1] = [i/(len(test_dataloader)*1) for i in test_synaptic[epoch-1]]

    if((epoch)%2 == 0):
        with open(os.path.join(output_dir, f'train_total.pkl','wb')) as f:
            pickle.dump(train_total,f)

        with open(os.path.join(output_dir, f'test_total.pkl','wb')) as f:
            pickle.dump(test_total,f)

        with open(train_loss_path,"wb") as f:
            pickle.dump(train_losses, f)
        
        with open(test_loss_path,"wb") as f:
            pickle.dump(test_losses, f)
            
        with open(train_accuracy_path,"wb") as f:
            pickle.dump(train_accuracy, f)
            
        with open(test_accuracy_path,"wb") as f:
            pickle.dump(test_accuracy, f)
        torch.save({'epochs': epoch,'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict()}, model_save_path)
        print(f"saved for epoch {epoch}")



with open(f'train_total.pkl','wb') as f:
    pickle.dump(train_total,f)

with open(f'test_total.pkl','wb') as f:
    pickle.dump(test_total,f)

with open(train_loss_path,"wb") as f:
    pickle.dump(train_losses, f)
    
with open(test_loss_path,"wb") as f:
    pickle.dump(test_losses, f)
    
with open(train_accuracy_path,"wb") as f:
    pickle.dump(train_accuracy, f)
    
with open(test_accuracy_path,"wb") as f:
    pickle.dump(test_accuracy, f)



# Plot train_total for all 250 neurons across epochs
plt.figure(figsize=(15, 10))
for neuron_idx in range(250):
    neuron_values = [train_total[epoch][neuron_idx] for epoch in range(num_epochs)]
    plt.plot(range(1, num_epochs + 1), neuron_values)
plt.xlabel('Epochs')
plt.ylabel('Neuron Value')
plt.title('Neuron Values Averaged Over Samples')
plt.grid(True)
plt.savefig(os.path.join(output_dir,'train_neuron_values.png'))


plt.figure(figsize=(15, 10))
for neuron_idx in range(250):
    neuron_values = [test_total[epoch][neuron_idx] for epoch in range(num_epochs)]
    plt.plot(range(1, num_epochs + 1), neuron_values)
plt.xlabel('Epochs')
plt.ylabel('Neuron Value')
plt.title('Neuron Values Averaged Over Samples')
plt.grid(True)
plt.savefig(os.path.join(output_dir,'test_neuron_values.png'))



plt.figure(figsize=(15, 15))
plt.plot(range(1, len(train_losses) + 1), train_losses, label='train loss', color='purple')
plt.plot(range(1, len(test_losses) + 1), test_losses, label='test loss', color='black')
plt.xlabel('Epochs')
plt.ylabel('loss')
plt.title('Epochs vs training/test loss')
plt.legend()
plt.grid(True)
plt.show()
plt.savefig(os.path.join(output_dir,f'loss_plot.png'))


plt.figure(figsize=(15, 15))
plt.plot(range(1, num_epochs + 1), train_accuracy, label='train acc', color='purple')
plt.plot(range(1, num_epochs + 1), test_accuracy, label='test acc', color='black')
plt.xlabel('Epochs')
plt.ylabel('acc')
plt.title('Epochs vs training/test acc')
plt.legend()
plt.grid(True)
plt.show()
plt.savefig(os.path.join(output_dir,f'accuracy_plot.png'))