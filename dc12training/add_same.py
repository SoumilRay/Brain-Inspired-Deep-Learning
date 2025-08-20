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

#####################################################################################################################
import os
import sys
output_dir = "./outputs_optimized/dc12training/add_same/"
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


device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)
random.seed(42)
torch.manual_seed(42)
np.random.seed(42)


num_epochs = 40
decaying_factor = 1.05
p = 0.0592
learning_rate = 0.0002


train_synaptic = [[0.0 for i in range(250)] for j in range(num_epochs)]
train_total = [[0.0 for i in range(250)] for j in range(num_epochs)]
test_synaptic = [[0.0 for i in range(250)] for j in range(num_epochs)]
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
    def __init__(self, p, dropconnect_p=0.8):
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
        self.p = p
        self.dropconnect_p = dropconnect_p
        self.epoch = 0

        # Coordinate + sequence initialization
        self.sequence = torch.randperm(250)
        self.coordinates = torch.rand(250, 2)
        self.delta_time = torch.zeros(250)
        for t, i in enumerate(self.sequence):
            self.delta_time[i] = t

        self.activated_before = [[] for _ in range(250)]
        for i in range(1, len(self.sequence)):
            self.activated_before[self.sequence[i]].extend(self.activated_before[self.sequence[i - 1]])
            self.activated_before[self.sequence[i]].append(self.sequence[i - 1].item())

        # Distance and decay
        self.distance = torch.zeros(250, 250)
        self.decay_matrix = torch.zeros(250, 250)
        for i in self.sequence:
            for j in range(250):
                if j not in self.activated_before[i] and i != j:
                    dx = self.coordinates[i][0] - self.coordinates[j][0]
                    dy = self.coordinates[i][1] - self.coordinates[j][1]
                    self.distance[i, j] = 1 / (dx ** 2 + dy ** 2)
                    self.decay_matrix[i, j] = decaying_factor ** (self.delta_time[i] - self.delta_time[j])

        self.ephaptic_matrix = (self.p * self.decay_matrix * self.distance) / 250
        self.ephaptic_matrix.fill_diagonal_(0)
        self.ephaptic_matrix = self.ephaptic_matrix.to(device)

    def forward(self, x):
        x = self.flat(x)       # [1, 1024]
        x = self.fc1(x)        # [1, 250]
        x = F.relu(x)



        if self.training:
            self.drop_connect = torch.bernoulli(self.dropconnect_p * torch.ones_like(self.fc2.weight, device=device))
            x = F.linear(x, self.fc2.weight * self.drop_connect, self.fc2.bias)
        else:
            x = self.fc2(x)


        # Remove batch dimension
        x = x.squeeze(0)                   # → [250]
        copy_x = x.detach()

        # Log synaptic activity before ephaptic effect
        if self.training:
            train_synaptic[self.epoch - 1] = copy_x.cpu().tolist()
        else:
            test_synaptic[self.epoch - 1] = copy_x.cpu().tolist()

        # Apply ephaptic effect
        eph = torch.matmul(copy_x.to(device), self.ephaptic_matrix)  # [250]
        x = x + eph
        x = F.relu(x)
        # Log total activity
        if self.training:
            train_total[self.epoch - 1] = x.detach().cpu().tolist()
        else:
            test_total[self.epoch - 1] = x.detach().cpu().tolist()

        x = self.fc3(x)
        x = F.relu(x)
        x = self.fc4(x)        # [10]

        return x.unsqueeze(0)  # restore batch dim → [1, 10]


model = VanillaModel(p).to(device)
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
    train_synaptic[epoch-1] = [i/(len(train_dataloader)*1) for i in train_synaptic[epoch-1]]
    test_total[epoch-1] = [i/(len(test_dataloader)*1) for i in test_total[epoch-1]]
    test_synaptic[epoch-1] = [i/(len(test_dataloader)*1) for i in test_synaptic[epoch-1]]

    if((epoch)%2 == 0):
        with open(os.path.join(output_dir, f'train_total.pkl'),'wb') as f:
            pickle.dump(train_total,f)

        with open(os.path.join(output_dir, f'train_synaptic.pkl'),'wb') as f:
            pickle.dump(train_synaptic,f)   

        with open(os.path.join(output_dir, f'test_total.pkl'),'wb') as f:
            pickle.dump(test_total,f)

        with open(os.path.join(output_dir, f'test_synaptic.pkl'),'wb') as f:
            pickle.dump(test_synaptic,f)

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


with open(os.path.join(output_dir, f'train_total.pkl'),'wb') as f:
    pickle.dump(train_total,f)

with open(os.path.join(output_dir, f'train_synaptic.pkl'),'wb') as f:
    pickle.dump(train_synaptic,f)   

with open(os.path.join(output_dir, f'test_total.pkl'),'wb') as f:
    pickle.dump(test_total,f)

with open(os.path.join(output_dir, f'test_synaptic.pkl'),'wb') as f:
    pickle.dump(test_synaptic,f)

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
plt.savefig(os.path.join(output_dir, 'train_neuron_values.png'))


plt.figure(figsize=(15, 10))
for neuron_idx in range(250):
    neuron_values = [test_total[epoch][neuron_idx] for epoch in range(num_epochs)]
    plt.plot(range(1, num_epochs + 1), neuron_values)
plt.xlabel('Epochs')
plt.ylabel('Neuron Value')
plt.title('Neuron Values Averaged Over Samples')
plt.grid(True)
plt.savefig(os.path.join(output_dir, 'test_neuron_values.png'))



plt.figure(figsize=(15, 15))
plt.plot(range(1, len(train_losses) + 1), train_losses, label='train loss', color='purple')
plt.plot(range(1, len(test_losses) + 1), test_losses, label='test loss', color='black')
plt.xlabel('Epochs')
plt.ylabel('loss')
plt.title('Epochs vs training/test loss')
plt.legend()
plt.grid(True)
plt.show()
plt.savefig(os.path.join(output_dir, f'loss_plot.png'))


plt.figure(figsize=(15, 15))
plt.plot(range(1, num_epochs + 1), train_accuracy, label='train acc', color='purple')
plt.plot(range(1, num_epochs + 1), test_accuracy, label='test acc', color='black')
plt.xlabel('Epochs')
plt.ylabel('acc')
plt.title('Epochs vs training/test acc')
plt.legend()
plt.grid(True)
plt.show()
plt.savefig(os.path.join(output_dir, f'accuracy_plot.png'))


train_synaptic_to_total_ratio = []
test_synaptic_to_total_ratio = []
for epoch in range(num_epochs):
    # Average over 250 neurons for train
    avg_train_synaptic = np.mean(train_synaptic[epoch])
    avg_train_total = np.mean(train_total[epoch])
    train_ratio = avg_train_synaptic / avg_train_total 
    train_synaptic_to_total_ratio.append(train_ratio)

    # Average over 250 neurons for test
    avg_test_synaptic = np.mean(test_synaptic[epoch])
    avg_test_total = np.mean(test_total[epoch])
    test_ratio = avg_test_synaptic / avg_test_total 
    test_synaptic_to_total_ratio.append(test_ratio)
# Plot the ratios
plt.figure(figsize=(15, 15))
plt.plot(range(1, num_epochs + 1), train_synaptic_to_total_ratio, label='Train Synaptic/Total Ratio', color='purple')
plt.plot(range(1, num_epochs + 1), test_synaptic_to_total_ratio, label='Test Synaptic/Total Ratio', color='black')
plt.xlabel('Epochs')
plt.ylabel('Synaptic/Total Ratio (Averaged over 250 Neurons)')
plt.title('Epochs vs Synaptic/Total Ratio')
plt.legend()
plt.grid(True)
plt.show()
plt.savefig(os.path.join(output_dir, 'synaptic_to_total_ratio_plot.png'))


# Calculate and plot histograms of synaptic/total ratios for each epoch
for epoch in range(num_epochs):
    # Calculate ratios for all 250 neurons in this epoch
    train_ratios = []
    test_ratios = []
    
    for neuron_idx in range(250):
        # Calculate ratio for this neuron
        train_ratio = train_synaptic[epoch][neuron_idx] / train_total[epoch][neuron_idx] if train_total[epoch][neuron_idx] != 0 else 0
        test_ratio = test_synaptic[epoch][neuron_idx] / test_total[epoch][neuron_idx] if test_total[epoch][neuron_idx] != 0 else 0
        train_ratios.append(train_ratio)
        test_ratios.append(test_ratio)
    
    # Create figure with two subplots side by side
    plt.figure(figsize=(15, 6))
    
    # Training histogram
    plt.subplot(1, 2, 1)
    plt.hist(train_ratios, bins=50, color='purple')
    plt.title(f'Train Synaptic/Total Ratios - Epoch {epoch + 1}')
    plt.xlabel('Synaptic/Total Ratio')
    plt.ylabel('Frequency')
    plt.grid(True)
    
    # Testing histogram
    plt.subplot(1, 2, 2)
    plt.hist(test_ratios, bins=50, color='black')
    plt.title(f'Test Synaptic/Total Ratios - Epoch {epoch + 1}')
    plt.xlabel('Synaptic/Total Ratio')
    plt.ylabel('Frequency')
    plt.grid(True)
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'synaptic_total_ratio_histogram_epoch_{epoch + 1}.png'))
    plt.close()  # Close the figure to free memory