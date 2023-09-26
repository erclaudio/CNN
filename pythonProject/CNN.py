'''resources:
https://wandb.ai/authors/ayusht/reports/Implementing-Dropout-in-PyTorch-With-Example--VmlldzoxNTgwOTE
https://nextjournal.com/gkoehler/pytorch-mnist'''
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
import matplotlib.pyplot as plt
train_set = torchvision.datasets.FashionMNIST(root = ".", train = True ,
download = True , transform = transforms.ToTensor())
test_set = torchvision.datasets.FashionMNIST(root = ".", train = False ,
download = True , transform = transforms.ToTensor())
training_loader = torch.utils.data.DataLoader(train_set , batch_size = 32,
shuffle = False)
test_loader = torch.utils.data.DataLoader(test_set , batch_size = 32,
shuffle = False)
torch.manual_seed(0)
# If you are using CuDNN , otherwise you can just ignore
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
use_cuda = True
class cnn(nn.Module):

    def __init__(self):
        super(cnn, self).__init__()
        # define first convolution layer with one image as input, 32 channels output and 5 x 5 convolution kernel
        self.conv1 = nn.Conv2d(1, 32, 5)
        # xavier initialisation
        nn.init.xavier_normal_(self.conv1.weight)
        # second convolution layer, 32 input image channel, 5 x 5 kernel size
        self.conv2 = nn.Conv2d(32, 64, 5)
        nn.init.xavier_normal_(self.conv2.weight)
        # first fully connected layer
        self.fc1 = nn.Linear(64 * 4 * 4, 256)
        nn.init.xavier_normal_(self.fc1.weight)
        # second fully connected layer
        self.fc2 = nn.Linear(256, 10)
        nn.init.xavier_normal_(self.fc2.weight)
        # dropout
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        # convolution and relu activation, max pooling over 2x2 window
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), (2, 2))
        # tensors get flattened
        x = x.view(-1, self.num_flat_features(x))
        # using relu activation after fully connected laye
        x = F.relu(self.fc1(x))
        #x = self.dropout(self.fc2(x))
        # last layer
        x = self.fc2(x)
        # dropout on the 2nd fully connected layer
        #x = self.dropout(self.fc2(x))

        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


model = cnn()

# use google colab gpu if resources are available
if use_cuda and torch.cuda.is_available():
    model.cuda()

# defining learning rate
l_rate = 0.1
# defining loss function
criterion = nn.CrossEntropyLoss()
# defining stochastic gradient descent for weights update
optimizer = optim.SGD(model.parameters(), lr=l_rate)


# evaluation method to calculate the accuracy of the model on training and test set
def evaluation(model, data_loader):
    model.eval()
    total, correct = 0, 0
    for data in data_loader:
        inputs, label = data
        if use_cuda and torch.cuda.is_available():
            inputs = inputs.cuda()
            label = label.cuda()
        outputs = model(inputs)
        _, pred = torch.max(outputs.data, 1)
        total = total + label.size(0)
        correct = correct + (pred == label).sum().item()
    # return accuracy
    return 100 * (correct / total)


X_totalloss = []
# training
X_training_accuracy = []
# testing
X_testing_accuracy = []
# epochs
epochs = 50

# training loop
for epoch in range(epochs):
    # loss list for batches
    X_loss = []
    for i, (images, labels) in enumerate(training_loader):
        # change the mode of the model to training
        model.train()
        if use_cuda and torch.cuda.is_available():
            images = images.cuda()
            labels = labels.cuda()
        outputs = model(images)
        # calculate loss
        loss = criterion(outputs, labels)
        # append loss to the loss list
        X_loss.append(loss.item())

        # set gradient buffers to zero
        optimizer.zero_grad()
        # Backprop and perform SGD optimisation
        loss.backward()
        optimizer.step()

    # get training accuracy
    train_acc = evaluation(model, training_loader)
    # get testing accuracy
    test_acc = evaluation(model, test_loader)
    X_training_accuracy.append(train_acc)
    X_testing_accuracy.append(test_acc)
    # get loss for the epoch and append to the total loss list
    X_totalloss.append(sum(X_loss))
    print('Epoch [{}/{}], Loss: {:.4f}, Train Accuracy: {:.2f}%, Test Accuracy: {:.2f}%'
          .format(epoch + 1, epochs, sum(X_loss), train_acc, test_acc))

plt.plot(X_training_accuracy, label="Train Acc")
plt.plot(X_testing_accuracy, label="Test Acc")
plt.title('Test and Train Accuracy at LR={}'.format(l_rate))
plt.legend()
plt.show()

plt.plot(X_totalloss, label="Loss")
plt.title('Loss per epoch at LR={}'.format(l_rate))
plt.legend()
plt.show()