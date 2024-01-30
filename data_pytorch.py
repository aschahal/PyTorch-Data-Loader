import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms



def get_data_loader(training = True):
    """
    INPUT: 
        An optional boolean argument (default value is True for training dataset)

    RETURNS:
        Dataloader for the training set (if training = True) or the test set (if training = False)
    """
    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
        ])
    
    # Uses correct dataset based on training flag
    if training:
        dataset = datasets.FashionMNIST('./data', train = True, download = True, transform = transform)
        loader = torch.utils.data.DataLoader(dataset, batch_size = 64, shuffle = True)
    else:
        dataset = datasets.FashionMNIST('./data', train = False, transform = transform)
        loader = torch.utils.data.DataLoader(dataset, batch_size = 64, shuffle = False)
    
    return loader


def build_model():
    """
    INPUT: 
        None

    RETURNS:
        An untrained neural network model
    """
    model = nn.Sequential(
        nn.Flatten(),
        nn.Linear(28*28, 128), 
        nn.ReLU(),
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Linear(64, 10)
    )

    return model


def train_model(model, train_loader, criterion, T):
    """
    TODO: implement this function.

    INPUT: 
        model - the model produced by the previous function
        train_loader  - the train DataLoader produced by the first function
        criterion   - cross-entropy 
        T - number of epochs for training

    RETURNS:
        None
    """
    optimizer = optim.SGD(model.parameters(), lr = 0.001, momentum = 0.9)
    model.train()

    for epoch in range(T):
        total_loss = 0
        correct = 0

        for data, target in train_loader:
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * data.size(0)

            pred = output.argmax(dim = 1, keepdim = True)
            correct += pred.eq(target.view_as(pred)).sum().item()

        avg_loss = total_loss / len(train_loader.dataset)
        accuracy = 100. * correct / len(train_loader.dataset)
        print(f'Train Epoch: {epoch} Accuracy: {correct}/{len(train_loader.dataset)}({accuracy: .2f}%) Loss: {avg_loss: .3f}')
    


def evaluate_model(model, test_loader, criterion, show_loss = True):
    """
    TODO: implement this function.

    INPUT: 
        model - the the trained model produced by the previous function
        test_loader    - the test DataLoader
        criterion   - cropy-entropy 

    RETURNS:
        None
    """
    model.eval()
    test_loss = 0
    correct = 0

    with torch.no_grad():
        for data, target in test_loader:
            output = model(data)
            test_loss += criterion(output, target).item() * data.size(0)
            pred = output.argmax(dim = 1, keepdim = True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    
    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)

    if show_loss:
        print(f'Average loss: {test_loss:.4f}')
    print(f'Accuracy: {accuracy:.2f}%')
    


def predict_label(model, test_images, index):
    """
    TODO: implement this function.

    INPUT: 
        model - the trained model
        test_images   -  a tensor. test image set of shape Nx1x28x28
        index   -  specific index  i of the image to be tested: 0 <= i <= N - 1


    RETURNS:
        None
    """
    if index >= len(test_images) or index < 0:
        print("Error: Invalid index provided:")
        return
    
    class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle Boot']

    model.eval()
    with torch.no_grad():
        image = test_images[index].unsqueeze(0)
        logits = model(image)
        prob = F.softmax(logits, dim = 1)
        top3_prob, top3_indices = torch.topk(prob, 3)

        for i in range(3):
            print(f"{class_names[top3_indices[0][i]]}: {top3_prob[0][i].item()*100:.2f}%")


if __name__ == '__main__':
    '''
    Feel free to write your own test code here to exaime the correctness of your functions. 
    Note that this part will not be graded.
    '''
    criterion = nn.CrossEntropyLoss()
