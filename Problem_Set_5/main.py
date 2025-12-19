import torch
import torch.nn as nn
import argparse
import torchvision
import torchvision.transforms as transforms
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import matplotlib.pyplot as plt
import time
from torch.utils.tensorboard import SummaryWriter
PATH = './mymodel.pt'
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
class LinearClassifier(nn.Module):
    # define a linear classifier
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        # inchannels: dimenshion of input data. For example, a RGB image [3x32x32] is converted to vector [3 * 32 * 32], so dimenshion=3072
        # out_channels: number of categories. For CIFAR-10, it's 10
        self.fc=nn.Linear(in_channels,out_channels)
    def forward(self, x: torch.Tensor):
        return self.fc(x)


class FCNN(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int):
        super().__init__()
        self.fc1 = nn.Linear(in_channels, hidden_channels)
        self.bn1 = nn.BatchNorm1d(hidden_channels)  
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.5)            

        self.fc2 = nn.Linear(hidden_channels, hidden_channels // 2) 
        self.bn2 = nn.BatchNorm1d(hidden_channels // 2)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(0.5)

        self.fc3 = nn.Linear(hidden_channels // 2, 512) 
        self.bn3 = nn.BatchNorm1d(512)
        self.relu3 = nn.ReLU()
        
        self.fc4 = nn.Linear(512, out_channels)

    def forward(self, x: torch.Tensor): 
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.dropout1(x)

        x = self.fc2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.dropout2(x)
        
        x = self.fc3(x)
        x = self.bn3(x)
        x = self.relu3(x)
        
        x = self.fc4(x)
        return x
def plot_training_curves(train_losses, test_accuracies, save_path='./training_curves.png'):
    """绘制训练损失和测试准确率曲线"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    epochs = range(1, len(train_losses) + 1)
    
    # 左图：训练损失
    ax1.plot(epochs, train_losses, 'b-', linewidth=2, label='Training Loss')
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('Training Loss', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # 右图：测试准确率
    ax2.plot(epochs, test_accuracies, 'r-', linewidth=2, label='Test Accuracy')
    ax2.axhline(y=60, color='g', linestyle='--', linewidth=2, label='Target (60%)')
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Accuracy (%)', fontsize=12)
    ax2.set_title('Test Accuracy', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f'\nTraining curves saved to {save_path}')
def train(model, optimizer, scheduler, args):
    '''
    Model training function
    input: 
        model: linear classifier or full-connected neural network classifier
        loss_function: Cross-entropy loss
        optimizer: Adamw or SGD
        scheduler: step or cosine
        args: configuration
    '''
    model.to(device)
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    criterion = nn.CrossEntropyLoss()
    batch_size = 128
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)
    epoch_cnt = 48
    train_losses = []
    test_accuracies = []
    
    current_time = time.strftime('%Y-%m-%d_%H-%M-%S', time.localtime())
    log_dir = f'./log/{current_time}'
    writer = SummaryWriter(log_dir)
    running_loss = 0.0
    
    for epoch in range(epoch_cnt):
        temp_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data[0].to(device), data[1].to(device)  
            inputs = nn.Flatten()(inputs)  
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            temp_loss += loss.item()
            if i % 125 == 124:
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 125:.3f}')
                running_loss = 0.0
        
        scheduler.step()
        
        # test
        correct = 0
        total = 0
        model.eval()
        with torch.no_grad():
            for data in testloader:
                images, labels = data[0].to(device), data[1].to(device)
                images = nn.Flatten()(images)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        print(f'Accuracy of the network on the 10000 test images: {100 * correct // total} %')
        model.train()
        
        average_loss = temp_loss / len(trainloader)
        accuracy = 100 * correct / total
        
        train_losses.append(average_loss)
        test_accuracies.append(accuracy)
        
        writer.add_scalar('Loss/train', average_loss, epoch)
        writer.add_scalar('Accuracy/test', accuracy, epoch)  # 修正：改为test

    # save checkpoint
    writer.flush()
    writer.close()
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'train_losses': train_losses,           # 新增：保存历史
        'test_accuracies': test_accuracies      # 新增：保存历史
    }, PATH)
    plot_training_curves(train_losses, test_accuracies)
def test(model, args):
    '''
    input: 
        model: linear classifier or full-connected neural network classifier
        loss_function: Cross-entropy loss
    '''
    model.to(device)
    checkpoint=torch.load(PATH)
    model.load_state_dict(checkpoint['model_state_dict'])
    transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    criterion = nn.CrossEntropyLoss()
    batch_size = 64
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)

    correct = 0
    total = 0
    model.eval()
    with torch.no_grad():
        # forward
        for data in testloader:
            images, labels = data[0].to(device), data[1].to(device)
            images = nn.Flatten()(images)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print(f'Accuracy of the network on the 10000 test images: {100 * correct // total} %')
    if 'train_losses' in checkpoint and 'test_accuracies' in checkpoint:
        plot_training_curves(checkpoint['train_losses'], checkpoint['test_accuracies'])
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='The configs')

    parser.add_argument('--run', type=str, default='train')
    parser.add_argument('--model', type=str, default='linear')
    parser.add_argument('--optimizer', type=str, default='adamw')
    parser.add_argument('--scheduler', type=str, default='step')
    args = parser.parse_args()

    # create model
    if args.model == 'linear':
        model = LinearClassifier(3*32*32,10)
    elif args.model == 'fcnn':
        model = FCNN(3*32*32,3072,10)
    else: 
        raise AssertionError

    # create optimizer
    if args.optimizer == 'adamw':
        # create Adamw optimizer
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-5)
    elif args.optimizer == 'sgd':
        # create SGD optimizer
        optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=1e-5)
    else:
        raise AssertionError
    
    # create scheduler
    if args.scheduler == 'step':
        # create torch.optim.lr_scheduler.StepLR scheduler
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.1)
    elif args.scheduler == 'cosine':
        # create torch.optim.lr_scheduler.CosineAnnealingLR scheduler
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)
    else:
        raise AssertionError

    if args.run == 'train':
        train(model, optimizer, scheduler, args)
    elif args.run == 'test':
        test(model, args)
    else: 
        raise AssertionError
    
# You need to implement training and testing function that can choose model, optimizer, scheduler and so on by command, such as:
# python main.py --run=train --model=fcnn --optimizer=adamw --scheduler=step

