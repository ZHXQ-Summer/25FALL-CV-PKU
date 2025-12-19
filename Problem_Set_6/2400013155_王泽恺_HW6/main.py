import torch
import torch.nn as nn
import argparse
import torchvision
import torchvision.transforms as transforms
import os
from models import VGG, ResNet, ResNext 

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import matplotlib.pyplot as plt
import time
from torch.utils.tensorboard import SummaryWriter

PATH = './mymodel.pt'
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

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
    ax2.axhline(y=80, color='g', linestyle='--', linewidth=2, label='Target (80%)')
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
    '''
    model.to(device)
    
    # 数据增强
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
    num_workers = 2 if os.name != 'nt' else 0
    
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    
    epoch_cnt = 48
    train_losses = []
    test_accuracies = []
    
    current_time = time.strftime('%Y-%m-%d_%H-%M-%S', time.localtime())
    log_dir = f'./log/{current_time}'
    writer = SummaryWriter(log_dir)
    running_loss = 0.0
    
    print(f"Start training {args.model} on {device}...")
    
    for epoch in range(epoch_cnt):
        model.train() # 确保在训练模式
        temp_loss = 0.0
        
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data[0].to(device), data[1].to(device)
            
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
        
        # Test phase
        correct = 0
        total = 0
        model.eval()
        with torch.no_grad():
            for data in testloader:
                images, labels = data[0].to(device), data[1].to(device)

                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        acc = 100 * correct / total
        print(f'Epoch {epoch+1} Accuracy: {acc:.2f} %')
        
        average_loss = temp_loss / len(trainloader)
        
        train_losses.append(average_loss)
        test_accuracies.append(acc)
        
        writer.add_scalar('Loss/train', average_loss, epoch)
        writer.add_scalar('Accuracy/test', acc, epoch)

    # save checkpoint
    writer.flush()
    writer.close()
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'train_losses': train_losses,
        'test_accuracies': test_accuracies
    }, PATH)
    plot_training_curves(train_losses, test_accuracies)

def test(model, args):
    '''
    Test function
    '''
    model.to(device)
    
    # 加载权重
    if os.path.exists(PATH):
        checkpoint = torch.load(PATH)
        model.load_state_dict(checkpoint['model_state_dict'])
        print("Loaded model checkpoint.")
    else:
        print(f"Error: No checkpoint found at {PATH}")
        return

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    batch_size = 64
    num_workers = 2 if os.name != 'nt' else 0
    
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    correct = 0
    total = 0
    model.eval()
    
    with torch.no_grad():
        for data in testloader:
            images, labels = data[0].to(device), data[1].to(device)
            
            
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
    parser.add_argument('--model', type=str, default='vgg')
    args = parser.parse_args()

    # create model
    if args.model == 'vgg':
        model = VGG()
    elif args.model == 'resnet':

        model = ResNet() 
    elif args.model == 'resnext':

        model = ResNext()
    else: 
        raise AssertionError(f"Unknown model: {args.model}")

    # create optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)
    
    if args.run == 'train':
        train(model, optimizer, scheduler, args)
    elif args.run == 'test':
        test(model, args)
    else: 
        raise AssertionError("Invalid run mode. Use 'train' or 'test'.")
