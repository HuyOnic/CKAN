import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
from models.cnn import CNN
from tqdm import tqdm
import argparse
import sys
from pathlib import Path
import os
# Global Variable
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))

def train(args):
    if args.dataset=="fashion_mnist":
        training_data = datasets.FashionMNIST(root='data', train=True, download=True, transform=ToTensor())
        test_data = datasets.FashionMNIST(root='data', train=False, download=True, transform=ToTensor())
        train_loader = DataLoader(training_data, batch_size=8, shuffle=True)
        test_loader = DataLoader(test_data, batch_size=8, shuffle=True)

    if args.conv.lower() == 'cnn':
        model = CNN(num_classes=10, in_channels=1, kernel_size=3, stride=1, padding=1)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, weight_decay=0.0005, momentum= 0.9)
    device = 'cpu' if args.device=='cpu' else 'cuda'
    num_epochs = args.epochs
    epochs = tqdm(range(num_epochs))
    losses = []
    loss_per_epoch = []

    model.to(device)
    for epoch in epochs:
        print(f"Start training epoch {epoch}/{num_epochs}")
        for i, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)
            output = model(images)
            loss = criterion(output, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_per_epoch.append(loss.item())
        mean_loss = sum(loss_per_epoch)/len(loss_per_epoch)
        print(f'Epoch: {epoch}/{num_epochs}, Loss: {mean_loss}')
        losses.append(mean_loss)
        loss_per_epoch = []
    torch.save(model.state_dict(), args.name)

def parse_opt():
    parser = argparse.ArgumentParser(description="Config some arguments")
    parser.add_argument('--dataset', type=str, default='fashion_mnist', help='Path to dataset')
    parser.add_argument('--epochs', type=int, default=100, help="Total training epochs")
    parser.add_argument('--batch-size', type=int, default=8, help="total batch size for all gpu")
    parser.add_argument('--device', default='cpu', help="cpu or cuda device like 0,1,2,...")
    parser.add_argument('--seed', type=int, default=0, help="seed number")
    parser.add_argument('--hyp', type=str, default=ROOT / 'data/hyps/hyp_scratch.yaml', help="hyper-parameting file path")
    parser.add_argument('--conv', type=str, default='cnn', help="Seclect A Convolutional Model")
    parser.add_argument('--name', type=str, default="cnn", help="Give a name for this model")
    return parser.parse_args()
def main():
    args = parse_opt()
    print(args)
    train(args)
    # for images, labels in train_loader:
    #     print(images.size())
    #     print(labels)
    #     out = model(images)
    #     print(out)
    #     y_pred = out.argmax(1)
    #     print(y_pred)
    #     break



if __name__=='__main__':
    args = parse_opt()
    print(args.batch_size)
    main()