import torch
import argparse
from models.cnn import CNN
from dataset import PrepareDataset
def eval(args):
    dataset = PrepareDataset()
    device = "cpu" if args.device=="cpu" else "cuda"
    if args.model.lower() == 'cnn':
        model = CNN()
    model.load_state_dict(torch.load(args.weights, weights_only=True))
    model.to(device)
    model.eval()
    correct = 0
    total_samples = 0
    with torch.no_grad():
        for i, (images, labels) in dataset.test_loader:
            images = images.to(device)
            labels = labels.to(device)
            y_pred = model(images)
            total_samples += i
            correct += (y_pred==labels).sum().item()
        print("Accuracy of the network: ", correct/total_samples*100)


def parse_args():
    parse = argparse.ArgumentParser("Config Evaluation")
    parse.add_argument("--weights", type=str, default='', help="Path to model weights")
    parse.add_argument("--model", type='str', default='cnn', help="Select model to evaluate")
    parse.add_argument("--device", default='cpu', help="Select device like cpu or 0,1,2,...")
def main():
    pass
if __name__=='__main__':
    eval()
