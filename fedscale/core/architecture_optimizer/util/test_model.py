import torch
import torchvision
import torchvision.transforms as transforms
def test(model: torch.nn.Module, testloader = None) -> float:
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    if testloader == None:
        transform = transforms.Compose(
                    [transforms.ToTensor(),
                     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
        testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                         shuffle=False, num_workers=1)
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            inputs, labels, = data[0].to(device), data[1].to(device)
            model.train(False)
            model.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    acc = correct / total
    return acc