import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from architecture_optimizer.util.test_model import test
def train(model: torch.nn.Module, epoch: int, trainloader = None, testloader = None) -> torch.nn.Module:
    accu_log = []
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    if trainloader == None:
        transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=4, shuffle=True, num_workers=1)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
    for e in range(epoch):
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, labels, = data[0].to(device), data[1].to(device)
            optimizer.zero_grad()
            model.to(device)
            model.train()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            optimizer.step()
            running_loss += loss.item()
            if i % 1000 == 999:
                last_loss = running_loss / 1000
                print(f"[{e+1}, {i+1:5d}] loss: {last_loss}")
                running_loss = 0.0
        scheduler.step()
        if e % 5 == 4:
            accu = test(model, testloader=testloader)
            print(f"training epoch {e}: test_accuracy = {accu}")
            accu_log.append(accu)
    print(f"finish training for {epoch} epochs")
    return model, accu_log
