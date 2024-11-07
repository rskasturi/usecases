# Data Parallel

Test Code: ResNet50 on CIFAR

```python
import torch
import torchvision
import torchvision.transforms as transforms
import intel_extension_for_pytorch as ipex

transform = transforms.Compose([##
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=10,
                                                  shuffle=True, num_workers=0)

if __name__ == "__main__":
    data = trainloader

    device = torch.device("xpu" if torch.xpu.is_available() else "cpu")

    net = torchvision.models.resnet50()
    if torch.xpu.device_count() > 1:
        print("Let's use", torch.xpu.device_count(), "GPUs!")
        net = torch.nn.DataParallel(net)
    net.to(device)
    loss_fn = torch.nn.CrossEntropyLoss(reduction="mean")

    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)

    # Train ResNet50 model
    net.train()
    for epoch in range(1, 1 + 1):  # only one epoch for debugging
        train_loss = 0.0
        for x, y in trainloader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = loss_fn(net(x), y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
```

## Experiment Result

FAILED

## Observation

[IPEX torch.nn.DataParallel Issue](https://github.com/intel/intel-extension-for-pytorch/issues/271)

> DP is not supported in IPEX.

> DP is deprecated in PyTorch.

> Distributed Data Parallel (DDP) is currently the standard for multi-GPU training