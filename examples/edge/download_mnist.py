from torchvision import datasets, transforms

training_data = datasets.MNIST('data',
                                   train=True,
                                   download=True,
                                   transform=transforms.Compose([
                                       transforms.ToTensor(),
                                       transforms.Normalize((0.1307, ),
                                                            (0.3081, ))
                                   ]))
                                   
