import os
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import glob

class NumberDataset(data.Dataset):
    def __init__(self):
        data_dir = 'set'
        paths = glob.glob(os.path.join(data_dir, '*.jpg'))
        self.dataset = []
        for path in paths:
            image = torch.Tensor(cv2.imread(path, cv2.IMREAD_GRAYSCALE))
            label = int(path.split('/')[-1].split('.')[-2])
            target = torch.LongTensor([label])
            self.dataset.append({
                'input': image,
                'target':  target
            })

            self.input_shape = image.shape

    def __getitem__(self, idx):
        item = self.dataset[idx]
        input = item['input'].view(-1)
        target = item['target']

        return (input, target)

    def __len__(self):
        return len(self.dataset)


class Network(nn.Module):
    def __init__(self, num_input, num_layers, num_classes):
        super(Network, self).__init__()
        self.num_classes = num_classes
        self.num_input = num_input

        self.layers = self._make_layers(num_layers)


    def _make_layers(self, num_layers):
        nodes = 500
        layers = []
        if num_layers == 1:
            layers += [nn.Linear(in_features=self.num_input, out_features=self.num_classes, bias=True), nn.Sigmoid()]
        else:
            layers += [nn.Linear(in_features=self.num_input, out_features=nodes, bias=True), nn.Sigmoid()] #nn.LeakyReLU(True)]
            for i in range(1, num_layers - 1):
                layers += [nn.Linear(in_features=nodes, out_features=nodes, bias=True), nn.Sigmoid()] #nn.LeakyReLU(True)]
            layers += [nn.Linear(in_features=nodes, out_features=self.num_classes, bias=True)] #, nn.Sigmoid()]

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.layers(x)
        x = F.softmax(x, dim=1)
        return x

class Loss(nn.Module):
    def __init__(self, num_classes):
        super(Loss, self).__init__()
        self.num_classes = num_classes

    def forward(self, pred, target):
        loss = F.cross_entropy(pred.view(-1, self.num_classes), target.view(-1), reduce=True)
        return loss

def main():
    dataset = NumberDataset()
    input_shape = dataset.input_shape
    num_inputs = 1
    for x in input_shape:
        num_inputs *= x

    num_classes = 10
    num_layers = 5
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=1)
    model = Network(num_input=num_inputs, num_layers=num_layers, num_classes=num_classes)
    loss_fn = Loss(num_classes)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.8)
    train_loss = 0
    for epoch in range(300):
        model.train()
        for idx, (input, target) in enumerate(train_loader):
            optimizer.zero_grad()
            pred = model(input)
            class_pred = torch.argmax(pred, dim=1)
            # print("pred ", class_pred, " target: ", target)

            train_loss = loss_fn(pred, target)
            train_loss.backward()
            optimizer.step()
            print('loss: %.3f' % train_loss.item())
    print('evaluate model')
    model.eval()
    with torch.no_grad():
        for (input, target) in dataset:
            pred = model(input.unsqueeze(0))
            class_pred = torch.argmax(pred, dim=1)
            print("pred ", class_pred, pred[0][class_pred], " target: ", target)



    state = {
        'model': model.state_dict(),
        'loss': train_loss
    }
    torch.save(state, 'net.pth')

if __name__ == '__main__':
    main()
