import os
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import glob
import timeit


def gaussian(ins, is_training, mean=0, stddev=0.001):
    if is_training:
        noise = ins.data.new(ins.size()).normal_(mean, stddev)
        return ins + noise
    return ins

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
        input = gaussian(input, is_training=True)

        return (input, target)

    def __len__(self):
        return len(self.dataset)




class Network(nn.Module):
    def __init__(self, num_input, num_layers, num_classes, num_nodes, activation, use_batch_norm=False):
        super(Network, self).__init__()
        self.num_classes = num_classes
        self.num_input = num_input
        self.num_nodes = num_nodes
        self.activation = activation
        self.use_batch_norm = use_batch_norm

        self.layers = self._make_layers(num_layers)


    def _make_layers(self, num_layers):
        layers = []
        if num_layers == 1:
            layers += self._make_layer(self.num_input, self.num_classes)
        else:
            layers += self._make_layer(self.num_input, self.num_nodes)
            for i in range(1, num_layers - 1):
                layers += self._make_layer(self.num_nodes, self.num_nodes)
            layers += self._make_layer(self.num_nodes, self.num_classes)

        return nn.Sequential(*layers)

    def _make_layer(self, inputs, ouputs):
        layers = []
        layers.append(nn.Linear(in_features=inputs, out_features=ouputs, bias=True))
        if self.use_batch_norm:
            layers.append(nn.BatchNorm1d(ouputs))
        layers.append(self.activation)
        return layers

    def forward(self, x):
        out = self.layers(x)
        out = F.softmax(out, dim=1)
        return out

class Loss(nn.Module):
    def __init__(self, num_classes):
        super(Loss, self).__init__()
        self.num_classes = num_classes

    def forward(self, pred, target):
        loss = F.cross_entropy(pred.view(-1, self.num_classes), target.view(-1))
        return loss

def main():
    dataset = NumberDataset()
    input_shape = dataset.input_shape
    num_inputs = 1
    for x in input_shape:
        num_inputs *= x

    num_classes = 10
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=10, shuffle=True, num_workers=1)

    activations = {
        'relu': nn.ReLU(True),
        'leaky': nn.LeakyReLU(True),
        'selu': nn.SELU(True),
        'sigmoid': nn.Sigmoid()
    }
    num_layers = [10]
    num_nodes = [50, 100]
    epochs = 1000
    hyperparams = [
        {'lr': 0.1, 'momentum': 0.0},
        {'lr': 0.1, 'momentum': 0.9},
        {'lr': 0.01, 'momentum': 0.0},
        {'lr': 0.01, 'momentum': 0.9},
                   ]
    # architecture
    for layers in num_layers:
        for nodes in num_nodes:
            for bn in range(2):
                print(f'layers: {layers},\tnodes: {nodes},\tbn: {bn}')
                for hyper in hyperparams:
                    print(f'\thyperparams(lr: {hyper["lr"]}, momentum: {hyper["momentum"]})')
                    for key in activations.keys():
                        model = Network(num_input=num_inputs, num_layers=layers, num_nodes=nodes, num_classes=num_classes, activation=activations[key], use_batch_norm=bn)
                        loss_fn = Loss(num_classes)
                        optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.0)

                        lambda2 = lambda epoch: 0.95 ** epoch
                        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=[lambda2])
                        train_loss = 0
                        mean_time = 0
                        for epoch in range(epochs):

                            start_time = timeit.default_timer()
                            model.train()
                            for idx, (input, target) in enumerate(train_loader):
                                optimizer.zero_grad()
                                pred = model(input)
                                train_loss = loss_fn(pred, target)
                                train_loss.backward()
                                optimizer.step()
                            model.eval()
                            scheduler.step()

                            elapsed_time = timeit.default_timer() - start_time
                            mean_time += elapsed_time
                        mean_time /= epochs
                        # print('activation: ', key, ' use bn: ', bn)
                        # print('mean_time: %.3f ms' % (mean_time*1000))
                        # print('evaluate model')
                        tp = 0
                        n = 0

                        model.eval()
                        with torch.no_grad():
                            for (input, target) in dataset:
                                pred = model(input.unsqueeze(0))
                                class_pred = torch.argmax(pred, dim=1)
                                # print("pred ", class_pred, pred[0][class_pred], " target: ", target)
                                if class_pred == target:
                                    tp += 1
                                n += 1
                        accuracy = tp / n
                        print(f'\t\tactivation: {key}\t\t\taccuracy: {accuracy:.1f}, \tmean_time: {mean_time*1000 :.3f} ms')
                    # print('accuracy: ', accuracy)
    # state = {
    #     'model': model.state_dict(),
    #     'loss': train_loss
    # }
    # torch.save(state, 'net.pth')

if __name__ == '__main__':
    main()
