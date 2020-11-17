### Copy paste from https://github.com/mateuszbuda/brain-segmentation-pytorch/blob/master/unet.py

from collections import OrderedDict

import argparse
import sys
import torch
import torchvision
import torchvision.transforms as transforms

from typing import Tuple, List

import sambaflow
import sambaflow.samba as samba
import sambaflow.samba.nn as nn
import sambaflow.samba.optim as optim
import sambaflow.samba.utils as utils

from sambaflow.samba.sambatensor import SambaTensor
from sambaflow.samba.utils.argparser import parse_app_args
from sambaflow.samba.utils.dataset.mnist import dataset_transform


class UNet(nn.Module):

    def __init__(self, in_channels=3, out_channels=20, init_features=32):
        super(UNet, self).__init__()

        features = init_features
        self.encoder1 = UNet._block(in_channels, features, name="enc1")
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder2 = UNet._block(features, features * 2, name="enc2")
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder3 = UNet._block(features * 2, features * 4, name="enc3")
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder4 = UNet._block(features * 4, features * 8, name="enc4")
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.bottleneck = UNet._block(features * 8, features * 16, name="bottleneck")

        self.upconv4 = nn.ConvTranspose2d(
            features * 16, features * 8, kernel_size=2, stride=2
        )
        self.decoder4 = UNet._block((features * 8) * 2, features * 8, name="dec4")
        self.upconv3 = nn.ConvTranspose2d(
            features * 8, features * 4, kernel_size=2, stride=2
        )
        self.decoder3 = UNet._block((features * 4) * 2, features * 4, name="dec3")
        self.upconv2 = nn.ConvTranspose2d(
            features * 4, features * 2, kernel_size=2, stride=2
        )
        self.decoder2 = UNet._block((features * 2) * 2, features * 2, name="dec2")
        self.upconv1 = nn.ConvTranspose2d(
            features * 2, features, kernel_size=2, stride=2
        )
        self.decoder1 = UNet._block(features * 2, features, name="dec1")

        self.conv = nn.Conv2d(
            in_channels=features, out_channels=out_channels, kernel_size=1
        )

    def forward(self, x):
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        enc3 = self.encoder3(self.pool2(enc2))
        enc4 = self.encoder4(self.pool3(enc3))

        bottleneck = self.bottleneck(self.pool4(enc4))

        dec4 = self.upconv4(bottleneck)
        dec4 = samba.cat((dec4, enc4), dim=1)
        dec4 = self.decoder4(dec4)
        dec3 = self.upconv3(dec4)
        dec3 = samba.cat((dec3, enc3), dim=1)
        dec3 = self.decoder3(dec3)
        dec2 = self.upconv2(dec3)
        dec2 = samba.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)
        dec1 = self.upconv1(dec2)
        dec1 = samba.cat((dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1)
        return samba.sigmoid(self.conv(dec1))

    @staticmethod
    def _block(in_channels, features, name):
        return nn.Sequential(
            OrderedDict(
                [
                    (
                        name + "conv1",
                        nn.Conv2d(
                            in_channels=in_channels,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=False,
                        ),
                    ),
                    (name + "norm1", nn.BatchNorm2d(num_features=features)),
                    (name + "relu1", nn.ReLU(inplace=True)),
                    (
                        name + "conv2",
                        nn.Conv2d(
                            in_channels=features,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=False,
                        ),
                    ),
                    (name + "norm2", nn.BatchNorm2d(num_features=features)),
                    (name + "relu2", nn.ReLU(inplace=True)),
                ]
            )
        )

def train(args: argparse.Namespace, model: nn.Module, optimizer: optim.SGD) -> None:
    # Train the model on RDU using the MNIST dataset (images and labels)
    train_dataset = torchvision.datasets.VOCSegmentation(root=f'{args.data_folder}',
                                               train=True,
                                               transform=dataset_transform(args),
                                               download=True)
    test_dataset = torchvision.datasets.VOCSegmentation(root=f'{args.data_folder}',
                                              train=False,
                                              transform=dataset_transform(args))

    # Data loader (input pipeline)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=args.batch_size, shuffle=False)

    # Train the model
    total_step = len(train_loader)
    for epoch in range(args.num_epochs):
        avg_loss = 0
        for i, (images, labels) in enumerate(train_loader):
            sn_images = samba.from_torch(images, name='image', batch_dim=0)
            sn_labels = samba.from_torch(labels, name='label', batch_dim=0)
            loss, outputs = model.run(sn_images, sn_labels)
            loss, outputs = samba.to_torch(loss), samba.to_torch(outputs)
            avg_loss += loss.mean()

            if (i + 1) % 10000 == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(epoch + 1, args.num_epochs, i + 1,
                                                                         total_step,
                                                                         avg_loss / (i + 1)))

        model.write_host_params()
        with torch.no_grad():
            correct = 0
            total = 0
            total_loss = 0
            for images, labels in test_loader:
                loss, outputs = model(images, labels)
                loss, outputs = samba.to_torch(loss), samba.to_torch(outputs)
                total_loss += loss.mean()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum()

            print('Test Accuracy: {} %'.format(100 * correct / total), ' Loss: ',
                  total_loss.item() / (len(test_loader)))

def add_args(parser: argparse.ArgumentParser):
    parser.add_argument('-e', '--num-epochs', type=int, default=1)
    parser.add_argument('--num-features', type=int, default=784)
    parser.add_argument('--num-classes', type=int, default=10)

def add_run_args(parser: argparse.ArgumentParser):
    parser.add_argument('--data-folder',
                        type=str,
                        default='mnist_data',
                        help="The folder to download the MNIST dataset to.")

def test(args: argparse.Namespace, model: nn.Module, inputs: Tuple[samba.SambaTensor],
         optimizer: optim.SGD) -> None:
    # Run a numerical check to ensure CPU and RDU runs are consistent.
    # Run CPU version of model.
    outputs_gold = model(*inputs)
    # Run RDU version of model.
    outputs_samba = model.run(*inputs)

    # check that all samba and torch outputs match numerically
    for i, (output_samba, output_gold) in enumerate(zip(outputs_samba, outputs_gold)):
        utils.assert_close(output_samba, output_gold, f'forward output #{i}', threshold=3e-3,
                           visualize=args.visualize)

    # training mode, check two of the gradients
    torch_loss, torch_gemm_out = outputs_gold
    torch_loss.mean().backward()

    # we choose two gradients from different places to test numerically
    gemm1_grad_gold = model.lin_layer.weight.grad
    gemm1_grad_samba = model.lin_layer.weight.sn_grad
    utils.assert_close(gemm1_grad_gold,
                       gemm1_grad_samba,
                       'lin_layer__weight__grad',
                       threshold=3e-3,
                       visualize=args.visualize)

def test_spatial(args: argparse.Namespace, model: nn.Module, inputs: Tuple[samba.SambaTensor], optimizer: optim.SGD,
                 mini_batch_size: int) -> None:
    # Similiar to test
    # Run CPU version of model
    for idx in range(args.num_spatial_batches):
        start_idx, end_idx = mini_batch_size * idx, mini_batch_size * (idx + 1)
        input_slice = (inputs[:, start_idx:end_idx], inputs[start_idx:end_idx, :])
        gold_loss, gold_output = model(*input_slice)
        gold_loss.mean().backward()
        optimizer.step()
    gold_weight = model.lin_layer.weight
    # Run RDU version of model
    model.run(*inputs)
    utils.assert_close(gold_weight.sn_data, gold_weight, 'weight',
                       0.01)  # FIXME(tanl) Checking the correctness based on one iteration seems not right

def main(argv: List[str]):
    # Set random seed for reproducibility.
    torch.manual_seed(256)

    # Get common args and any user added args.
    args = parse_app_args(argv=argv, common_parser_fn=add_args, run_parser_fn=add_run_args)

    # Instantiate the model.
    model = LogReg(args.num_features, args.num_classes)

    # Instantiate a optimizer.
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0, weight_decay=1e-2)

    # Dummy inputs required for tracing.
    inputs = (samba.randn(1, args.num_features, name='image',
                          batch_dim=0), samba.randint(args.num_classes, (1,), name='label', batch_dim=0))

    if args.command == "compile":
        # Run model analysis and compile, this step will produce a PEF.
        utils.compile(args, model, inputs, optimizer, name='logreg', app_dir=utils.get_file_dir(__file__))
    elif args.command == "test":
        # Run a numerical check comparing CPU to RDU.
        if args.num_spatial_batches > 1:
            # spatial
            mini_batch_size = args.batch_size
            args.batch_size *= args.num_spatial_batches
            inputs = (SambaTensor(samba.to_torch(inputs[0]).repeat(args.num_spatial_batches, 1),
                                  name='image',
                                  batch_dim=0),
                      SambaTensor(samba.to_torch(inputs[1]).repeat(args.num_spatial_batches), name='label',
                                  batch_dim=0))
            utils.trace_graph(args, model, inputs, optimizer)
            test_spatial(args, model, inputs, optimizer, mini_batch_size)
        else:
            # section by section
            utils.trace_graph(args, model, inputs, optimizer)
            test(args, model, inputs, optimizer)
    elif args.command == "measure-performance":
        # Get inference latency and throughput statistics
        utils.trace_graph(args, model, inputs, optimizer)
        utils.measure_performance(model,
                                  inputs,
                                  args.batch_size,
                                  run_graph_only=args.run_graph_only,
                                  n_iterations=args.num_iterations,
                                  json=args.json)
    elif args.command == "run":
        # Train Logreg model
        utils.trace_graph(args, model, inputs, optimizer)
        train(args, model, optimizer)

if __name__ == '__main__':
    main(sys.argv[1:])