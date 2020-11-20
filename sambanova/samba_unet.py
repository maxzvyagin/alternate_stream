#!/usr/bin/python
# encoding: utf-8

# Copyright © 2020 by SambaNova Systems, Inc. Disclosure, reproduction,
# reverse engineering, or any other use made without the advance written
# permission of SambaNova Systems, Inc. is unauthorized and strictly
# prohibited. All rights of ownership and enforcement are reserved.

import argparse
import numpy as np
import os
import sys
from typing import Tuple, List

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

import sambaflow
import sambaflow.samba as samba
import sambaflow.samba.nn as sn
import sambaflow.samba.utils as sn_utils

from sambaflow.mac.metadata import TilingMetadata
from sambaflow.samba import to_torch
from sambaflow.samba.utils import assert_close, set_seed
from sambaflow.samba.utils.argparser import parse_app_args
from sambaflow.samba.utils.common import common_app_driver
from sambaflow.models.unet import UNet
from sambaflow.samba import SambaTensor
from sambaflow.samba.experiment_logger import AccMeter, DisplayMeter, CountMeter, Logger

from unet_utils.dataset import BrainSegmentationDataset as Dataset
from unet_utils.utils import DiceLoss, transforms

import pickle

def add_args(parser: argparse.ArgumentParser):
    parser.add_argument('--in-height', type=int, default=32, help='Height of the input image')
    parser.add_argument('--in-width', type=int, default=32, help='Width of the input image')
    parser.add_argument('--in-channels', type=int, default=3, help='Number of channels in the input image')
    parser.add_argument('--out-channels', type=int, default=1, help='Number of channels in the output image')
    parser.add_argument('--init-features', type=int, default=32, help='Number of initial features for first conv')
    parser.add_argument('--enable-tiling', type=bool, default=False, help='Enable DRAM tiling')
    parser.add_argument('--num-row-tiles', type=int, default=2, help='If tiling is enabled, number of row tiles')
    parser.add_argument('--num-col-tiles', type=int, default=2, help='If tiling is enabled, num of col tiles')
    parser.add_argument('--host-row-padding',
                        type=int,
                        default=2,
                        help='If tiling is enabled, amount of row padding to add on host')
    parser.add_argument('--host-col-padding',
                        type=int,
                        default=2,
                        help='If tiling is enabled, amount of col padding to add on host')
    parser.add_argument('--default-par-factors', action="store_true")
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="number of epochs to train (default: 100)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.0001,
        help="initial learning rate (default: 0.001)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=0,
        help="number of workers for data loading (default: 1)",
    )
    parser.add_argument("--images", type=str, default="./kaggle_3m", help="root folder with images")
    parser.add_argument(
        "--image-size",
        type=int,
        default=32,
        help="target input image size (default: 256)",
    )
    parser.add_argument(
        "--aug-scale",
        type=int,
        default=0.05,
        help="scale factor range for augmentation (default: 0.05)",
    )
    parser.add_argument(
        "--aug-angle",
        type=int,
        default=15,
        help="rotation angle range in degrees for augmentation (default: 15)",
    )
    parser.add_argument('--ch', action="store_true")
    parser.add_argument('--model-dir',
                        type=str,
                        default='sambaflow/apps/image/samba/unet/model_dir',
                        help='location for training outputs and checkpoints')
    parser.add_argument('--use-real-data', action="store_true", help="Run the training on real data")
    parser.add_argument('--run-backward', action="store_true", help="Run backward")


def get_inputs(args: argparse.Namespace, tiled=False) -> Tuple[samba.SambaTensor]:
    # If tiling is enabled, we trace the graph with the tile size, and pass the original size through tiling metadata
    if tiled:
        assert args.in_height % args.num_row_tiles == 0, "Input image is not evenly divisible by row tiles"
        assert args.in_width % args.num_col_tiles == 0, "Input image is not evenly divisible by col tiles"
        height = args.in_height // args.num_row_tiles
        width = args.in_width // args.num_col_tiles
    else:
        height = args.in_height
        width = args.in_width

    return (samba.randn(args.batch_size, args.in_channels, height, width, name='input', batch_dim=0,
                        requires_grad=True), )


def prepare_for_tiling(args: argparse.Namespace, padded_input_shape: samba.SambaTensor) -> None:
    """
    Change the shapes of inputs and input grad. This is necessary because tiling requires padding on the host.
    The input shapes are different when they are traced.
    """
    assert len(samba.session.tensor_dict['inputs']) == 1, "Only one input expected"

    # Modify the shapes of the input and input grad tensor.
    samba.session.tensor_dict['inputs']['input'] = samba.SambaTensor(torch.zeros(padded_input_shape),
                                                                     name='input',
                                                                     batch_dim=0)

    if not args.inference:
        samba.session.tensor_dict['grads']['input__grad'] = samba.SambaTensor(torch.zeros(padded_input_shape),
                                                                              name='input_grad',
                                                                              batch_dim=0)


def test(args: argparse.Namespace, inputs: Tuple[samba.SambaTensor], model: sn.Module) -> None:
    if args.enable_tiling:
        test_inputs = get_inputs(args, False)

        # TODO(tejasn): Host padding of input tensor. This should not be an input from the test but should be
        # computed within tiled conv utils.
        padded_input = F.pad(
            samba.to_torch(test_inputs[0]),
            [args.host_col_padding, args.host_col_padding, args.host_row_padding, args.host_row_padding])

        padded_input = samba.SambaTensor(padded_input, name='input', batch_dim=0)
        prepare_for_tiling(args, padded_input.shape)

        samba.session.to_device()
        gold_outputs = model(*test_inputs)
        inputs = (padded_input, )
        samba_outputs = model.run(inputs)[0]
    else:
        samba.session.to_device()
        gold_outputs = model(*inputs)
        samba_outputs = model.run(inputs)[0]

    print(f'gold_output_abs_sum: {gold_outputs.abs().sum()}', gold_outputs)
    print(f'samba_output_abs_sum: {samba_outputs.abs().sum()}', samba_outputs)
    sn_utils.assert_close(samba_outputs, gold_outputs, 'output', threshold=3e-3)

    if args.run_backward:
        output_grad = samba.randn_like(gold_outputs)
        gold_outputs.backward(output_grad)
        model.run(inputs, grad_of_outputs=[output_grad])[0]

        # checking input grad and last grad
        if args.enable_tiling:
            gold_input_grad = test_inputs[0].grad
        else:
            gold_input_grad = inputs[0].grad
        samba_input_grad = inputs[0].sn_grad

        # Unpadding the image for the tiled version before comparing.
        if args.enable_tiling:
            samba_input_grad = samba.from_torch(
                samba.to_torch(samba_input_grad)
                [:, :, args.host_col_padding:-args.host_col_padding, args.host_row_padding:-args.host_row_padding])

        print(f'gold_input_grad_abs_sum: {gold_input_grad.abs().sum()}')
        print(f'samba_input_grad_abs_sum: {samba_input_grad.data.abs().sum()}')
        sn_utils.assert_close(gold_input_grad, samba_input_grad, 'input_grad', threshold=0.05, visualize=args.visualize)

        gold_weight_grad = model.down1.two_conv.conv1.weight.grad
        samba_weight_grad = model.down1.two_conv.conv1.weight.sn_grad

        print(f'gold_weight_grad_abs_sum: {gold_weight_grad.abs().sum()}')
        print(f'samba_weight_grad_abs_sum: {samba_weight_grad.abs().sum()}')
        # TODO(tejasn): The weight grad mismatch for the tiling version is large. The mismatch will be much lower once
        # we add support for 9-tile version.
        sn_utils.assert_close(gold_weight_grad,
                              samba_weight_grad,
                              'weight_grad',
                              threshold=3.1 if args.enable_tiling else 0.22,
                              rtol=0.1,
                              visualize=args.visualize)


def tiled_compile(args: argparse.Namespace,
                  model: sn.Module,
                  inputs: Tuple[samba.SambaTensor],
                  optim: samba.optim.SGD = None,
                  name: str = None,
                  squeeze_bs_dim=False,
                  app_dir=None) -> str:

    metadata = dict()
    original_size = [args.batch_size, args.in_channels, args.in_height, args.in_width]
    dummy_input = torch.randn(original_size)

    metadata[TilingMetadata.key] = TilingMetadata(original_size=original_size,
                                                  num_tiles=[args.batch_size, args.num_row_tiles, args.num_col_tiles])

    return samba.session.compile(model,
                                 inputs,
                                 optim,
                                 name=name,
                                 squeeze_bs_dim=squeeze_bs_dim,
                                 app_dir=app_dir,
                                 metadata=metadata,
                                 config_dict=vars(args))


def train(args: argparse.Namespace, model: sn.Module, optimizer, ch=False):
    if args.enable_tiling and ch:
        padded_input_shape = [
            args.batch_size, args.in_channels, args.in_height + 2 * args.host_row_padding,
            args.in_width + 2 * args.host_col_padding
        ]
        prepare_for_tiling(args, padded_input_shape)

    if ch:
        samba.session.to_device()

    loader_train, loader_valid = data_loaders(args)
    logger = get_loggers()
    dsc_loss = DiceLoss()

    step = 0
    best_validation_dsc = 0.0

    pbar = tqdm(range(args.epochs), total=args.epochs, dynamic_ncols=True)

    for epoch in pbar:
        model.train()

        for batch_index, (x, y) in enumerate(loader_train):

            if not ch:
                model.zero_grad()
                gold = model(x)
                loss = dsc_loss(samba.to_torch(gold), y)
                loss.backward()
                optimizer.step()

            else:
                ### Run dice_loss in torch
                section_types = ['fwd']

                # Pad the input if tiling is enabled.
                if args.enable_tiling:
                    x = F.pad(
                        x, [args.host_col_padding, args.host_col_padding, args.host_row_padding, args.host_row_padding])
                xs = samba.from_torch(x)
                samba_outputs = model.run((samba.from_torch(x), ), section_types=section_types)[0]
                samba_outputs.requires_grad_(True)
                loss = dsc_loss(samba.to_torch(samba_outputs), y)

                loss.backward()
                section_types = ['bckwd', 'opt']
                samba.session.ctx.set_tensors({"sigmoid__outputs__0__grad": samba_outputs.grad.numpy()})
                model.run((samba.from_torch(x), ), section_types=section_types)

            logger.update({"loss": loss.mean().item()})
            logger.dump(args.model_dir, logger.meters.keys())
            logger.update({"train_step": 1})
            pbar.set_description(str(logger))
            step += 1


def get_loggers():
    """
    Helpers to return the train and test loggers.
    """
    train_logger_dict = {
        'train_step': CountMeter(),
        'loss': DisplayMeter(format="0.8f"),
    }
    train_logger = Logger(train_logger_dict)
    return train_logger


def data_loaders(args):
    dataset_train, dataset_valid = datasets(args)

    loader_train = DataLoader(dataset_train,
                              batch_size=args.batch_size,
                              shuffle=True,
                              drop_last=True,
                              num_workers=args.workers)

    loader_valid = DataLoader(dataset_valid, batch_size=args.batch_size, drop_last=False, num_workers=args.workers)

    return loader_train, loader_valid


def datasets(args):
    f = open("pt_gis_rgb.pkl")
    train, valid = pickle.load(f)
    return train, valid
    # train = Dataset(
    #     images_dir=args.images,
    #     subset="train",
    #     image_size=args.image_size,
    #     transform=transforms(scale=args.aug_scale, angle=args.aug_angle, flip_prob=0.5),
    # )
    # valid = Dataset(
    #     images_dir=args.images,
    #     subset="validation",
    #     image_size=args.image_size,
    #     random_sampling=False,
    # )
    # return train, valid



def main(argv: List[str]):
    # Set random seed for reproducibility.
    sn_utils.set_seed(256)

    # Get common args and any user added args.
    args = parse_app_args(argv=argv, common_parser_fn=add_args)

    # Dummy inputs required for tracing.
    inputs = get_inputs(args)
    print(f'Inputs: {[d.shape for d in inputs]}')

    # Instantiate the model.
    model = UNet(in_channels=args.in_channels, out_channels=args.out_channels, init_features=args.init_features)

    # Instantiate a optimizer.
    optim = samba.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.997),
                              weight_decay=0) if not args.inference else None
    if not args.inference: inputs[0].requires_grad_(True)

    if args.inference:
        model.eval()

    if args.command == 'compile':
        inputs = get_inputs(args, args.enable_tiling)
        if args.enable_tiling:
            tiled_compile(args,
                          model,
                          inputs,
                          optim,
                          name='Unet',
                          squeeze_bs_dim=True,
                          app_dir=sn_utils.get_file_dir(__file__))
        else:
            samba.session.compile(model,
                                  inputs,
                                  optim,
                                  name='Unet',
                                  squeeze_bs_dim=True,
                                  app_dir=sn_utils.get_file_dir(__file__),
                                  config_dict=vars(args))
    elif args.command == 'test':
        # Test numerical correctness between CPU and RDU
        sn_utils.trace_graph(model, inputs, optim, squeeze_bs_dim=True, transfer_device=False, config_dict=vars(args))
        test(args, inputs, model)
    elif args.command == "measure-performance":
        # Get inference latency and throughput statistics
        sn_utils.trace_graph(model, inputs, optim, squeeze_bs_dim=True, config_dict=vars(args))
        sn_utils.measure_performance(model,
                                     inputs,
                                     args.batch_size,
                                     run_graph_only=args.run_graph_only,
                                     n_iterations=args.num_iterations,
                                     json=args.json,
                                     data_parallel=args.data_parallel,
                                     world_size=args.world_size,
                                     reduce_on_rdu=args.reduce_on_rdu,
                                     min_duration=args.min_duration)
    elif args.command == "run":
        # Train the model
        sn_utils.trace_graph(model, inputs, optim, squeeze_bs_dim=True, transfer_device=False, config_dict=vars(args))
        train(args, model, optim, args.ch)

    else:
        # This section of the code will be removed soon for packaging purposes (use --debug in your commands to invoke all remaining functions)
        common_app_driver(args,
                          model,
                          inputs,
                          optim,
                          'Unet',
                          squeeze_bs_dim=True,
                          app_dir=sn_utils.get_file_dir(__file__))


if __name__ == '__main__':
    main(sys.argv[1:])
