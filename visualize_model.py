from model import TransformerVisual
from options import Options

import torch
from torch.nn import MSELoss, CrossEntropyLoss
from torch.optim import AdamW
from torch.utils.data import DataLoader # random_split
from torch._utils import _accumulate
from torch import randperm
import torch.onnx

import numpy as np
import matplotlib.pyplot as plt

import argparse
import os

def update_options_from_args(options, args):
	for arg in vars(args):
		setattr(options, arg, getattr(args, arg))

def add_options_to_parser(parser):
	dummy_options = Options()
	for name in dummy_options.get_option_names():
		default_value = getattr(dummy_options, name)
		if type(default_value) in [str, int, float]:
			parser.add_argument("--" + name, type=type(default_value), default=default_value)
		elif type(default_value) == bool:
			parser.add_argument("--" + name, action="store_false" if default_value else "store_true")

def main(args):
	options = Options()
	update_options_from_args(options, args)

	transformer_model = TransformerVisual(options)
	dummy_input = torch.randn(options.window_size, options.batch_size, options.num_input_features)
	torch.onnx.export(transformer_model, dummy_input, "transformer_model.onnx")

if __name__ == "__main__":
	parser = argparse.ArgumentParser(description='Transformer on crossbar')

	add_options_to_parser(parser)
	
	args = parser.parse_args()
	main(args)