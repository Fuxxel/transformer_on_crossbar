from model import TransformerModel
from options import Options
from data import FakeDataGenerator

import torch
from torch.nn import MSELoss
from torch.optim import AdamW
import numpy as np

import argparse
from time import time
from itertools import tee

def add_options_to_parser(parser):
	dummy_options = Options()
	for name in dummy_options.get_option_names():
		default_value = getattr(dummy_options, name)
		if type(default_value) in [str, int, float, bool]:
			parser.add_argument("--" + name, type=type(default_value), default=default_value)

def update_options_from_args(options, args):
	for arg in vars(args):
		setattr(options, arg, getattr(args, arg))

def main(args):
	
	options = Options()
	update_options_from_args(options, args)

	if not options.seed:
		options.seed = int(time() * 10000) % (2**32 - 1)

	torch.manual_seed(options.seed)
	np.random.seed(options.seed)

	options.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

	data_generator = FakeDataGenerator(options)
	transformer = TransformerModel(options).to(options.device)
	print(f"Num parameters: {sum([p.numel() for p in transformer.parameters()]):,}")

	criterion = MSELoss()
	optimizer = AdamW(params=transformer.parameters(), lr=options.learning_rate)

	num_samples = options.num_samples
	random_frequencies, training_data_generator, validation_data_generator = data_generator.get_data_generators(num_samples=num_samples)

	transformer.train()
	total_train_loss = 0.0
	total_val_loss = 0.0

	batch_size = options.batch_size
	num_epochs = options.num_epochs
	
	iterators = tee(zip(training_data_generator, validation_data_generator), num_epochs)
	for current_epoch in range(num_epochs):
		num_samples = 0
		for (train_x, train_y), (val_x, val_y) in iterators[current_epoch]:
			optimizer.zero_grad()

			predictions = transformer(train_x)
			loss = criterion(input=predictions, target=train_y)
			total_train_loss += loss.item()

			loss.backward()
			optimizer.step()

			with torch.no_grad():
				val_predictions = transformer(val_x)
				val_loss = criterion(input=val_predictions, target=val_y)
				total_val_loss += val_loss.item()
			
			num_samples += 1

		print(f"Epoch: {current_epoch + 1:03d} | Train loss: {total_train_loss / num_samples:.05f}"
			f" | Val loss: {total_val_loss / num_samples:.05f}")
		total_train_loss = 0.0
		total_val_loss = 0.0


if __name__ == "__main__":
	parser = argparse.ArgumentParser(description='Short sample app')

	add_options_to_parser(parser)
	
	args = parser.parse_args()
	main(args)