from model import TransformerModel
from options import Options
from data import FakeDataGenerator, FakeDataSet

import torch
from torch.nn import MSELoss
from torch.optim import AdamW
from torch.utils.data import DataLoader, random_split
import numpy as np
import matplotlib.pyplot as plt

import argparse
from time import time
from itertools import tee
import os

def add_options_to_parser(parser):
	dummy_options = Options()
	for name in dummy_options.get_option_names():
		default_value = getattr(dummy_options, name)
		if type(default_value) in [str, int, float, bool]:
			parser.add_argument("--" + name, type=type(default_value), default=default_value)

def update_options_from_args(options, args):
	for arg in vars(args):
		setattr(options, arg, getattr(args, arg))

def mse(x, y):
	return np.mean((x-y)**2)

def main(args):
	
	options = Options()
	update_options_from_args(options, args)

	if not options.seed:
		options.seed = int(time() * 10000) % (2**32 - 1)

	torch.manual_seed(options.seed)
	np.random.seed(options.seed)

	options.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	print(f"Running on: {options.device}")

	data_generator = FakeDataGenerator(options)
	transformer = TransformerModel(options).to(options.device)
	print(f"{transformer}")
	print(f"Num parameters: {sum([p.numel() for p in transformer.parameters()]):,}")

	criterion = MSELoss()
	optimizer = AdamW(params=transformer.parameters(), lr=options.learning_rate)

	num_samples = options.num_samples
	random_frequencies, training_data_generator, validation_data_generator = data_generator.get_data_generators(num_samples=num_samples)

	batch_size = options.batch_size

	fake_data = FakeDataSet(options)
	fake_data_len = len(fake_data)
	training_data, validation_data = random_split(fake_data, [int(fake_data_len * 0.8), int(fake_data_len * 0.2)])
	loaded_training_data = DataLoader(training_data, batch_size=batch_size, shuffle=True)
	loaded_validation_data = DataLoader(validation_data, batch_size=batch_size, shuffle=False)

	test_sample = torch.unsqueeze(validation_data[np.random.randint(0, len(loaded_validation_data))], 0).permute(1, 0, 2)
	test_sample_cpu = test_sample.to(torch.device("cpu")).numpy()[:, 0, 0]

	artifact_dir = options.artifact_dir
	os.makedirs(artifact_dir, exist_ok=True)

	batches_per_epoch = len(loaded_training_data) // batch_size
	current_epoch = 1
	plt.figure(figsize=(16,9), dpi=160)
	for current_epoch in range(1, options.num_epochs + 1):
		transformer.train()
		
		num_train_predictions = 0
		num_val_predictions = 0
		total_train_loss = 0
		total_val_loss = 0

		for train_batch in loaded_training_data:
			num_train_predictions += 1
			optimizer.zero_grad()

			train_batch = train_batch.permute(1, 0, 2)
			predictions = transformer(train_batch)
			loss = criterion(input=predictions.permute(1, 0, 2), target=train_batch.permute(1, 0, 2))
			total_train_loss += loss.item()

			loss.backward()
			torch.nn.utils.clip_grad_norm_(transformer.parameters(), 0.7)
			optimizer.step()

		with torch.no_grad():
			for val_batch in loaded_validation_data:
				num_val_predictions += 1

				val_batch = val_batch.permute(1, 0, 2)
				val_predictions = transformer(val_batch)
				val_loss = criterion(input=val_predictions.permute(1, 0, 2), target=val_batch.permute(1, 0, 2))
				total_val_loss += val_loss.item()

		print(f"Epoch: {current_epoch:03d} | Train loss: {total_train_loss / num_train_predictions:.05f}"
		f" | Val loss: {total_val_loss / num_val_predictions:.05f}")
		total_train_loss = 0.0
		total_val_loss = 0.0

		with torch.no_grad():
			transformer.eval()
			predicted = transformer(test_sample).to(torch.device("cpu")).permute(1, 0, 2).numpy()[0, :, 0]
			print(f"Test loss: {mse(predicted, test_sample_cpu)}")
			
			plt.plot(test_sample_cpu, label="Original")
			plt.plot(predicted, label="Reconstructed")
			
			gca = plt.gca()
			gca.set_xlim([-5, len(test_sample_cpu) + 5])
			gca.set_ylim([-1, 1])

			plt.legend()
			plt.savefig(os.path.join(artifact_dir, f"test_sample_{current_epoch:03d}.png"))
			plt.clf()

if __name__ == "__main__":
	parser = argparse.ArgumentParser(description='Short sample app')

	add_options_to_parser(parser)
	
	args = parser.parse_args()
	main(args)