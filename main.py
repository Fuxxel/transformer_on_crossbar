from model import TransformerModel, TransformerClassifier
from options import Options
from data import FakeDataSet, CoinDataSetPreparer, CoinDataSubset

import torch
from torch.nn import MSELoss, CrossEntropyLoss
from torch.optim import AdamW
from torch.utils.data import DataLoader # random_split
from torch._utils import _accumulate
from torch import randperm
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from collections import Counter
import argparse
from time import time
import os
from datetime import datetime

def random_split(dataset, lengths):
	if sum(lengths) != len(dataset):
		raise ValueError("Sum of input lengths does not equal the length of the input dataset!")

	indices = randperm(sum(lengths)).tolist()
	return [CoinDataSubset(dataset.get_options(), dataset.preloaded_references, indices[offset - length:offset]) for offset, length in zip(_accumulate(lengths), lengths)]

def add_options_to_parser(parser):
	dummy_options = Options()
	for name in dummy_options.get_option_names():
		default_value = getattr(dummy_options, name)
		if type(default_value) in [str, int, float]:
			parser.add_argument("--" + name, type=type(default_value), default=default_value)
		elif type(default_value) == bool:
			parser.add_argument("--" + name, action="store_false" if default_value else "store_true")

def update_options_from_args(options, args):
	for arg in vars(args):
		setattr(options, arg, getattr(args, arg))

def mse(x, y):
	return np.mean((x-y)**2)

def classification_accuracy(input, target):
	return (input.argmax(-1) == target).float().mean()

def write_options_to_file(options, path, additional_info=None):
	print("--------------------")
	print("Options:")
	with open(path, "w") as out_file:
		for name in options.get_option_names():
			value = getattr(options, name)
			if type(value) in [str, int, float, bool]:
				out_file.write(f"{name}:{value}\n")
				print(f"{name}:{value}")

		if additional_info:
			for name, value in additional_info.items():
				out_file.write(f"{name}:{value}\n")
				print(f"{name}:{value}")
	print("--------------------")

def index_to_coin(index):
	coins = [1, 2, 5, 20, 50, 100, 200]
	return coins[index]

def main(args):
	options = Options()
	update_options_from_args(options, args)

	if not options.seed:
		options.seed = int(time() * 10000) % (2**32 - 1)

	torch.manual_seed(options.seed)
	np.random.seed(options.seed)

	options.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	print(f"Running on: {options.device}")

	transformer = TransformerModel(options).to(options.device) if options.problem == "fake" else TransformerClassifier(options).to(options.device)
	print(f"{transformer}")
	print(f"Num parameters: {sum([p.numel() for p in transformer.parameters()]):,}")
	
	optimizer = AdamW(params=transformer.parameters(), lr=options.learning_rate)

	batch_size = options.batch_size

	if options.problem == "coin":
		timestamp = datetime.now().strftime("%y_%m_%d_%H_%M_%S_%f")
		model_save_path = os.path.join(options.artifact_dir, timestamp)
		print(f"Save path: {model_save_path}")
		os.makedirs(model_save_path, exist_ok=False)
		write_options_to_file(options, os.path.join(model_save_path, "options.txt"), additional_info={
			"num_parameters": sum([p.numel() for p in transformer.parameters()])
		})
		print("Sampling individual timesteps" if options.sample_individual_timesteps else "Collecting multiple timesteps into embedding vector")
		
		criterion = CrossEntropyLoss()

		coin_data = CoinDataSetPreparer(options)
		coin_data_len = len(coin_data)
		references = coin_data.preloaded_references

		train_len = round(coin_data_len * 0.7)
		val_len = round(coin_data_len * 0.15)
		test_len = coin_data_len - val_len - train_len
		print(f"Training set length: {train_len}, validation set length: {val_len}, test set length: {test_len}")
		training_data, validation_data, test_data = random_split(coin_data, [train_len, val_len, test_len])
		print(f"After expanding samples: Training set length: {len(training_data)}, validation set length: {len(validation_data)}, test set length: {len(test_data)}")
		
		os.makedirs(os.path.join(model_save_path, "used_data"))
		training_data.save_used_indices(os.path.join(model_save_path, "used_data", "training.txt"))
		validation_data.save_used_indices(os.path.join(model_save_path, "used_data", "validation.txt"))
		test_data.save_used_indices(os.path.join(model_save_path, "used_data", "test.txt"))
		
		loaded_training_data = DataLoader(training_data, batch_size=min(batch_size, len(training_data)) , shuffle=True, drop_last=True)
		loaded_validation_data = DataLoader(validation_data, batch_size=min(batch_size, len(validation_data)), shuffle=False, drop_last=True)
		loaded_test_data = DataLoader(test_data, batch_size=min(batch_size, len(test_data)), shuffle=False, drop_last=True)

		batches_per_epoch_train = len(loaded_training_data)
		batches_per_epoch_val = len(loaded_validation_data)
		batches_per_epoch_test = len(loaded_test_data)
		current_epoch = 1
		best_val_acc = 0
		path_to_best_model_weights = None
		for current_epoch in range(1, options.num_epochs + 1):
			transformer.train()
			
			num_train_predictions = 0
			num_val_predictions = 0
			total_train_loss = 0
			total_train_acc = 0
			total_val_loss = 0
			total_val_acc = 0

			print("Train steps...")
			for current_batch_num, train_batch in enumerate(loaded_training_data):
				num_train_predictions += 1
				optimizer.zero_grad()

				labels, input_sequences = train_batch

				input_sequences = input_sequences.permute(1, 0, 2)
				predictions = transformer(input_sequences)

				loss = criterion(input=predictions, target=labels)
				total_train_loss += loss.item()
				acc = classification_accuracy(input=predictions, target=labels)
				total_train_acc += acc

				loss.backward()
				torch.nn.utils.clip_grad_norm_(transformer.parameters(), 0.7)
				optimizer.step()
				
				print(f"\r{current_batch_num + 1}/{batches_per_epoch_train}, loss: {total_train_loss / (current_batch_num + 1):.05f}, acc: {total_train_acc / (current_batch_num + 1):.05f}", end="")
			print()

			transformer.eval()
			print("Validation steps...")
			with torch.no_grad():
				for current_batch_num, val_batch in enumerate(loaded_validation_data):
					num_val_predictions += 1

					labels, input_sequences = val_batch

					input_sequences = input_sequences.permute(1, 0, 2)
					val_predictions = transformer(input_sequences)
					val_loss = criterion(input=val_predictions, target=labels)
					total_val_loss += val_loss.item()
					acc = classification_accuracy(input=val_predictions, target=labels)
					total_val_acc += acc

					print(f"\r{current_batch_num + 1}/{batches_per_epoch_val}, loss: {total_val_loss / (current_batch_num + 1):.05f}, acc: {total_val_acc / (current_batch_num + 1):.05f}", end="")
			print()

			if best_val_acc < total_val_acc / num_val_predictions:
				best_val_acc = total_val_acc / num_val_predictions
				path_to_best_model_weights = os.path.join(model_save_path, f"model_{current_epoch + 1}.pt")
				torch.save(transformer.state_dict(), path_to_best_model_weights)
				
			print(f"Epoch: {current_epoch:03d} | Train loss: {total_train_loss / num_train_predictions:.05f} | Train acc: {total_train_acc / num_train_predictions:.05f}"
			f" | Val loss: {total_val_loss / num_val_predictions:.05f} | Val acc: {total_val_acc / num_val_predictions:.05f}")


		evaluation_result_text = ""
		
		print("Running evaluation on test set")
		print("Testing individual windows")
		print(f"Using {path_to_best_model_weights} weights")
		evaluation_result_text += f"Evaluated model: {path_to_best_model_weights} (saved as model_for_eval.pt)\n" 

		transformer.load_state_dict(torch.load(path_to_best_model_weights))
		torch.save(transformer.state_dict(), os.path.join(model_save_path, "model_for_eval.pt"))
		transformer.to(options.device)
		transformer.eval()
		
		predicted_classes = []
		gt_classes = []
		with torch.no_grad():
			for current_batch_num, test_batch in enumerate(loaded_test_data):
				print(f"\r{current_batch_num + 1}/{batches_per_epoch_test}", end="")
				labels, input_sequences = test_batch
				input_sequences = input_sequences.permute(1, 0, 2)
				val_predictions = transformer(input_sequences).argmax(-1)
				for pred, reference in zip(val_predictions, labels):
					gt_classes.append(index_to_coin(reference.to("cpu").numpy()[()]))
					predicted_classes.append(index_to_coin(pred.to("cpu").numpy()[()]))
		print()

		overall_accuracy = (np.asarray(gt_classes) == np.asarray(predicted_classes)).sum() / len(gt_classes)
		evaluation_result_text += f"window based accuracy: {overall_accuracy*100:.4f}%\n"
		cm_save_path = os.path.join(model_save_path, "confusion_matrix_window_based")
		cm = confusion_matrix(y_true=gt_classes, y_pred=predicted_classes, labels=[1, 2, 5, 20, 50, 100, 200], normalize="true") 
		cm *= 100
		cm_display = ConfusionMatrixDisplay(cm, display_labels=[1, 2, 5, 20, 50, 100, 200])
		cm_display.plot(cmap=plt.cm.Blues, values_format=".3g")
		plt.title(f"Confusion matrix (Accuracy: {overall_accuracy*100:.2f}%)")
		plt.savefig(cm_save_path + ".pdf")
		plt.savefig(cm_save_path + ".png")
		plt.clf()
		
		print("Testing on samples with majority vote")
		predicted_classes = []
		gt_classes = []
		hop_length = options.hop_length if options.hop_length else options.window_size // 2
		with torch.no_grad():
			num_samples = test_data.reference_length()
			for i in range(num_samples):
				print(f"\r{i + 1}/{num_samples}", end="")
				gt_class, normalized_timeseries = test_data.sample_of_reference(i)
				
				votes = []
				max_length = normalized_timeseries.shape[0]
				if options.sample_individual_timesteps:
					hop_length = options.hop_length if options.hop_length else options.window_size // 2
					num_windows = (max_length - options.window_size) // hop_length
					i = 0
					while i < num_windows:
						input_batch = []
						for _ in range(min(options.batch_size, num_windows)):
							if i <= normalized_timeseries.shape[0] - options.window_size:
								window = normalized_timeseries[i:i + options.window_size]
								i += hop_length
								input_batch.append(window)
						input_batch = torch.stack(input_batch).permute(1, 0, 2)
						predictions = transformer(input_batch).argmax(-1)
						for pred in predictions:
							votes.append(pred.to("cpu").numpy()[()])
				else:
					hop_length = options.hop_length if options.hop_length else options.num_input_features

					normalized_timeseries = normalized_timeseries.squeeze(-1)
					all_embedings = []
					for i in range(0, max_length - options.num_input_features, hop_length):
						all_embedings.append(normalized_timeseries[i:i + options.num_input_features])
					if max_length > 2 * options.num_input_features and (max_length - options.num_input_features) % hop_length != 0:
						all_embedings.append(normalized_timeseries[-options.num_input_features:])

					total_windows = len(all_embedings) // options.window_size
					for i in range(0, total_windows, options.window_size):
						window_to_predict = torch.stack(all_embedings[i:i + options.window_size]).unsqueeze(0).permute(1, 0, 2)
						prediction = transformer(window_to_predict).argmax(-1)[0]
						votes.append(prediction.to("cpu").numpy()[()])
					if len(all_embedings) > options.window_size and len(all_embedings) % options.window_size != 0:
						window_to_predict = torch.stack(all_embedings[-options.window_size:]).unsqueeze(0).permute(1, 0, 2)
						prediction = transformer(window_to_predict).argmax(-1)[0]
						votes.append(prediction.to("cpu").numpy()[()])

				if len(votes) > 0:
					c = Counter(votes)
					pred_class, _ = c.most_common()[0]
					predicted_classes.append(index_to_coin(pred_class))
					gt_classes.append(index_to_coin(gt_class.to("cpu").numpy()[()]))
		print()

		overall_accuracy = (np.asarray(gt_classes) == np.asarray(predicted_classes)).sum() / len(gt_classes)
		evaluation_result_text += f"sample based accuracy: {overall_accuracy*100:.4f}%\n"
		cm = confusion_matrix(y_true=gt_classes, y_pred=predicted_classes, labels=[1, 2, 5, 20, 50, 100, 200], normalize="true") 
		cm *= 100
		cm_save_path = os.path.join(model_save_path, "confusion_matrix_sample_based")
		cm_display = ConfusionMatrixDisplay(cm, display_labels=[1, 2, 5, 20, 50, 100, 200])
		cm_display.plot(cmap=plt.cm.Blues, values_format=".3g")
		plt.title(f"Confusion matrix (Accuracy: {overall_accuracy*100:.2f}%)")
		plt.savefig(cm_save_path + ".pdf")
		plt.savefig(cm_save_path + ".png")
		plt.clf()

		with open(os.path.join(model_save_path, "test_results.txt"), "w") as txt_out:
			txt_out.write(evaluation_result_text)

	else:
		criterion = MSELoss()

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
	parser = argparse.ArgumentParser(description='Transformer on crossbar')

	add_options_to_parser(parser)
	
	args = parser.parse_args()
	main(args)