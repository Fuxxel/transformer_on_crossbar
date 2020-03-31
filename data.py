from options import Options

import torch
from torch.utils.data import Dataset
import h5py

import random
import numpy as np

class CoinDataSubset(Dataset):
	def __init__(self, options, references, indices):
		super().__init__()

		assert(type(options) == Options)

		self.indices = indices
		self.__options = options

		self.window_size = self.__options.window_size
		self.device = self.__options.device

		self.min = self.__options.scaler_min
		self.max = self.__options.scaler_max

		self.preloaded_data = []
		self.preloaded_references = references

		self.data_file = h5py.File(self.__options.path_to_coins, "r")
		
		self.__expand_indices()

	@staticmethod
	def coin_to_index(coin):
		coins = [1, 2, 5, 20, 50, 100, 200]
		return coins.index(coin)

	@staticmethod
	def index_to_coin(index):
		coins = [1, 2, 5, 20, 50, 100, 200]
		return coins[index]

	def __to_one_hot(self, index, max_index):
		encoding = np.zeros(max_index)
		encoding[index] = 1
		return encoding

	def __normalize_sample(self, sample):
		scale = (self.max - self.min) / (sample.max() - sample.min())
		return scale * sample + self.min - sample.min() * scale

	def __len__(self):
		return len(self.preloaded_data)

	def __getitem__(self, idx):
		return self.preloaded_data[idx]

	def reference_length(self):
		return len(self.preloaded_references)

	def save_used_indices(self, path):
		with open(path, "w") as out:
			for index in self.indices:
				coin, sample_num = self.preloaded_references[index]
				out.write(f"{coin}:{sample_num}\n")

	def sample_of_reference(self, index):
		coin, sample_num = self.preloaded_references[index]
		one_hot_index, normalized_timeseries = self.__load_sample_from_data(coin, sample_num)
		return one_hot_index, normalized_timeseries

	def __load_sample_from_data(self, coin, sample):
		timeseries = self.data_file[coin][sample][()]
		normalized_timeseries = self.__normalize_sample(timeseries)

		one_hot_index = np.asarray(self.coin_to_index(int(coin)))
		one_hot_index = torch.from_numpy(one_hot_index).long().to(self.device)
		normalized_timeseries = torch.from_numpy(normalized_timeseries[..., None]).float().to(self.device)
		return one_hot_index, normalized_timeseries

	def __expand_indices(self):
		for current, index in enumerate(self.indices):
			print(f"\r{current + 1}/{len(self.indices)}", end="")
			coin, sample_num = self.preloaded_references[index]
			one_hot_index, normalized_timeseries = self.__load_sample_from_data(coin, sample_num)

			if self.__options.sample_individual_timesteps:
				hop_length = self.__options.hop_length if self.__options.hop_length else self.window_size // 2
				max_length = normalized_timeseries.shape[0]
				for i in range(0, max_length - self.window_size, hop_length):
					self.preloaded_data.append((one_hot_index, normalized_timeseries[i:i + self.window_size]))
			else:
				hop_length = self.__options.hop_length if self.__options.hop_length else self.__options.num_input_features

				all_embedings = []
				max_length = normalized_timeseries.shape[0]
				normalized_timeseries = normalized_timeseries.squeeze(-1)
				for i in range(0, max_length - self.__options.num_input_features, hop_length):
					all_embedings.append(normalized_timeseries[i:i + self.__options.num_input_features])
				if max_length > 2 * self.__options.num_input_features and (max_length - self.__options.num_input_features) % hop_length != 0:
					all_embedings.append(normalized_timeseries[-self.__options.num_input_features:])
				
				total_windows = len(all_embedings) // self.window_size
				for i in range(0, total_windows, self.window_size):
					self.preloaded_data.append((one_hot_index, torch.stack(all_embedings[i:i + self.window_size])))
				if len(all_embedings) > self.window_size and len(all_embedings) % self.window_size != 0:
					self.preloaded_data.append((one_hot_index, torch.stack(all_embedings[-self.window_size:])))
		print()

class CoinDataSetPreparer():
	def __init__(self, options):
		super().__init__()

		assert(type(options) == Options)

		self.__options = options
	
		self.preloaded_references = []

		self.data_file = h5py.File(self.__options.path_to_coins, "r")
		
		self.__prepare_data()

		self.data_file.close()

	def get_options(self):
		return self.__options

	def __len__(self):
		return len(self.preloaded_references)

	def __prepare_data(self):
		# First determine how many coins to load for each type
		minimum = 100000
		for coin in self.data_file.keys():
			minimum = np.minimum(minimum, len(self.data_file[coin].keys()))

		if np.isinf(minimum):
			raise ValueError(f"ERROR: No data found in {self.__options.path_to_coins}")

		print(f"Loading {minimum} random samples per coin class.")
		
		for coin in self.data_file.keys():
			samples = random.choices(list(self.data_file[coin].keys()), k=minimum)
			for i in samples:
				self.preloaded_references.append((coin, i))

		random.shuffle(self.preloaded_references)

class FakeDataSet(Dataset):
	def __init__(self, options):
		super().__init__()
		
		assert(type(options) == Options)

		self.__options = options
		self.__data_scaler = MinMaxScaler(feature_range=(self.__options.scaler_min, self.__options.scaler_max))

		self.window_size = self.__options.window_size
		self.num_samples = self.__options.num_samples
		self.device = self.__options.device

		self.min = self.__options.scaler_min
		self.max = self.__options.scaler_max

		self.__prepare_data()

	def __normalize_sample(self, sample):
		scale = (self.max - self.min) / (sample.max() - sample.min())
		return scale * sample + self.min - sample.min() * scale

	def __len__(self):
		return self.__options.num_samples

	def __getitem__(self, idx):
		start_index = idx * self.window_size
		return self.__normalize_sample(self.signal[start_index:start_index + self.window_size]).unsqueeze(-1)
		
	def __random_frequency(self):
		freq_range_low = self.__options.random_frequency_range_low
		freq_range_high = self.__options.random_frequency_range_high
		return np.random.uniform(freq_range_low, freq_range_high)

	def __generate_timeseries(self, time, random_frequencies=None):
		if not random_frequencies:
			num_random_frequencies = self.__options.num_random_frequencies
			random_frequencies = [self.__random_frequency() for _ in range(num_random_frequencies)]

		carrier = np.sin(time)
		for random_frequency in random_frequencies:
			carrier += np.sin(time * random_frequency)

		if self.__options.add_noise:
			noise_range = self.__options.noise_range
			carrier += np.random.uniform(-noise_range, noise_range, len(carrier))

		return random_frequencies, carrier

	def __prepare_data(self, random_frequencies=None):
		num_predict_forward_steps = self.__options.num_predict_forward_steps

		num_timesteps = self.num_samples * self.window_size
		multiplier = num_timesteps // 1000
		time = np.linspace(0, 2 * np.pi * multiplier, num_timesteps)

		self.random_frequencies, self.signal = self.__generate_timeseries(time, random_frequencies=random_frequencies)
		# self.signal = np.reshape(self.signal, (-1, 1))
		# self.signal = self.__data_scaler.fit_transform(self.signal)
		self.signal = torch.from_numpy(self.signal).float().to(self.device)

class FakeDataGenerator():

	def __init__(self, options):
		assert(type(options) == Options)

		self.__options = options
		self.__data_scaler = MinMaxScaler(feature_range=(self.__options.scaler_min, self.__options.scaler_max))

	def __random_frequency(self):
		freq_range_low = self.__options.random_frequency_range_low
		freq_range_high = self.__options.random_frequency_range_high
		return np.random.uniform(freq_range_low, freq_range_high)

	def __generate_timeseries(self, time, random_frequencies=None):
		if not random_frequencies:
			num_random_frequencies = self.__options.num_random_frequencies
			random_frequencies = [self.__random_frequency() for _ in range(num_random_frequencies)]

		carrier = np.sin(time)
		for random_frequency in random_frequencies:
			carrier += np.sin(time * random_frequency)

		if self.__options.add_noise:
			noise_range = self.__options.noise_range
			carrier += np.random.uniform(-noise_range, noise_range, len(carrier))

		return random_frequencies, carrier

	def __batch_generator(self, signal, window_size, batch_size):
		device = self.__options.device
		num_predict_forward_steps = self.__options.num_predict_forward_steps

		batch_indices = np.arange(0, len(signal) - window_size - num_predict_forward_steps)
		np.random.shuffle(batch_indices)
		batch_indices = np.reshape(batch_indices, (len(batch_indices) // batch_size, batch_size))

		for batch_index in batch_indices:
			batch_input = np.asarray([signal[i:i + window_size] for i in batch_index])
			batch_target = np.asarray([signal[i + num_predict_forward_steps:i + num_predict_forward_steps + window_size] for i in batch_index])

			# Add extra dimension per signal time step --> (batch_size, window_size, 1)
			# then permute to (window_size, batch_size, 1)
			batch_input = torch.from_numpy(batch_input).to(device)[..., None].permute(1, 0, 2).float()
			batch_target = torch.from_numpy(batch_target).to(device)[..., None].permute(1, 0, 2).float()

			yield (batch_input, batch_target)

		# rand_upper_bound = [len(signal) - window_size - num_predict_forward_steps for _ in range(batch_size)]

		# while True:
		# 	random_start_index = np.random.randint(0, rand_upper_bound)
		# 	batch_input = np.asarray([signal[i:i + window_size] for i in random_start_index])
		# 	batch_target = np.asarray([signal[i + num_predict_forward_steps:i + num_predict_forward_steps + window_size] for i in random_start_index])

		# 	# Add extra dimension per signal time step --> (batch_size, window_size, 1)
		# 	# then permute to (window_size, batch_size, 1)
		# 	batch_input = torch.from_numpy(batch_input).to(device)[..., None].permute(1, 0, 2).float()
		# 	batch_target = torch.from_numpy(batch_target).to(device)[..., None].permute(1, 0, 2).float()

		# 	yield (batch_input, batch_target)

	# returns: random_frequencies, generator(x, y), generator(val_x, val_y)
	#                                                    ^---->> None if validation_split = 0
	def get_data_generators(self, num_samples, random_frequencies=None, validation_split=0.1):
		window_size = self.__options.window_size
		batch_size = self.__options.batch_size
		num_predict_forward_steps = self.__options.num_predict_forward_steps

		num_timesteps = num_samples * window_size
		num_timesteps = ((num_timesteps - window_size - num_predict_forward_steps) // batch_size) * batch_size
		multiplier = num_timesteps // 1000
		time = np.linspace(0, 2 * np.pi * multiplier, num_timesteps)

		random_frequencies, signal = self.__generate_timeseries(time, random_frequencies=random_frequencies)

		split_point = int(len(signal) * (1 - validation_split))
		train_signal = signal[:split_point]
		validation_signal = signal[split_point:]

		return random_frequencies, self.__batch_generator(train_signal, window_size, batch_size), None if validation_split == 0 else self.__batch_generator(validation_signal, window_size, batch_size)

		
