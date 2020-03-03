from options import Options

import torch
from torch.utils.data import Dataset

import numpy as np
from sklearn.preprocessing import MinMaxScaler

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

		
