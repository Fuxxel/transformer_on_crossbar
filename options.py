
class Options(object):
	def __init__(self):
		# Default Options
		self.__options = {
			# General 
			"seed": None,
			"num_input_features": 250,
			"learning_rate": 0.0005,
			"num_epochs": 20,
			"artifact_dir": "artifacts",

			# Fake Data Generator Options
			"num_predict_forward_steps": 1,
			"num_samples": 100,
			"window_size": 100,
			"batch_size": 10,
			"add_noise": True,
			"noise_range": 0.2,
			"num_random_frequencies": 5,
			"random_frequency_range_low": 0.1,
			"random_frequency_range_high": 1.0,
			"scaler_min": -1.0,
			"scaler_max": 1.0,

			# Encoder Layer Options
			"encoder_number_of_heads": 10,
			"encoder_feedforward_dimension": 2048,
			"encoder_dropout": 0.1,
			"encoder_activation": "relu",

			# Encoder Layer Stacking Options
			"num_encoder_layers": 1,
			"norm": None,

			# Decoder Opttions
			"weight_intialization_range": 0.1
		}

	def get_option_names(self):
		return self.__options.keys()
	
	def __setattr__(self, name, value):
		if name == "_Options__options":
			object.__setattr__(self, name, value)

		option_keys = self.__options.keys()
		if name in option_keys:
			self.__options[name] = value

		object.__setattr__(self, name, value)

	def __getattribute__(self, name):
		if name == "_Options__options":
			return object.__getattribute__(self, name)

		option_keys = self.__options.keys()
		if name in option_keys:
			return self.__options[name]

		return object.__getattribute__(self, name)