from options import Options
from encoding import PositionalEncoding

import torch
from torch.nn import Module, TransformerEncoder, TransformerEncoderLayer, Linear


class TransformerModel(torch.nn.Module):
	def __init__(self, options):
		super(TransformerModel, self).__init__()
		
		assert(type(options) == Options)
		self.__options = options

		self.pos_encoder = PositionalEncoding(self.__options.num_input_features)

		self.encoder_layer = TransformerEncoderLayer(d_model=self.__options.num_input_features,
													 nhead=self.__options.encoder_number_of_heads,
													 dim_feedforward=self.__options.encoder_feedforward_dimension,
													 dropout=self.__options.encoder_dropout,
													 activation=self.__options.encoder_activation)

		self.encoder = TransformerEncoder(encoder_layer=self.encoder_layer, 
										  num_layers=self.__options.num_encoder_layers,
										  norm=self.__options.norm)
		self.src_mask = None

		self.decoder = Linear(in_features=options.num_input_features, out_features=1)

		self.__init_decoder_weights()

	def __init_decoder_weights(self):
		init_range = self.__options.weight_intialization_range

		self.decoder.bias.data.zero_()
		self.decoder.weight.data.uniform_(-init_range, init_range)

	def forward(self, x):
		if self.src_mask is None or self.src_mask.size(0) != len(x):
			self.src_mask = self.__generate_square_subsequent_mask(len(x)).to(x.device)

		x = self.pos_encoder(x)
		output = self.encoder(x, self.src_mask)
		output = self.decoder(output)
		return output

	def __generate_square_subsequent_mask(self, size):
		mask = (torch.triu(torch.ones(size, size)) == 1).transpose(0, 1)
		return mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))