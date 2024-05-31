from typing import Tuple
from typing import List

import torch
import torch.nn as nn

class BG_LSTM(nn.Module):
	def __init__(self,
		h_size:int,
		fc_layers_config:List[Tuple[any, int]]=None
	):
		super(BG_LSTM, self).__init__()
		self.h_size       = h_size
		self.lstm_cell    = nn.LSTMCell(1, h_size)

		if fc_layers_config is None:
			self.fc = nn.Sequential(nn.ReLU(), nn.Linear(h_size, 1))
		else:
			layers = list()
			last_size = h_size
			n_layers  = len(fc_layers_config)
			for i, layer_config in enumerate(fc_layers_config):
				layer_activation, layer_size = layer_config

				layers.append( layer_activation() )
				layers.append( nn.Linear(last_size, layer_size) )
				last_size = layer_size
			self.fc = nn.Sequential(*layers)

	def forward(self, x:torch.Tensor):
		device   = x.get_device()
		batch_sz = x.size()[0]
		h = torch.zeros(batch_sz, self.h_size, device=device)
		c = torch.zeros(batch_sz, self.h_size, device=device)
		for i in range(x.size()[1]):
			h, c = self.lstm_cell(x[:,i].view(batch_sz, 1), (h, c))
		out = self.fc(h)
		return out

class BG_GRU(nn.Module):
	def __init__(self,
		h_size:int,
		fc_layers_config:List[Tuple[any, int]]=None
	):
		super(BG_GRU, self).__init__()
		self.h_size = h_size
		self.gru_cell = nn.GRUCell(1, h_size)

		if fc_layers_config is None:
			self.fc = nn.Sequential(nn.ReLU(), nn.Linear(h_size, 1))
		else:
			layers = list()
			last_size = h_size
			for layer_config in fc_layers_config:
				layer_activation, layer_size = layer_config

				layers.append( layer_activation() )
				layers.append( nn.Linear(last_size, layer_size) )
				last_size = layer_size
			self.fc = nn.Sequential(*layers)

	def forward(self, x:torch.Tensor):
		device   = x.get_device()
		batch_sz = x.size()[0]
		h = torch.zeros(batch_sz, self.h_size, device=device)
		for i in range(x.size()[1]):
			h = self.gru_cell(x[:,i].view(batch_sz, 1), h)
		out = self.fc(h)
		return out
