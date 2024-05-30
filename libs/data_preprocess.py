from typing import Tuple
from typing import List

import random
import torch
import numpy as np

from torch.utils.data import Dataset

class MultiLenDataset(Dataset):
	def __init__(self,
		X:List[np.ndarray|torch.Tensor],
		Y:np.ndarray | torch.Tensor,
		shuffle:bool=True
	):
		self.len   = len(X)
		self.max_f = 0
		for x in X:
			if len(x) > self.max_f:
				self.max_f = len(x)
		self.shuffled_idxs = [i for i in range(self.len)]
		if shuffle:
			random.shuffle(self.shuffled_idxs)
		self.X = [X[i] for i in self.shuffled_idxs]
		self.Y = [Y[i] for i in self.shuffled_idxs]

	def __len__(self):
		return self.len
	
	def __getitem__(self, index) -> Tuple[torch.Tensor, torch.Tensor]:
		feature_shape = []
		x = self.X[index]
		if type(x) == np.ndarray or type(x) == torch.Tensor:
			if len(x.shape) > 1:
				feature_shape = x[0].shape
		X_tensor = torch.zeros(self.max_f, *feature_shape)
		for i, feature in enumerate(x):
			X_tensor[i] = feature

		return X_tensor, self.Y[index]