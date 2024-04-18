import glob

import numpy
import torch

def get_element(line:str): # -> tuple[str, tuple[float, float float]]:
	data = line.split()

	element = data[0]
	coor    = [float(f) for f in data[1:]]
	return element, coor

def space_layout(
	coor_file   ,
	element_dict:dict,
	n_bins      :int,
	range_      :float,
	offset      :float      = 0,
	out         :torch.tensor = None,
	return_n_mol:bool       = False
) -> torch.tensor:
	if out is None:
		out = torch.zeros((n_bins, n_bins, n_bins))

	n_mol:int = 0
	line :str = coor_file.readline()

	n_mol = int(line)
	for i in range(n_mol):
		line = coor_file.readline()
		while len(line.split()) <= 0:
			line = coor_file.readline()
		element, coors = get_element(line)

		element = element_dict[element]
		coors   = (numpy.array(coors)-offset) * n_bins / range_
		coors   = numpy.floor(coors).astype(int)

		out[coors[0], coors[1], coors[2]] = element
	if return_n_mol:
		return out, n_mol
	return out

class MolDataLoader():
	def __init__(
		self,
		file_pattern :str,
		element_dict :dict,
		batch_size   :int   = None,
		shuffle      :bool  = False,
		offset       :float = 0.0,
		_range       :float = 0.0,
		n_bins       :int   = 30,
		normalize_val:float = 1.0,
		device       :any   = None
	) -> None:
		self.dir       = file_pattern
		self.f_names   = numpy.array(glob.glob(self.dir))
		if shuffle:
			numpy.random.shuffle(self.f_names)
		self.len       = len(self.f_names)
		self.b_size    = batch_size if batch_size else self.len

		self.begin  = None
		self.end    = None
		self.buffer = None

		self.el_dict= element_dict
		self.offset = offset
		self.range  = _range
		self.n_bins = n_bins
		self.n_val  = normalize_val

		self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
		pass

	def __getitem__(self, idx) -> torch.tensor:
		batch_idx = idx // self.b_size

		if self.buffer is None:
			self.begin  = batch_idx  * self.b_size
			self.end    = self.begin + self.b_size
			self.buffer = torch.zeros(
				(self.b_size, self.n_bins, self.n_bins, self.n_bins)
			).to(self.device)

			for i in range(self.b_size):
				with open(self.f_names[(self.begin+i)%self.len]) as coor_file:
					space_layout(
						coor_file   = coor_file,
						element_dict= self.el_dict,
						n_bins      = self.n_bins,
						range_      = self.range,
						offset      = self.offset,
						out         = self.buffer[i]
					)
			self.buffer /= self.n_val
		if idx < self.begin or idx >= self.end:
			self.begin  = batch_idx  * self.b_size
			self.end    = self.begin + self.b_size
			self.buffer.zero_()
			for i in range(self.b_size):
				with open(self.f_names[(self.begin+i)%self.len]) as coor_file:
					space_layout(
						coor_file   = coor_file,
						element_dict= self.el_dict,
						n_bins      = self.n_bins,
						range_      = self.range,
						offset      = self.offset,
						out         = self.buffer[i]
					)
			self.buffer /= self.n_val
		return self.buffer[idx % self.b_size]
	
	def __iter__(self):
		for i in range(self.len):
			yield self.__getitem__(i)

class BandGapDataLoader():
	def __init__(self, file_pattern:str, device:any=None) -> None:
		self.dir     = file_pattern
		self.f_names = numpy.array(glob.glob(file_pattern))
		self.len     = len(self.f_names)

		self.device  = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

		self.buffer  = torch.zeros((self.len)).to(self.device)

		for i, f_name in enumerate(self.f_names):
			with open(f_name, 'r') as tar_file:
				self.buffer[i] = float(tar_file.readline().split()[0])
		pass

	def __getitem__(self, idx) -> float:
		return self.buffer[idx]

	def __iter__(self):
		for target in self.buffer:
			yield target
