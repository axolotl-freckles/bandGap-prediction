from typing import Tuple
from typing import List

import torch
import torch.nn as nn
import numpy    as np

import torch.utils.data as th_data

from tqdm import tqdm

class ModelTrainer():
	def __init__(self,
		name      :str,
		model     :nn.Module,
		device,
		dataloader:th_data.DataLoader,
		loss_func,
		optimizer,
		save_checkpoint:bool=True,
		out_dir        :str =None
	):
		self.name          = name
		self.model         = model
		self.chekpoint_fnm = f"{out_dir if out_dir is not None else '.'}/{name}_ckp.pth"
		self.saved_mdl_fnm = f"{out_dir if out_dir is not None else '.'}/{name}.pth"
		self.dataloader    = dataloader
		self.loss_func     = loss_func
		self.optimizer     = optimizer
		self.save_ckpt     = save_checkpoint
		self.device        = device

	def train(self,
		n_epochs       :int,
		load_checkpoint:bool  = False,
		val_X_Y        :Tuple[torch.Tensor, torch.Tensor] = None
	) -> Tuple[np.ndarray, List[int]] | Tuple[np.ndarray, np.ndarray, List[int]]:
		checkpoint = {
			'epoch'               : 0,
			'model_state_dict'    : self.model.state_dict(),
			'optimizer_state_dict': self.optimizer.state_dict(),
			'loss'                : float('inf')
		}
		if load_checkpoint:
			print('Loading previous checkpoint...', end=' ')
			checkpoint = torch.load(self.chekpoint_fnm)
			self.model    .load_state_dict(checkpoint['model_state_dict'])
			self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
			print(f"Done!, resuming from epoch[{checkpoint['epoch']}]")

		curr_epoch = checkpoint['epoch']
		best_loss  = checkpoint['loss']
		# curr_loss  = checkpoint['loss']

		trn_loss_hist = np.zeros((n_epochs))
		val_loss_hist = np.zeros((n_epochs))
		chckpt_hist   = list()

		val_X, val_Y = None, None
		if val_X_Y is not None:
			val_X, val_Y = val_X_Y

		for epoch in range(n_epochs):
			for i, (X, Y) in tqdm(
				enumerate(self.dataloader),
				desc=f'Epoch [{curr_epoch+epoch+1}/{curr_epoch+n_epochs}] | Starting Loss ({curr_loss:.3e}): '
			):
				self.model.to(self.device)
				self.model.train()
				X = X.to(self.device)
				Y = Y.to(self.device)

				out = self.model(X)
				loss = self.loss_func(out, Y)

				self.optimizer.zero_grad()
				loss.backward()
				self.optimizer.step()

				curr_loss = loss.item()
				trn_loss_hist[epoch] = curr_loss

				if val_X_Y is not None:
					self.model.eval()
					X = val_X.to(self.device)
					Y = val_Y.to(self.device)
					out = self.model(X)
					val_loss_hist[epoch] = self.loss_func(out, Y).item()

					if val_loss_hist[epoch] < best_loss:
						checkpoint['epoch'] = curr_epoch + epoch + 1
						checkpoint['loss']  = val_loss_hist[epoch]
						checkpoint['model_state_dict']     = self.model.state_dict()
						checkpoint['optimizer_state_dict'] = self.optimizer.state_dict()
						best_loss = val_loss_hist[epoch]
						chckpt_hist.append(epoch)
				else:
					if trn_loss_hist[epoch] < best_loss:
						checkpoint['epoch'] = curr_epoch + epoch + 1
						checkpoint['loss']  = trn_loss_hist[epoch]
						checkpoint['model_state_dict']     = self.model.state_dict()
						checkpoint['optimizer_state_dict'] = self.optimizer.state_dict()
						best_loss = trn_loss_hist[epoch]
						chckpt_hist.append(epoch)
				if self.save_ckpt:
					torch.save(checkpoint, self.chekpoint_fnm)
				continue
			continue
		print(f'Final train loss: {curr_loss:.3e}')

		self.model.load_state_dict(checkpoint['model_state_dict'])
		torch.save(self.model, self.saved_mdl_fnm)
		if val_X_Y is None:
			return trn_loss_hist, chckpt_hist
		return trn_loss_hist, val_loss_hist, chckpt_hist
