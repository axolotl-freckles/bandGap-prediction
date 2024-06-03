import os
import sys
import pickle
import random

import torch
import torch.utils.data as trch_data
import torch.nn         as nn
import numpy            as np
import pandas           as pd
import torch.utils
from   torch.utils.data import DataLoader

import models.configs  as configs
import models.lstm_mol as models
from   libs.data_preprocess import MultiLenDataset
from   libs.train           import ModelTrainer

AVAILABLE_MODELS = {
	'LSTM' : models.BG_LSTM,
	'GRU'  : models.BG_GRU
}

MAX_EPOCHS =  300
BATCH_SIZE = 1024

DATA_CSV       = './datasets/full/omdb_smile_data_set.csv'
MODEL_DIR      = './saved_models'
TRAIN_HIST_DIR = './results/hist'

def main(nargs:int, argv:list) -> int:
	random    .seed       (42)
	torch     .manual_seed(42)
	# torch.cuda.seed_all   (42)
	np.random .seed       (42)
	picklename = f'{MODEL_DIR}/finished.pkl'
	# Set[model_name:str]
	finished_models = set()
	with open(picklename, 'rb') as picklefile:
		finished_models = \
			pickle.load(picklefile)\
			if os.path.isfile(picklename) else set()
	
	device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
	print(f'Training on "{torch.cuda.get_device_name(device)}"\n')

	dataframe = pd.read_csv(DATA_CSV)
	print(dataframe.info(), end='\n\n')

	dataset = MultiLenDataset(
		[
			torch.tensor( [float(ord(char)) for char in smile] ) \
			for smile in dataframe['SMILE']
		],
		torch.Tensor(dataframe['bgs']).unsqueeze(1),
		shuffle=False
	)
	dataset_len = len(dataset)
	train_split = int(dataset_len*0.7)
	val_split   = dataset_len - train_split
	train_set, val_set = trch_data.random_split(dataset, [train_split, val_split])
	print(f'Dataset lenght: {dataset_len} [{train_split}, {val_split}]\n\n')

	train_loader = DataLoader(train_set, BATCH_SIZE, shuffle=True, pin_memory=True)
	val_loader   = DataLoader(  val_set, BATCH_SIZE, shuffle=True, pin_memory=True)

	for name, config in configs.DEF_CONFIGS.items():
		for model_name, model_clss in AVAILABLE_MODELS.items():
			savename  = f'{model_name}_{name}'
			model     = model_clss(config['h_size'], config['l_config'])
			loss_fn   = nn.MSELoss()
			optimizer = torch.optim.SGD(model.parameters(), lr=0.002)
			trainer   = ModelTrainer(
				savename, model, device, train_loader, loss_fn, optimizer,
				save_checkpoint=True, out_dir=MODEL_DIR
			)
			if savename in finished_models:
				print(f'"{savename}" already trained!')
				continue
			print(f'\nTraining: {savename}')
			train_hist, val_hist, _ = trainer.train(MAX_EPOCHS, load_checkpoint=False, val_X_Y=val_loader)
			np.save(f'{TRAIN_HIST_DIR}/{savename}_trHist.npy', train_hist)
			np.save(f'{TRAIN_HIST_DIR}/{savename}_vlHist.npy', val_hist)
			finished_models.add(savename)
			with open(picklename, 'wb') as picklefile:
				pickle.dump(finished_models, picklefile)
			continue
		continue

	return 0

if __name__ == '__main__':
	retval = main(len(sys.argv), sys.argv)
	print(f'\nProcess ended with code \'{retval:04d}\'')
