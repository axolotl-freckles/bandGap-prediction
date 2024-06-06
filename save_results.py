import os
import sys
import pickle
import random

import torch
import torch.nn          as nn
import numpy             as np
import pandas            as pd
import matplotlib.pyplot as plt
import torch.utils
from   torch.utils.data import DataLoader

import models.configs  as configs
import models.lstm_mol as models
from   libs.data_preprocess import MultiLenDataset
from   libs.train           import ModelTrainer
from   models.configs       import DEF_CONFIGS

DATA_CSV       = './datasets/full/omdb_smile_data_set.csv'
MODEL_DIR      = './saved_models'
TRAIN_HIST_DIR = './results/hist'
RESULT_OUT_DIR = './results/results'

AVAILABLE_MODELS = {
	'LSTM',
	'GRU'
}

def main(argn:int, argv:list) -> int:
	picklename = f'{MODEL_DIR}/finished.pkl'
	finished_models = None
	with open(picklename, 'rb') as picklefile:
		finished_models = pickle.load(picklefile)
	assert finished_models != None, "There should be at least one model trained"

	n_configs = len(DEF_CONFIGS)
	n_models  = len(AVAILABLE_MODELS)
	best_MSE  = np.zeros((n_configs, 2, n_models))

	fig, ax = plt.subplots(len(DEF_CONFIGS), len(AVAILABLE_MODELS), figsize=(8.0, 10.0))
	fig.suptitle('MSE by Epoch')
	fig.tight_layout(pad=2.8)
	for i, configname in enumerate(DEF_CONFIGS):
		for j, model_type in enumerate(AVAILABLE_MODELS):
			modelname = f'{model_type}_{configname}'
			print(f'Loading "{modelname}"\'s results...')
			train_hist = np.load(f'{TRAIN_HIST_DIR}/{modelname}_trHist.npy')
			val_hist   = np.load(f'{TRAIN_HIST_DIR}/{modelname}_vlHist.npy')

			train_min = np.min(train_hist)
			val_min   = np.min(val_hist)
			print(f'   Min MSE (t):{train_min:10.3e}')
			print(f'   Min MSE (v):{val_min  :10.3e}')

			best_MSE[i,:,j] = np.array([val_min, train_min])

			ax[i, j].set_title(f'{modelname}')
			ax[i, j].plot(train_hist)
			ax[i, j].plot(val_hist)
			if i==0:
				ax[i, j].set_xlabel('Epoch')
			if j==0:
				ax[i, j].set_ylabel('MSE')
			if i==0 and j==0:
				ax[i, j].legend(['train', 'test'])

			# print(f'Testing "{modelname}"...')
			continue
		continue
	print()
	plot_filename = f'{RESULT_OUT_DIR}/train_hist.png'
	plt.savefig(plot_filename)
	print(f'Training history saved to: {plot_filename}\n')

	table_filename = f'{RESULT_OUT_DIR}/best_MSE.txt'

	max_confgname_len = max([len(name) for name in DEF_CONFIGS])
	max_modelname_len = max([len(name) for name in AVAILABLE_MODELS])
	column_w  = max(max_modelname_len, 10)
	row_name_len = max_confgname_len+len('(_)')
	table_width = max_confgname_len+len('(_)')+column_w*n_models
	table_buff  = ' '*((table_width-len('Best MSE'))//2)+'Best MSE\n'
	h_line      = '-'*row_name_len+('+'+'-'*(column_w))*n_models+'-\n'
	table_buff += h_line
	table_buff += ' '*(max_confgname_len+len('(_)'))
	for modelname in AVAILABLE_MODELS:
		table_buff += '|'+' '*(column_w-len(modelname))+modelname
	table_buff += '\n'

	for i, configname in enumerate(DEF_CONFIGS):
		table_buff += h_line
		table_buff += ' '*(row_name_len-len(configname)-len('(_)'))+f'{configname}(v)'
		for best_mse in best_MSE[i,0]:
			table_buff += '|'+' '*(column_w-10)+f'{best_mse:10.3e}'
		table_buff += '\n'+' '*(row_name_len-len('(_)'))+'(t)'
		for best_mse in best_MSE[i,1]:
			table_buff += '|'+' '*(column_w-10)+f'{best_mse:10.3e}'
		table_buff += '\n'
		continue
	print(table_buff)

	with open(table_filename, 'w+') as table_file:
		table_file.write(table_buff)
	print(f'Result table saved to: {table_filename}')

	return 0

if __name__ == '__main__':
	retval = main(len(sys.argv), sys.argv)
	print(f'\nProcess finished with code {retval:04d}')
