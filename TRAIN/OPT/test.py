#!/usr/bin/env python3

import sys, time, os
import optuna
import numpy as np
from ase.io import write
from programs import *

def mytrain(gauss, feat, inter, step):

	print('step =', step)

	directory = 'NNs/'+name+'-'+str(step)+'/'
	try: os.mkdir(directory)
	except: pass

	parameters = {
		'learning_rate': 1.0e-3,
		'cut_off': 5.2,
		'features': feat,
		'interactions': inter,
		'gaussians': gauss,
		'energy_weight': 0.1,
		'force_weight': 0.9,
	}

	scheduler = {
		's_class': torch.optim.lr_scheduler.ReduceLROnPlateau,
		's_metric': 'val_loss',
		's_args': {
					'mode': 'min', 
					'factor': 0.3, 
					'patience': 20, 
					'threshold': 1e-2, 
					'threshold_mode': 'rel', 
					'cooldown': 20, 
					'min_lr': 0, 
					'verbose': True,
				}
	}

	dm = myModule(directory, images, batch_size)
	task = myModel(parameters, scheduler)
	trainer = myTrainer(directory, epochs)

	t1 = time.time()
	trainer.fit(task, datamodule=dm)
	t2 = time.time()

	minutes = round((t2-t1)/60,2)
	print('Training complete --', minutes, 'min')

	validation_loss = trainer.callback_metrics.get('val_loss')
	if validation_loss is None:
		raise ValueError('Validation loss not found')
	
	return validation_loss.item()

def objective(trial):

	try:
		step = trial.number
		gauss = trial.suggest_int('gauss', 30, 80)
		feat = trial.suggest_int('feat', 100, 900)
		inter = trial.suggest_int('inter', 3, 9)

		print(gauss, feat, inter)

		val_loss = mytrain(gauss, feat, inter, step)
		return val_loss

	except Exception as e:
		# Log the error and return a large loss value
		print(f"Trial {trial.number} failed with error: {e}")
		return float("inf") 

if __name__ == "__main__":


	name, path = sys.argv[1], sys.argv[2]

	images = db2im(path)

	index = np.random.permutation(len(images))
	images = [images[i] for i in index]

	batch_size = 32
	epochs = 100

	study = optuna.create_study(direction='minimize')
	study.optimize(objective, n_trials=50,  n_jobs=1)
	print(study.best_params, '\n')


###
