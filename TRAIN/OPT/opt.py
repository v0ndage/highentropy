#!/usr/bin/env python3

import sys, time, os
import optuna
import numpy as np
from ase.io import write
from programs import *

name, path = sys.argv[1], sys.argv[2]

images = db2im(path)

index = np.random.permutation(len(images))
images = [images[i] for i in index]

directory = 'NNs/'+name+'/'
try: os.mkdir(directory)
except: pass

batch_size = 20
epochs = 200

def train(gauss, feat, inter, trial):

	parameters = {
		'learning_rate': 5.0e-4,
		'cut_off': 5.2,
		'features': feat,
		'interactions': inter,
		'gaussians': gauss,
		'energy_weight': 0.15,
		'force_weight': 0.85,
	}

	scheduler = {
		's_class': torch.optim.lr_scheduler.ReduceLROnPlateau,
		's_metric': 'val_loss',
		's_args': {
				'mode': 'min', 
				'factor': 0.75, 
				'patience': 10, 
				'threshold': 1e-4, 
				'threshold_mode': 'rel', 
				'cooldown': 15, 
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

	gauss = trial.suggest_int('gauss', 20, 25, step=5)
	feat = trial.suggest_int('feat', 100, 500, step=50)
	inter = trial.suggest_int('inter', 5, 15, step=1)
	
	print(f'Trial {trial.number}: gauss={gauss}, feat={feat}, inter={inter}')
	validation_loss = train(gauss, feat, inter, trial.number)
	return validation_loss

study = optuna.create_study(
	direction="minimize",
	sampler=optuna.samplers.TPESampler(seed=137))

study.optimize(objective, n_trials=20)
print('best params:', study.best_params)

###/END
