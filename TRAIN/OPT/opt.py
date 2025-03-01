#!/usr/bin/env python3

# meant for optimizing parameters on smaller datasets (single node)

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

batch_size = 64
epochs = 100

def train(gauss, feat, inter, trial):

	parameters = {
		'learning_rate': 1.0e-4,
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
					'patience': 15, 
					'threshold': 1e-2, 
					'threshold_mode': 'rel', 
					'cooldown': 15, 
					'min_lr': 0, 
					'verbose': True,
				}
	}
	
	dm = myModule(directory, images, batch_size)
	task = myModel(parameters, scheduler)
	trainer = myTrainer(directory, epochs, trial)

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

	if trial.number < 5:
		grid_points = [
			(20, 200, 4),
			(30, 300, 6),
			(40, 400, 8),
			(20, 400, 8),
			(40, 200, 4)
		]
		idx = trial.number % len(grid_points)
		gauss, feat, inter = grid_points[idx]
	else:
		gauss = trial.suggest_int('gauss', 10, 50, step=10) 
		feat = trial.suggest_int('feat', 100, 500, step=100)
		inter = trial.suggest_int('inter', 2, 10, step=2)
	
	print(f'Trial {trial.number}: gauss={gauss}, feat={feat}, inter={inter}')
	validation_loss = train(gauss, feat, inter, trial.number)
	return validation_loss

pruner = optuna.pruners.MedianPruner(
	n_startup_trials=5,
	n_warmup_steps=30,
	interval_steps=10
)

study = optuna.create_study(
	direction="minimize",
	pruner=pruner,
	sampler=optuna.samplers.TPESampler(seed=42))

study.optimize(objective, n_trials=100)
print('best params:', study.best_params)

# Train final model with best parameters
#best_params = study.best_params
#print(f"Training final model with: gauss={best_params['gauss']}, feat={best_params['feat']}, inter={best_params['inter']}")
#final_loss = train(best_params['gauss'], best_params['feat'], best_params['inter'])
#print(f"Final validation loss: {final_loss}")

###/END
