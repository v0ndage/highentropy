#!/usr/bin/env python3
import os
import sys
import time
import argparse
import numpy as np
import torch
import pytorch_lightning as pl

from programs import (
	db2im, create_database, wait4db,
	build_datamodule, myModel, myTrainer,
	compare, ratioplot, deltaplot, histoplot, barplot, trainplot
)

# Now works on distributed (multi-node) systems.
# num_workers may need optimization
# hyperparameters do indeed need optimization

def main():

	name, path = sys.argv[1], sys.argv[2]

	# Initialize
	images = db2im(str(path))
	if len(images) == 0:
		print('Empty database')
		sys.exit(1)
	
	directory = os.path.join('./NNs', name)
	os.makedirs(directory, exist_ok=True)

	epochs = 100
	batch_size = 32
	cut_off = 5.2

	parameters = {
		'learning_rate': 1.0e-4,
		'cut_off': cut_off,
		'features': 50,
		'interactions': 6,
		'gaussians': 20,
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
		}
	}

	# DATA CURATION
	if int(os.environ.get("LOCAL_RANK", 0)) == 0:
		create_database(directory, images)
	if torch.distributed.is_initialized():
		torch.distributed.barrier()
	db_file = os.path.join(directory, 'new_dataset.db')

	wait4db(db_file, stability_time=10, check_interval=1, timeout=300)

	rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
	print(f"[Rank {rank}] Database file found: {db_file}")

	# TRAINING
	dm = build_datamodule(directory, images, batch_size, cut_off, prepare=False)
	task = myModel(parameters, scheduler)
	trainer = myTrainer(directory, epochs)

	t1 = time.time()
	trainer.fit(task, datamodule=dm)
	t2 = time.time()

	print('Training complete --', round((t2-t1)/60, 2), 'min')

	# PLOTTING
	limit = 10
	results = compare(directory, images, limit)
	MAEs = [
		results[0][4], results[0][5], 
		results[1][4], results[1][5], 
		results[2][4], results[2][5]
	]
	testE = [results[2][0], results[2][1]]
	testF = [results[2][2], results[2][3]]

	print("MAEs:", MAEs)
	np.save(os.path.join(directory, 'maes.npy'), MAEs)

	title = 'test'
	ratioplot(name, 'energy', testE[0], testE[1], title, save=True)
	ratioplot(name, 'force', testF[0], testF[1], title, save=True)
	deltaplot(name, 'energy', testE[0], testE[1], title, save=True)
	histoplot(name, 'energy', testE[0], testE[1], title, save=True)
	histoplot(name, 'force', testF[0], testF[1], title, save=True)
	barplot(name, MAEs, save=True)
	trainplot(name, save=True)

if __name__ == "__main__": main()

###------------------------------------------

###/END
