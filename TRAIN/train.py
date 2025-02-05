#!/usr/bin/env python3

# Basic training with SchnetPack
# All work is done in local programs.py file
# Here we simply initialize, and train
# Plotting functions are optional

import sys, time, os
import numpy as np
from ase.io import write
from programs import *
from ase.calculators.emt import EMT
from ase.calculators.singlepoint import SinglePointCalculator

###------------------------------------------
### INITIALIZE
###------------------------------------------

name = sys.argv[1]
path = sys.argv[2]

print(name, path)

images = db2im(path)

# shuffle
index = np.random.permutation(len(images))
images = [images[i] for i in index]

# sort
#index = np.argsort([i.get_total_energy() for i in images])
#images = [images[i] for i in index[::-1]]

# rotate images randomly
#images = [spin(i) for i in images]

# filter images if need be
#images = safe(images)

print(len(images), 'images collected')

directory = 'NNs/'+name+'/'
try: os.mkdir(directory)
except: pass

#Hyperparameterization

batch_size = 32
epochs = 10

parameters = {
	'learning_rate': 1.0e-4,
	'cut_off': 5.2,
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
		'verbose': True,
	}
}

print('batch_size:', batch_size)
print('epochs: ', epochs)
print(parameters)
print('s_args:', scheduler['s_args'])

dm = myModule(directory, images, batch_size)
task = myModel(parameters, scheduler)
trainer = myTrainer(directory, epochs)

###------------------------------------------
### TRAINING
###------------------------------------------

# load from checkpoint if needed
#ckpt = 'NNs/ledge/lightning_logs/version_0/checkpoints/epoch=293-step=198744.ckpt'

t1 = time.time()
#trainer.fit(task, datamodule=dm, ckpt_path=ckpt)
trainer.fit(task, datamodule=dm)
t2 = time.time()

minutes = round((t2-t1)/60,2)
print('Training complete --', minutes, 'min')

validation_loss = trainer.callback_metrics.get('val_loss')
if validation_loss is None:
	raise ValueError('Validation loss not found')

print(validation_loss.item())

###------------------------------------------
### PLOTTING
###------------------------------------------

limit = 10
results = compare(directory, images, limit)
MAEs = [results[0][4], results[0][5], 
		results[1][4], results[1][5], 
		results[2][4], results[2][5]]

testE = [results[2][0], results[2][1]]
testF = [results[2][2], results[2][3]]

print(MAEs)
save = True
np.save(directory+'maes.npy', MAEs)

save = True
title = 'test'

ratioplot(name, 'energy', testE[0], testE[1], title, save)
ratioplot(name, 'force', testF[0], testF[1], title, save)
deltaplot(name, 'energy', testE[0], testE[1], title, save)
histoplot(name, 'energy', testE[0], testE[1], title, save)
histoplot(name, 'force', testF[0], testF[1], title, save)
barplot(name, MAEs, save)

###------------------------------------------
### OUTPUT
###------------------------------------------

"""

T = '1910394746:AAHEppJQDdrdp8-0UYuCQk5DSgHXj1c26SA'
I = '1925962595'
from tqdm.contrib.telegram import tqdm
for i in tqdm(range(0), desc='Training Done', token=T, chat_id=I): continue

"""

###------------------------------------------
### DONE
###------------------------------------------
