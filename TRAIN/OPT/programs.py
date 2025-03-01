#!/usr/bin/env python3

# Meant for optuna training on single node

import os, time
import torch, torchmetrics
import schnetpack as spk
import pytorch_lightning as pl
from schnetpack.data import ASEAtomsData, AtomsDataModule

import numpy as np
from ase.db import connect

from tqdm import tqdm

###------------------------------------------
### UTILITY FUNCTIONS
###------------------------------------------

#trys to convert db to images from path
def db2im(p):
	assert os.path.isfile(p), 'no db at path='+str(p)
	db = connect(p)
	im = [db.get_atoms(id=i+1) for i in range(len(db))]
	print(len(im), 'images read')
	return im

###------------------------------------------
### SCHNET FUNCTIONS
###------------------------------------------

def LoadSchNetCalc(directory):

	calculator = spk.interfaces.SpkCalculator(
		model_file=os.path.join(directory, 'best_inference_model'),
		neighbor_list=spk.transform.ASENeighborList(cutoff=5.2),
		energy_key='energy',
		force_key='forces',
		energy_unit='eV',
		force_units='eV/Angstrom',
		position_unit='Angstrom',
	)
	return calculator

def compare(directory, images):

	#Link model to Calc
	calculator = LoadSchNetCalc(directory)

	#Get split indicies [train, val, test]
	splitfile = np.load(os.path.join(directory, 'split.npz'))
	splits = [splitfile[i] for i in splitfile]

	measures = []

	for s in splits:

		trueE = []; predE = []
		trueF = []; predF = []

		for i in tqdm(range(len(s)), desc='comparing values'):
			image = images[s[i]]
			trueE.append(image.get_potential_energy())
			trueF.append(image.get_forces())
			copy = image.copy()
			copy.set_calculator(calculator)
			predE.append(copy.get_potential_energy())
			predF.append(copy.get_forces())

		trueE = np.array(list(trueFlat(trueE)))
		predE = np.array(list(trueFlat(predE)))
		trueF = np.array(list(trueFlat(trueF)))
		predF = np.array(list(trueFlat(predF)))

		MAEe = np.mean(np.abs(trueE-predE))
		MAEf = np.mean(np.abs(trueF-predF))

		measures.append([trueE, predE, trueF, predF, MAEe, MAEf])
		#measures.append([trueE, predE, MAEe])

	return measures

def myModule(directory, images, bs):

	#First convert images to ASEAtomsData
	path = os.path.join(directory, 'new_dataset.db')
	if not os.path.exists(path):

		property_list = []
		for i in images:
			energy = np.array([i.get_potential_energy()])
			forces = np.array(i.get_forces())
			property_list.append({'energy': energy, 'forces': forces})
			
		new_dataset = ASEAtomsData.create(
			os.path.join(directory, 'new_dataset.db'), 
			distance_unit='Ang',
			property_unit_dict={'energy':'eV', 'forces':'eV/Ang'},
		)

		new_dataset.add_systems(property_list, images)

	#Now wrap AtomsData in DataModule class
	DM = AtomsDataModule(
		path,
		split_file = os.path.join(directory, 'split.npz'),
		batch_size=bs, val_batch_size=bs, test_batch_size=bs,
		num_train=int(0.7*len(images)), 
		num_val=int(0.15*len(images)), 
		num_test=int(0.15*len(images)),
		load_properties = ['energy', 'forces'],
		transforms=[
			spk.transform.ASENeighborList(cutoff=5.2),
			spk.transform.RemoveOffsets('energy', remove_mean=True, remove_atomrefs=False),
			spk.transform.CastTo32()
		],
		num_workers=4, #cores
		pin_memory=True, # set to false, when not using a GPU
	)

	DM.prepare_data()
	DM.setup()

	return DM

def myModel(parameters, scheduler):
	
	#Parameterize
	lr = parameters['learning_rate']
	cutoff = parameters['cut_off']
	features = parameters['features']
	interactions = parameters['interactions']
	gaussians = parameters['gaussians']
	e_weight = parameters['energy_weight']
	f_weight = parameters['force_weight']
	
	#Build model
	radial_basis = spk.nn.GaussianRBF(n_rbf=gaussians, cutoff=cutoff)
	schnet = spk.representation.SchNet(
		n_atom_basis=features, 
		n_interactions=interactions,
		radial_basis=radial_basis,
		cutoff_fn=spk.nn.CosineCutoff(cutoff)
	)
	
	#Assign training features
	pred_energy = spk.atomistic.Atomwise(n_in=features, output_key='energy')
	pred_forces = spk.atomistic.Forces(energy_key='energy', force_key='forces')
	
	#Build potential
	nnpot = spk.model.NeuralNetworkPotential(
		representation=schnet,
		input_modules=[spk.atomistic.PairwiseDistances()],
		output_modules=[pred_energy, pred_forces],
		postprocessors=[
			spk.transform.CastTo64(),
			spk.transform.AddOffsets('energy', add_mean=True, add_atomrefs=False)
		]
	)
	
	#Define loss weights for features
	output_energy = spk.task.ModelOutput(
		name='energy',
		loss_fn=torch.nn.HuberLoss(delta=0.1),
		loss_weight=e_weight,
		metrics={"MAE": torchmetrics.MeanAbsoluteError()}
	)
	
	output_forces = spk.task.ModelOutput(
		name='forces',
		loss_fn=torch.nn.MSELoss(),
		loss_weight=f_weight,
		metrics={"MAE": torchmetrics.MeanAbsoluteError()}
	)

	#Define task
	task = spk.task.AtomisticTask(
		model=nnpot,
		outputs=[output_energy, output_forces],
		optimizer_cls=torch.optim.AdamW,
		optimizer_args={'lr': lr},
		scheduler_cls=scheduler['s_class'],
		scheduler_monitor=scheduler['s_metric'],
		scheduler_args=scheduler['s_args'],
	)

	return task

def myTrainer(directory, epochs, trial):
	
	#Assign logs
	logger = pl.loggers.TensorBoardLogger(save_dir=directory)
	callbacks = [
		spk.train.ModelCheckpoint(
			model_path=os.path.join(directory, 'net-'+str(trial)),
			save_top_k=1,
			monitor='val_loss'
		),
		pl.callbacks.EarlyStopping(monitor='val_loss', patience=30, verbose=True),
	]

	#Define trainer
	trainer = pl.Trainer(
		devices = 1,
		accelerator = 'gpu',
		strategy = 'ddp',
		precision = '16-mixed',
		callbacks=callbacks,
		logger=logger,
		log_every_n_steps=10,
		default_root_dir=directory,
		max_epochs=epochs,
	)
	
	return trainer

####/END
