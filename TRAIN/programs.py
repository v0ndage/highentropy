#!/usr/bin/env python3

import os, time
import torch, torchmetrics
import torch.distributed as dist

import pytorch_lightning as pl
from pytorch_lightning.utilities.rank_zero import rank_zero_only

import schnetpack as spk
from schnetpack.data import ASEAtomsData, AtomsDataModule

import numpy as np
from pandas import read_csv

import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
import matplotlib.ticker as ticker
from scipy.stats import gaussian_kde, lognorm
from scipy.spatial.transform import Rotation
from collections.abc import Iterable

from ase.db import connect
from ase.io import read, write
from ase.io.vasp import read_vasp_out
from ase.calculators.singlepoint import SinglePointCalculator
from ase.data import chemical_symbols

from tqdm import tqdm

###------------------------------------------
### UTILITY FUNCTIONS
###------------------------------------------

# wrap output as string to flatten nested ragged arrays
def trueFlat(A):
	for a in A:
		if isinstance(a,Iterable) and not isinstance(a,(str,bytes)): 
			yield from trueFlat(a)
		else: yield a

# random n ints that sum to k
def randSum(n, k):
	if n==1: return [k]
	num = np.random.randint(1, k)
	return [num] + randSum(n-1, k-num)

# for stoiplot
def CoM(inner_list):
		positions = range(1, len(inner_list)+1)
		return sum(p*w for p, w in zip(positions, inner_list)) / sum(inner_list)

# rotates atoms object as well as forces
def spin(atoms):
	e = atoms.get_total_energy()
	f = atoms.get_forces()
	p = atoms.get_positions()
	test = atoms.copy()
	rot = Rotation.random().as_matrix()
	p = np.dot(p, rot.T)
	f = np.dot(f, rot.T)
	test.set_positions(p)
	test.center()
	test.calc = SinglePointCalculator(test)
	test.calc.results['energy'] = np.float64(e)
	test.calc.results['forces'] = np.float64(f)
	return test

# lognorm sample of size s from y, scaled to y
def normlog(y, s):
	rng = np.random.default_rng()
	assert len(y)>s, 'sample >= space'
	yabs = np.abs(y)
	x = np.arange(len(y))
	p = lognorm(yabs).pdf(x)
	p = p/np.sum(p)
	
	idx = np.sort(rng.choice(x, size=s, replace=False, p=p))
	sample = [y[i] for i in idx]
	return idx, sample

# trys to convert db to images from path
def db2im(p):
	assert os.path.isfile(p), 'no db at path='+str(p)
	db = connect(p)
	im = [db.get_atoms(id=i+1) for i in range(len(db))]
	print(len(im), 'images read')
	return im

# checks for bounds, proximity and wild energies
def safe(images):
	emax = 0.0
	dmin = 1.0
	valid = []
	for i in images:
		scaled = i.get_scaled_positions()
		if (scaled >= 0).all() and (scaled < 1).all():
			dist = i.get_all_distances()
			if all([d<1e-9 or d>dmin for d in dist.flatten()]):
				if np.abs(i.get_total_energy()) < emax:
					valid.append(i)

	print(len(images)-len(valid), 'unsafe images out of', len(images))
	return valid

# filters images based on their forces standard deviations
def trim(images, nstd):
	valid = []
	for i in images:
		forces = i.get_forces()
		magnitudes = np.linalg.norm(forces, axis=1)
		threshold = np.mean(magnitudes) + nstd * np.std(magnitudes)
		if all(m < threshold for m in magnitudes):
			valid.append(i)

	print(len(images)-len(valid), 'forces trimed out of', len(images))
	return valid

# filters images based on energetic degeneracy, also assures unit cell bounds
def prune(images, decimals):
	#filter by rounding total or atomization energies
	AE = [im2ae(i) for i in images]
	unique_index = np.unique(np.round(AE, decimals), return_index=True)[1]
	unique = [images[i] for i in unique_index]
	sort_index = np.argsort([i.get_potential_energy() for i in unique])
	valid = [unique[i] for i in sort_index]
	
	print(len(images)-len(valid), 'energies pruned out of', len(images))
	return valid

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

def compare(directory, images, limit):

	#Link model to Calc
	calculator = LoadSchNetCalc(directory)

	#Get split indicies [train, val, test]
	splitfile = np.load(os.path.join(directory, 'split.npz'))
	splits = [splitfile[i] for i in splitfile]

	valid = []
	invalid = []
	measures = []
	for s in splits:

		trueE = []; predE = []
		trueF = []; predF = []

		for i in tqdm(range(len(s)), desc='comparing values'):
			image = images[s[i]]
			tE = image.get_potential_energy()
			tF = image.get_forces()
			copy = image.copy()
			copy.set_calculator(calculator)
			pE = copy.get_potential_energy()
			pF = copy.get_forces()
			
			if np.abs(tE-pE) > limit:
				invalid.append(image)
				continue
			
			valid.append(image)
			trueE.append(tE)
			trueF.append(tF)
			predE.append(pE)
			predF.append(pF)
		
		trueE = np.array(list(trueFlat(trueE)))
		predE = np.array(list(trueFlat(predE)))
		trueF = np.array(list(trueFlat(trueF)))
		predF = np.array(list(trueFlat(predF)))

		MAEe = np.mean(np.abs(trueE-predE))
		MAEf = np.mean(np.abs(trueF-predF))

		measures.append([trueE, predE, trueF, predF, MAEe, MAEf])
	
		write(directory+'valid.db', valid)
		write(directory+'invalid.db', invalid)
	
	return measures

###------------------------------------------
### SCHNET TRAINING FUNCTIONS
###------------------------------------------

def wait4db(file_path, stability_time=10, check_interval=1, timeout=5*60):

	t0 = time.time()
	last_size = -1
	stable_start = None

	while True:
		if os.path.exists(file_path):
			current_size = os.path.getsize(file_path)
			if current_size == last_size:
				if stable_start is None:
					stable_start = time.time()
				elif time.time() - stable_start >= stability_time:
					print(f"File {file_path} is stable.")
					return
			else:
				last_size = current_size
				stable_start = None
		if time.time() - t0 > timeout:
			raise TimeoutError(f"File {file_path} did not stabilize within {timeout} s")
		time.sleep(check_interval)

@rank_zero_only
def create_database(directory, images):

	new_path = os.path.join(directory, 'new_dataset.db')
	try: os.remove(new_path)
	except Exception: pass

	property_list = []
	for i in images:
		energy = np.array([i.get_potential_energy()])
		forces = np.array(i.get_forces())
		property_list.append({'energy': energy, 'forces': forces})

	new_dataset = ASEAtomsData.create(
		new_path, 
		distance_unit='Ang',
		property_unit_dict={'energy': 'eV', 'forces': 'eV/Ang'},
	)
	new_dataset.add_systems(property_list, images)
	print("[Rank 0] Database created.")

def build_datamodule(directory, images, bs, co, prepare=True):

	new_path = os.path.join(directory, 'new_dataset.db')
	DM = AtomsDataModule(
		new_path,
		split_file=os.path.join(directory, 'split.npz'),
		batch_size=bs, 
		val_batch_size=bs, 
		test_batch_size=bs,
		num_train=int(0.7*len(images)), 
		num_val=int(0.15*len(images)), 
		num_test=int(0.15*len(images)),
		load_properties=['energy', 'forces'],
		transforms=[
			spk.transform.ASENeighborList(cutoff=co),
			spk.transform.RemoveOffsets('energy', remove_mean=True, remove_atomrefs=False),
			spk.transform.CastTo32()
		],
		num_workers=2, #may require manual adjustment
		pin_memory=True if torch.cuda.is_available() else False,
	)
	if prepare: DM.prepare_data()
	DM.setup()
	print(f"[Rank {os.environ.get('LOCAL_RANK', 0)}] DataModule set up.")
	return DM

def myModel(parameters, scheduler):
	
	radial_basis = spk.nn.GaussianRBF(n_rbf=parameters['gaussians'], cutoff=parameters['cut_off'])
	schnet = spk.representation.SchNet(
		n_atom_basis=parameters['features'], 
		n_interactions=parameters['interactions'],
		radial_basis=radial_basis,
		cutoff_fn=spk.nn.CosineCutoff(parameters['cut_off'])
	)
	
	pred_energy = spk.atomistic.Atomwise(n_in=parameters['features'], output_key='energy')
	pred_forces = spk.atomistic.Forces(energy_key='energy', force_key='forces')
	
	nnpot = spk.model.NeuralNetworkPotential(
		representation=schnet,
		input_modules=[spk.atomistic.PairwiseDistances()],
		output_modules=[pred_energy, pred_forces],
		postprocessors=[
			spk.transform.CastTo64(),
			spk.transform.AddOffsets('energy', add_mean=True, add_atomrefs=False)
		]
	)
	
	output_energy = spk.task.ModelOutput(
		name='energy',
		loss_fn=torch.nn.MSELoss(),
		loss_weight=parameters['energy_weight'],
		metrics={"MAE": torchmetrics.MeanAbsoluteError()}
	)
	output_forces = spk.task.ModelOutput(
		name='forces',
		loss_fn=torch.nn.MSELoss(),
		loss_weight=parameters['force_weight'],
		metrics={"MAE": torchmetrics.MeanAbsoluteError()}
	)
	
	task = spk.task.AtomisticTask(
		model=nnpot,
		outputs=[output_energy, output_forces],
		optimizer_cls=torch.optim.AdamW,
		optimizer_args={'lr': parameters['learning_rate']},
		scheduler_cls=scheduler['s_class'],
		scheduler_monitor=scheduler['s_metric'],
		scheduler_args=scheduler['s_args'],
	)
	return task

def myTrainer(directory, epochs):
	csv_logger = pl.loggers.CSVLogger(save_dir=directory, name='csv_logs')
	callbacks = [
		spk.train.ModelCheckpoint(
			model_path=os.path.join(directory, 'best_inference_model'),
			save_top_k=5,
			monitor='val_loss'
		),
		pl.callbacks.EarlyStopping(monitor='val_loss', patience=200),
	]

	trainer = pl.Trainer(
		accelerator='gpu' if torch.cuda.is_available() else 'cpu',
		devices=torch.cuda.device_count(),
		num_nodes=int(os.getenv("SLURM_NNODES", 2)),
		strategy='ddp',
		precision=16,
		callbacks=callbacks,
		logger=csv_logger,
		log_every_n_steps=10,
		default_root_dir=directory,
		max_epochs=epochs,
	)
	return trainer

###------------------------------------------
### PLOTTING FUNCTIONS
###------------------------------------------

def stoiPlots(Stois, name):
	
	f, (ax1, ax2) = plt.subplots(1, 2, figsize=(6,3))
	colors = plt.cm.viridis(np.linspace(0, 1, 6))
	E = [chemical_symbols[i] for i in list(trueFlat(Stois))]
	hist, bins, patch = ax1.hist(np.sort(E), bins=np.arange(7)-0.5, width=1, edgecolor='k')

	for i in range(6):
		patch[i].set_facecolor(colors[i])
	
	ax1.set_xticks(np.arange(6))
	ax1.set_title('Elemental Ratios')
	ax1.set_ylabel('count')
	ax1.set_xlabel('element')

	counts = [list(np.unique(s, return_counts=True)[1]) for s in Stois]
	
	s = len(counts)
	n = len(counts[0])

	normalized = [[x/sum(i) for x in i] for i in counts]
	
	values = [CoM(i) for i in normalized]
	index = np.argsort(values)
	sort = [normalized[i] for i in index]
	
	base_sorted = np.zeros(s)
	colors = plt.cm.viridis(np.linspace(0, 1, n))
	for i, inner_list in enumerate(zip(*sort)):
		ax2.bar(range(s), inner_list, bottom=base_sorted, color=colors[i], edgecolor=colors[i])
		base_sorted += np.array(inner_list)
	
	ax2.yaxis.tick_right()
	ax2.yaxis.set_label_position('right')
	ax2.set_ylim(0,1)
	ax2.set_xticks(np.linspace(0,len(Stois),6))
	ax2.set_ylabel('ratio')
	ax2.set_xlabel('index')
	ax2.set_title('Stoichiometric Ratios')
	
	plt.savefig(name+'.png', dpi=200, bbox_inches='tight')
	plt.clf()

import matplotlib.ticker as ticker

def trainplot(name, save=False):

    format = ticker.FormatStrFormatter('%.3f')

    data = read_csv('NNs/'+name+'/csv_logs/version_0/metrics.csv')
	data.fillna(method='ffill', inplace=True)
    epochs = data['epoch']
    metrics = [[data['train_loss'], data['val_loss']],
               [data['train_energy_MAE'], data['val_energy_MAE']],
               [data['train_forces_MAE'], data['val_forces_MAE']]]
    skip = int(0.05*len(epochs))
    
    fig, axes = plt.subplots(3, 1, figsize=(5, 5), sharex=True)
    
    labs = ['train', 'valid']
    colors = ['b', 'g', 'r']
    ylabs = ['loss', 'energy', 'forces']
    for i in range(3):
        ax = axes[i]
        for j in range(2):
            final = np.round(metrics[i][j].iloc[-1], 3)
            ax.plot(epochs[skip:], metrics[i][j][skip:], color=colors[i], 
                    alpha=0.5*(j+1), label=labs[j]+' : '+str(final))
            ax.set_ylabel(ylabs[i])
            ax.grid(ls=':')
            ax.legend(loc='upper right')
            ax.yaxis.set_major_formatter(format)
    	plt.tight_layout(pad=1)
    
    if save: plt.savefig('NNs/'+name+'/train.png')
    else: plt.show()
    plt.clf()

def barplot(name, measures, save=False):

	Max = np.max(measures)

	plt.figure(figsize=(5,5))
	plt.bar(np.arange(3)-0.2, [measures[0],measures[2],measures[4]], 
		label='energy', width=0.4, color='b', edgecolor='k', alpha=0.75, zorder=2)
	plt.bar(np.arange(3)+0.2, [measures[1],measures[3],measures[5]], 
		label='forces', width=0.4, color='r', edgecolor='k', alpha=0.75, zorder=2)

	t1 = plt.text(0-0.1, 0.1*Max, '70%', fontsize=10)
	t1.set_bbox(dict(facecolor='white', alpha=0.75, edgecolor='k'))
	t2 = plt.text(1-0.1, 0.1*Max, '15%', fontsize=10)
	t2.set_bbox(dict(facecolor='white', alpha=0.75, edgecolor='k'))
	t3 = plt.text(2-0.1, 0.1*Max, '15%', fontsize=10)
	t3.set_bbox(dict(facecolor='white', alpha=0.75, edgecolor='k'))

	plt.xticks(np.arange(3), ['train', 'valid', 'test'])
	plt.ylabel('MAE')
	plt.legend(); plt.grid(ls=':', zorder=0); plt.tight_layout(pad=1)
	if save: plt.savefig('NNs/'+name+'/bar.png')
	else: plt.show()
	plt.clf()

def ratioplot(name, kind, x, y, title, save=False):

	line = [np.min(y), np.max(y)]
	if kind=='energy' or kind=='relax': unit=' (eV)'
	elif kind=='force': unit=' (eV/atom/\u212B)'
	else: print('[either energy, forces, or relax]'); return

	MAE = np.round(np.mean(np.abs(x-y)), 4)

	xy = np.vstack([x,y])
	z = gaussian_kde(xy)(xy)
	idx = z.argsort()
	x, y, z = x[idx], y[idx], z[idx]
	norm = plt.Normalize()
	colors = plt.cm.turbo(norm(z))

	plt.figure(figsize=(5,5))
	plt.plot(line, line, c='black', linestyle='dotted', 
		linewidth=0.75, zorder=3)
	plt.scatter(x, y, facecolor=colors, edgecolor='None', 
		s=20, alpha=0.75, label='MAE '+str(MAE), zorder=2)
	plt.xlabel('DFT '+kind+unit)
	plt.ylabel('NN '+kind+unit)
	plt.legend(); plt.grid(ls=':', zorder=0); plt.tight_layout(pad=1)
	if save: plt.savefig('NNs/'+name+'/ratio-'+kind+'.png')
	else: plt.show()
	plt.clf()

def deltaplot(name, kind, y1, y2, title, save=False):

	if kind=='energy' or kind=='relax': unit=' (eV)'
	elif kind=='force': unit=' (eV/atom/\u212B)'
	else: print('[either energy, forces, or relax]'); return

	MAE = np.round(np.mean(np.abs(y1-y2)), 4)

	index = np.argsort(y1)
	y1 = y1[index]; y2 = y2[index]
	line = [np.min(y2), np.max(y2)]

	fig, ax = plt.subplots(figsize=(5,5), nrows=2, ncols=1, sharex=True)
	x = np.arange(len(y1))

	ax[0].scatter(x, y1, label='DFT', marker='o', s=100, c='r', alpha=0.5, zorder=2)
	ax[0].scatter(x, y2, label='NN', s=1, c='b', alpha=0.5, zorder=3)
	ax[0].legend()
	ax[0].set_ylabel(unit)
	ax[0].tick_params(labelbottom=False)    
	ax[0].grid(ls=':', zorder=0)

	delta = (y1-y2)
	ax[1].plot(delta, c='r', linewidth=0.5, alpha=1.0, zorder=2)
	ax[1].scatter(x,delta,label='MAE '+str(MAE), facecolors='w',
		alpha=0.75, edgecolors='k', s=5, zorder=3)
	ax[1].set_ylabel('$DFT_i - NN_i$'+unit)
	ax[1].legend(loc='upper left')
	ax[1].grid(ls=':', zorder=0)

	plt.subplots_adjust(wspace=0.0, hspace=0.05)
	plt.xlabel('species number (sorted by energy)')
	plt.tight_layout(pad=1)
	if save: plt.savefig('NNs/'+name+'/delta-'+str(kind)+'.png')
	else: plt.show()
	plt.clf()

def histoplot(name, kind, x, y, title, save):

	if kind=='energy' or kind=='relax': 
		c = 'b'; unit=' (eV/atom)'
	elif kind=='force': 
		c = 'r'; unit=' (eV/atom/\u212B)'
	else: print('[either energy, forces, or relax]'); return

	lab = True
	delta = x-y
	ME = np.round(np.mean(delta),3)
	if abs(ME) < 0.001: lab = False

	dmin, dmax = round(np.min(np.abs(delta)),3), round(np.max(np.abs(delta)),3)
	buffer = (dmax-dmin)/10
	print('absolute', kind, 'difference range =', '['+str(dmin)+','+str(dmax)+']')
	res = 100

	#Fit gaussian to data
	#rule of thumb for covariance factor
	h = 1.06*np.std(delta)*len(delta)**(-0.2)
	distribution = gaussian_kde(delta)
	distribution.covariance_factor = lambda: h
	distribution._compute_covariance()
	x = np.linspace(np.min(delta)-buffer, np.max(delta)+buffer, res)
	density = distribution(x)
	density = density/np.max(density)

	#Plotting
	plt.figure(figsize=(5,5))
	plt.vlines(0, 0, 1, color='k', alpha=0.75, lw=1, zorder=3)
	if lab: 
		plt.vlines(ME, 0, 1, ls='--', color='k', alpha=0.75,
					label='ME '+str(ME), lw=1, zorder=3)
	else: 
		plt.vlines(ME, 0, 1, ls='--', color='k', alpha=0.75,
					label='ME $\\approx$ 0.0', lw=1, zorder=3)
	plt.plot(x, density, linewidth=1, c=c, zorder=2)
	plt.fill_between(x, density, color=c, lw=1, alpha=0.5, zorder=2)

	plt.xlabel('$DFT_i-NN_i$ '+kind+unit)
	plt.ylabel('Normalized Density')
	plt.legend(); plt.grid(ls=':'); plt.tight_layout(pad=1)
	if save: plt.savefig('NNs/'+name+'/histo-'+kind+'-.png')
	else: plt.show()
	plt.clf()

###------------------------------------------
