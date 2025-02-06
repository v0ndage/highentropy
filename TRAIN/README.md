### Here we exemplify training CNNs via Pytorch Lightning on ASE Atoms objects with SchNetPack

This main directory is for training large NNs.
The OPT directory is for optimizing hyperparameters prior to training large networks.

Optimization utilizes Optuna: https://optuna.readthedocs.io/en/stable/index.html

Optuna takes an objective function like

```
def objective(trial):
  step = trial.number
  gauss = trial.suggest_int('gauss', 30, 80)
  feat = trial.suggest_int('feat', 100, 900)
  inter = trial.suggest_int('inter', 3, 9)
```
And uses Bayesian inference to try and minimize the loss function's parameter space.
It is iterated like this

```
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=50,  n_jobs=1)
```

Both the OPT and main TRAIN directories utilize programs.py to implement SchNetPack models.

After initializing hyperparameters, the basic training proceedure is this:

```
dm = myModule(directory, images, batch_size)
task = myModel(parameters, scheduler)
trainer = myTrainer(directory, epochs)
trainer.fit(task, datamodule=dm)
```
All functions starting with 'my' are defined in programs.py

The rest is simply utility functions and plotting functions.

##END
