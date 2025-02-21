#!/usr/bin/env python3

import os
import sys
import time
import argparse
import numpy as np
import torch
import pytorch_lightning as pl
import torch.distributed as dist

from programs import (
    db2im, create_database, wait4db,
    build_datamodule, myModel, myTrainer,
)
import optuna

def mytrain(gauss, feat, inter, run_id):
    print("Run =", run_id)
    directory = os.path.join('NNs', name + '-' + str(run_id))
    os.makedirs(directory, exist_ok=True)

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

    # DATA CURATION: Only rank 0 creates the database.
    if int(os.environ.get("LOCAL_RANK", 0)) == 0:
        create_database(directory, images)
    if dist.is_initialized():
        dist.barrier()

    db_file = os.path.join(directory, 'new_dataset.db')
    wait4db(db_file, stability_time=10, check_interval=1, timeout=300)

    current_rank = dist.get_rank() if dist.is_initialized() else 0
    print(f"[Rank {current_rank}] Database file found: {db_file}")

    # Build the datamodule with the adjusted function.
    dm = build_datamodule(directory, images, batch_size, parameters['cut_off'], prepare=True)
    task = myModel(parameters, scheduler)
    trainer = myTrainer(directory, epochs)

    t1 = time.time()
    trainer.fit(task, datamodule=dm)
    t2 = time.time()

    print('Training complete --', round((t2 - t1) / 60, 2), 'min')

    validation_loss = trainer.callback_metrics.get('val_loss')
    if validation_loss is None:
        raise ValueError('Validation loss not found')
    return validation_loss.item()


def objective(trial):
    # Only rank 0 runs the hyperparameter study.
    gauss = trial.suggest_int('gauss', 10, 50, step=10)
    feat = trial.suggest_int('feat', 100, 500, step=100)
    inter = trial.suggest_int('inter', 2, 10, step=2)
    print(f"Trial {trial.number} hyperparameters: gauss={gauss}, feat={feat}, inter={inter}")
    val_loss = mytrain(gauss, feat, inter, trial.number)
    return val_loss


if __name__ == "__main__":
    # Get name and database path.
    name, path = sys.argv[1], sys.argv[2]
    images = db2im(path)
    index = np.random.permutation(len(images))
    images = [images[i] for i in index]

    batch_size = 16
    epochs = 100

    current_rank = 0
    if dist.is_initialized():
        current_rank = dist.get_rank()

    best_params = None
    if current_rank == 0:
        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=50, n_jobs=1)
        best_params = study.best_params
        print("Best hyperparameters:", best_params)
    else:
        best_params = None

    if dist.is_initialized():
        best_params_list = [best_params]
        dist.broadcast_object_list(best_params_list, src=0)
        best_params = best_params_list[0]

    # Final training using best hyperparameters.
    best_gauss = best_params['gauss']
    best_feat = best_params['feat']
    best_inter = best_params['inter']
    final_loss = mytrain(best_gauss, best_feat, best_inter, run_id=-1)
    print("Final validation loss:", final_loss)

###/END
