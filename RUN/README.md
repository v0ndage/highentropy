## Example Usage:

Note 'iter' can be a number or a name,
but it is expected that there exists a DFT$iter directory.

```sbatch script iter db2dft.py DBs/2-H.db```
### wait till files are created
```sbatch runDFTs.sh iter 1 100```
### jobs should run
```sbatch clean.sh iter 1 100```

etc.
