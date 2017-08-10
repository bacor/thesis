This repository contains everything related to my thesis 'The Bayesian Naming Game.'

## Requirements
All experiments were run in an Anaconda environment; a dump is found in `conda`. With that environment, all should run fine. 

In many cases the data directly underlying the plot is included in the repo, in the experiment's directory. The 'raw' data from which the plot data was extracted can also be downloaded from ()[]. All files assume the `results` directory is in the root directory of this repo.

## Structure
* `src/` Code for all experiments etc
* `figures` All figures: fully generated inside Jupyter notebook, i.e. requires no external data.
* `experiments/` All experiments in separate directories in which one typically finds:
  *  A bash script to run the experiment. This often also contains a short description, but for the full description one is referred to the thesis. The experiment can be identified by its code. 
  *  A Jupyter notebook with analyses and, mainly, plots
  *  A csv file with data used to generate the plots
  *  A JSON file with experiment parameters
  *  A pdf/png with the resulting figure

## Missing experiments
Some experiments are (still) missing
* All che counting games (ch6)
* The WebPPl replications of Kalish & Griffiths. The analyses *are* included.

