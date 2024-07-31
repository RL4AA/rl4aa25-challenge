# RL4AA'25 Workshop Tutorial

Tutorial for the RL4AA'25 Workshop. Indico link to the event: [https://indico.scc.kit.edu/event/4216/]

## Installation

In virtual environment, run `pip install -r requirements.txt`

To-do (last stage before workshop): create a conda `environment.yml` file for better reproducibility, copy the instruction for last years here.

## Folder Structure

- `src` Contains the source code for the RL environment and the GP-MPC controller
  - `src/environments/ea` contains the gymnasium environment for the ARES-EA transverse tuning task
  - `src/reward` contains files for the reward engineering (combination of rewards, transformation, ...)
  - `src/wrappers` contains custom wrappers for the EA environment
  - `src/train` contains scripts to train a default PPO agent to solve the task
  - `src/gpmpc` contains the GP-MPC controller
    - `src/gpmpc/control_object` implements the controller
    - `src/gpmpc/utils` contains utility functions for the GP-MPC controller
- `data/trail.yaml` contains the pre-selected task configurations for evaluation
- `config/` config files for running GP-MPC control
