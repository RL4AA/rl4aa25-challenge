<!-- [![DOI](https://zenodo.org/badge/700362904.svg)](https://zenodo.org/doi/10.5281/zenodo.10886639)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0) -->

# Tutorial on Meta-Reinforcement Learning and GP-MPC at the RL4AA'24 Workshop

This repository contains the material for tutorial and challenge of the [RL4AA'25](https://indico.scc.kit.edu/event/4216/overview) event.

Homepage for RL4AA Collaboration: [https://rl4aa.github.io/](https://rl4aa.github.io/)

## Theory slides
- [Introduction to GP-MPC](https://indico.scc.kit.edu/event/4216/sessions/4250/#20250402), Simon Hirländer

## Tutorial and challenge: optimisers vs GP-MPC at ARES

- GitHub repository containing the material: [https://github.com/RL4AA/rl4aa25-tutorial](https://github.com/RL4AA/rl4aa25-tutorial)
- Tutorial in slide form: [here](https://rl4aa.github.io/rl4aa25-tutorial/)

## Getting started

- First, download the material to your computer by cloning the repository:
`git clone https://github.com/RL4AA/rl4aa25-tutorial.git`
- If you don't have git installed, you can click on the green button that says "Code", and choose to download it as a `.zip` file.
- You will find the jupyter notebooks in the `notebooks` folder.

## Setting-up your virtual environment
### Using Conda

- If you don't have conda installed already, you can install the `miniconda` as [described here](https://docs.conda.io/projects/miniconda/en/latest/miniconda-install.html).
- We recommend to install `miniconda` the day beforehand to avoid network overload during the tutorial &#x2757; &#x2757;

Once `miniconda` is installed run this command in your terminal:

```bash
conda env create -f environment.yml
```

This should create a virtual environment named `rl25-tutorial` and install the necessary packages inside.

Afterwards, activate the environment using

```bash
conda activate rl25-tutorial
```

### Using venv

If you don't have conda installed, you can create the virtual env with:

```bash
python3 -m venv rl-tutorial
```

and activate the env with `$ source <venv>/bin/activate` (bash) or `C:> <venv>/Scripts/activate.bat` (Windows)

Then, install the packages with `pip` within the activated environment

```bash
python -m pip install -r requirements.txt
```

Afterwards, you should be able to run the provided scripts.

## Folder Structure

- `src` Contains the source code for the RL environment and the GP-MPC controller
  - `src/environments/ea` contains the gymnasium environment for the ARES-EA transverse tuning task
  - `src/reward` contains files for the reward engineering (combination of rewards, transformation, ...)
  - `src/wrappers` contains custom wrappers for the EA environment
    - `src/wrappers/ea_mpc_episode_with_plotting` contains the wrapper for running GP-MPC (mainly it creates the visualization)
  - `src/train` contains scripts to train a default PPO agent to solve the task (can be used as a benchmark for evaluating MPC controller)
  - `src/gpmpc` contains the GP-MPC controller
    - `src/gpmpc/control_object` implements the controller
      - `gp_models` implements the GP model for modeling the transition of the environment
      - `gp_mpc_controller` implements the controller
    - `src/gpmpc/utils` contains utility functions for the GP-MPC controller
- `data/trail.yaml` contains the pre-selected task configurations for evaluation
- `config/` config files for running GP-MPC control

## Part 1 Optimisers

## Part 2 GP-MPC
Simply run `python -m src.run_gpmpc --config=config/config_ea.yaml`

You can run experiments with different settings by creating a new config file.

- `env` section handles how the ARES-EA is created and wrapped
- the other sections defines the behavior of the GP-MPC controller


## Citing the tutorial
<!-- 
This tutorial is uploaded to [Zenodo](https://zenodo.org/doi/10.5281/zenodo.10886639).
Please use the following DOI when citing this code:

```bibtex
@software{hirlaender_2024_10887397,
    title        = {{Tutorial on Meta-Reinforcement Learning and GP-MPC at the RL4AA'24 Workshop}},
    author       = {Hirlaender, Simon and Kaiser, Jan and Xu, Chenran and Santamaria Garcia, Andrea},
    year         = 2024,
    month        = mar,
    publisher    = {Zenodo},
    doi          = {10.5281/zenodo.10887397},
    url          = {https://doi.org/10.5281/zenodo.10887397},
    version      = {v1.0.2}
} -->




## To be removed before publishing: NOTES FOR DEVS

### Code Formatting

Please install `black`, `isort`, and `flake8` for cohesive code formatting.

If you are using VS Code, this can be done by installing the native extensions.

- VS Code → View → Extensions → [Black Formatter](https://marketplace.visualstudio.com/items?itemName=ms-python.black-formatter)
- VS Code → View → Extensions → [isort](https://marketplace.visualstudio.com/items?itemName=ms-python.isort)
  - In the `isort` extension setting, add `--profile black` to the Args
- VS Code → View → Extensions → [flake8](https://marketplace.visualstudio.com/items?itemName=ms-python.flake8)
  - In the `flake8` extension setting, add `--max-line-length=88` and `--extend-ignore=E203, E701, W503` to the Args

You can check Format on Save so that the formatting is done automatically. (`"editor.formatOnSave": true`)

### Look into the environment

- `make_env` creates the wrapped env for GP-MPC control

Right now the wrapped ARES-EA environment has:

- 5-d action: delta / direct settings to the magnets (q1, q2, cv, q3, ch)
- 13-d observation: 4-d current beam + 5d current magnet + 4-d target beam
  - 4-d current beam $(\mu_x, \sigma_x, \mu_y, \sigma_y)$
  - 5-d current magnet setting $(k_{Q1}, k_{Q2}, \theta_{CV}, k_{Q3}, \theta_{CH})$
  - 4-d target beam $(\mu_x, \sigma_x, \mu_y, \sigma_y)$

The normalization is done using the wrappers by default

Now it's a bit of a hack to make it compatible (should be changed later, see below the todos):

- The weights of the observation for cost calculation are set to 0.0 for target_beam and magnet_setting

### Code To-Dos

Functionalities:

- Move the normalization from GP-MPC controller to Env Wrapper
  - i.e. the methods `to_normed_obs_tensor`, `to_normed_var_tensor`, `to_normed_action_tensor`, `denorm_action` should be no longer needed when `RescaleAction` and `RescaleObservation` wrappers are used?; And change the calculations in `gp_mpc_controller`
- Support different reward/cost mode
  - `compute_cost` should take additional argument / have different reward mode
  - `compute_cost_unnormalized` will not be needed when observation and action are normalized by env wrappers
  - The cost mode should better be separated, for the sake of tutorial (reward engineering...)
- Remove the target beam from the current observation,
  - this can be done via an extra wrapper on the Env
  - The cost calculation of GP-MPC should take the `env.target_beam`, instead of statically from the config file
- Implement proper model saving (this was removed in the RL4AA'24 tutorial?)
- Fix a Cheetah version for the code / update the env with cheetah 0.7.0 (now 0.6.3 works fine)
- Refactor `LivePlotSequential` in `gpmpc/utils/utils.py`, now the plot gives generic legends, and plot all the states including the target_beam.
- Refactor the `observation_space` limit for current beam (possibly also in `RescaleObservation` wrapper) (it was `[-inf, inf]` otherwise the scaling in gp-mpc and plotting now will complain), now it's rather arbitrarily chosen to be `[-5e-3,0,-5e-3,0] -> [5e-3,5e-3,5e-3,5e-3]`

Maintainability

- Add Typing in all the GP-MPC part;
- Fill the missing doc-strings in the GP-MPC part;
- Refactor variable names to be understandable
- Implement proper unit tests; At a later stage, add the tests to CI/CDs
- Improve the plotting in `EAMpcEpisodeWithPlotting` wrapper

### Some general remark

GP-MPC aims to plan the best trajectory, which brings the **state** to a **target-state**
GP model predicts the transition probability (state_{t}, action) → (state_{t+1})
So it only works with problems, for which the reward can be calculated entirely form the current state (Luckily this is the case for most accelerator tuning tasks we focus on)

As far as I understand now, the math has been derived mainly for cost consisting of

- linear combination of states
- exponential combination (like in the PILCO paper)

Will this work for a general cost function definition?

### Further general questions to be investigated

- Does it still make sense to keep the magnet settings in the observation?
  - At least the prediction from (current magnet, action) → (next magnet) should be deterministic and not modeled by the GP?
  - Customize the GP model in forward? It should actually just model the magnet strength → beam, the transition probability can be derived form it?

## Other Resources

For GP-MPC running on AWAKE, please visit the RL4AA'24 tutorial

- <https://github.com/RL4AA/rl4aa24-tutorial>

For more examples and details on the ARES RL environment, c.f.

- Paper: [Reinforcement learning-trained optimisers and Bayesian optimisation for online particle accelerator tuning](https://www.nature.com/articles/s41598-024-66263-y)
- Code repository: <https://github.com/desy-ml/rl-vs-bo>

Now we add new tests

