# RL4AA'25 Workshop Tutorial

Tutorial for the RL4AA'25 Workshop. Indico link to the event: [https://indico.scc.kit.edu/event/4216/]

## Installation

In virtual environment, run `pip install -r requirements.txt`

To-do (this can happen at the last stage before workshop): create a conda `environment.yml` file for better reproducibility, copy the instruction for last years here.

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

## AWAKE-compatible GP-MPC

Now the code is undergoing some refactoring for it to work with ARES-EA Cheetah environment. For reference, you can go back to the previous commit to run it with AWAKE. [fb5139e](https://github.com/RL4AA/rl4aa25-tutorial/commit/fb5139e57022ae23d89d113a8c05b2a24ea9465c)

## Development Note

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
- Refactor `LivePlotSequential` in  `gpmpc/utils/utils.py`, now the plot gives generic legends, and plot all the states including the target_beam.
- Refactor the `observation_space` limit for current beam (possibly also in `RescaleObservation` wrapper) (it was `[-inf, inf]` otherwise the scaling in gp-mpc and plotting now will complain), now it's rather arbitrarily chosen to be `[-5e-3,0,-5e-3,0] -> [5e-3,5e-3,5e-3,5e-3]`
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
