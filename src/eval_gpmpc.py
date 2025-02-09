import argparse

import numpy as np
import torch
import yaml
from gymnasium.wrappers import (
    FilterObservation,
    FlattenObservation,
    RescaleAction,
    TimeLimit,
)

from src.environments import ea
from src.gpmpc.control_objects.gp_mpc_controller import GpMpcController
from src.gpmpc.cost_functions.base_cost import QuadraticCostFunction
from src.gpmpc.utils.utils import close_run
from src.trial import Trial, load_trials
from src.wrappers import RecordEpisode, RescaleObservation


def try_problem(
    trial_index: int,
    trial: Trial,
    config: dict,
    write_data: bool = True,
) -> None:
    # Create the environment
    env_config = config["env"]
    wrapper_config = config["env_wrapper"]
    env_config["backend_args"]["incoming_mode"] = trial.incoming_beam.astype(np.float32)
    env_config["backend_args"]["misalignment_mode"] = trial.misalignments.astype(
        np.float32
    )
    env_config["magnet_init_mode"] = trial.initial_magnets.astype(np.float32)
    env_config["target_beam_mode"] = trial.target_beam.astype(np.float32)
    env = ea.TransverseTuning(**env_config)
    env = TimeLimit(env, wrapper_config["max_episode_steps"])
    if write_data:
        env = RecordEpisode(
            env,
            save_dir=(f"data/eval_gpmpc/problem_{trial_index:03d}"),
        )
    if (
        wrapper_config["normalize_observation"]
        and not wrapper_config["running_obs_norm"]
    ):
        env = RescaleObservation(env, 0, 1)
    if wrapper_config["rescale_action"]:
        env = RescaleAction(env, 0, 1)
    env = FilterObservation(env, ["beam"])
    env = FlattenObservation(env)

    # Initialize the Controller
    cost_function = QuadraticCostFunction(
        target_state=torch.tensor(config["controller"]["target_state_norm"]),
        target_action=torch.tensor(config["controller"]["target_action_norm"]),
        weight_state_matrix=torch.diag(
            torch.tensor(config["controller"]["weight_state"])
        ),
        weight_action_matrix=torch.diag(
            torch.tensor(config["controller"]["weight_action"])
        ),
    )

    ctrl_obj = GpMpcController(
        observation_space=env.observation_space,
        action_space=env.action_space,
        params_dict=config,
        cost_function=cost_function,
    )

    num_repeat_actions = config["controller"]["num_repeat_actions"]
    random_actions_init = config["random_actions_init"]

    (
        ctrl_obj,
        env,
        obs,
        action,
        cost,
        obs_prev_ctrl,
        _,
        _,
        _,
    ) = init_control(
        ctrl_obj=ctrl_obj,
        env=env,
        random_actions_init=random_actions_init,
        num_repeat_actions=num_repeat_actions,
    )

    info_dict = None
    terminated = False
    truncated = False
    # Perform the control loop
    iter_ctrl = 0
    while not terminated and not truncated:
        if iter_ctrl % num_repeat_actions == 0:
            # Store the previous action and cost
            if info_dict is not None:
                predicted_state = info_dict["predicted states"][0]
                predicted_state_std = info_dict["predicted states std"][0]
                check_storage = True
            else:
                predicted_state = None
                predicted_state_std = None
                check_storage = False
            ctrl_obj.add_memory(
                obs=obs_prev_ctrl,
                action=action,
                obs_new=obs,
                reward=-cost,
                check_storage=check_storage,
                predicted_state=predicted_state,
                predicted_state_std=predicted_state_std,
            )

            # Compute the action
            print("Cost: " + str(cost))
            print("Step: " + str(iter_ctrl))
            action, info_dict = ctrl_obj.compute_action(
                obs_mu=obs, wait_for_training=True
            )
            if config["verbose"]:
                for key in info_dict:
                    print(key + ": " + str(info_dict[key]))

        # perform action on the system
        obs_new, reward, terminated, truncated, _ = env.step(action)
        cost, cost_var = ctrl_obj.compute_cost_unnormalized(obs, action)
        # set obs to previous control
        obs_prev_ctrl = obs
        obs = obs_new
        iter_ctrl += 1
        # print("time loop: " + str(time.time() - time_start) + " s\n")
    close_run(ctrl_obj=ctrl_obj, env=env)


def init_control(ctrl_obj, env, random_actions_init, num_repeat_actions=1):
    """
    Initializes the control environment with random actions and updates visualization
    and memory.

    Args:
        ctrl_obj (GpMpcController): Control object for computing cost and managing
            memory.
        env (gym.Env): Gym environment for obtaining observations and applying actions.
        live_plot_obj: Object for real-time 2D graph visualization, with an `update`
            method.
        rec: Real-time environment visualization object, with a `capture_frame` method.
        params_general (dict): General parameters
            (render_env, save_render_env, render_live_plots_2d).
        random_actions_init (int): Number of initial random actions.
        costs_tests (np.array): Array to store costs for analysis
            (shape: (num_runs, num_timesteps)).
        idx_test (int): Current test index.
        num_repeat_actions (int): Number of consecutive constant actions;
            affects memory storage.
    """
    obs_lst, actions_lst, rewards_lst = [], [], []
    obs, _ = env.reset()
    action, cost, obs_prev_ctrl = None, None, None
    terminated = False
    truncated = False

    # Set the target state
    new_target_state = env.get_wrapper_attr("normalized_target_beam")(
        min_observation=ctrl_obj.obs_space.low,
        max_observation=ctrl_obj.obs_space.high,
    )
    ctrl_obj.cost_function.set_target_state(torch.tensor(new_target_state))

    # Perform random actions to initialize the memory
    for idx_action in range(random_actions_init):
        if action is None:
            action = env.action_space.sample()
        elif idx_action % num_repeat_actions == 0:
            if ctrl_obj.limit_action_change:
                # Sample action within the limit_action_range
                delta_action = (
                    (np.random.rand(*action.shape) - 0.5)
                    * 2
                    * ctrl_obj.max_change_action_norm
                )
                normed_action = ctrl_obj.to_normed_action_tensor(action) + delta_action
                normed_action = torch.clamp(normed_action, 0, 1)
                action = ctrl_obj.denorm_action(normed_action).numpy()
            else:
                action = env.action_space.sample()

            if obs_prev_ctrl is not None and cost is not None:
                ctrl_obj.add_memory(
                    obs=obs_prev_ctrl,
                    action=action,
                    obs_new=obs,
                    reward=cost,
                    check_storage=False,
                )
        if terminated or truncated:
            obs, _ = env.reset()
        obs_new, _, terminated, truncated, _ = env.step(action)
        obs_prev_ctrl = obs
        obs = obs_new
        cost, _ = ctrl_obj.compute_cost_unnormalized(obs_new, action)
        # cost=reward!
        cost = cost

        # Update lists for visualization
        obs_lst.append(obs)
        actions_lst.append(action)
        rewards_lst.append(cost)

        # Store the last action for potential future use
        ctrl_obj.action_previous_iter = torch.tensor(action)

    return (
        ctrl_obj,
        env,
        obs,
        action,
        cost,
        obs_prev_ctrl,
        obs_lst,
        actions_lst,
        rewards_lst,
    )


def main(args):
    trials = load_trials(args.trials_file)
    with open(args.config, "r") as file:
        config = yaml.safe_load(file)

    for trial_index in args.trial_index:
        try_problem(trial_index, trials[trial_index], config)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--trials_file",
        type=str,
        default="data/trials.yaml",
        help="Path to the trials file",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/config_ea_evaluate.yaml",
        help="Path to the configuration file",
    )
    parser.add_argument(
        "--trial_index",
        type=list,
        default=[0],
        help="Indices of the trials to run",
    )
    args = parser.parse_args()
    main(args)
