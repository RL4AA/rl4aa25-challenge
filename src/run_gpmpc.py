import argparse
import time

import gymnasium as gym
import torch
import yaml
from gymnasium.wrappers import (
    FilterObservation,
    FlattenObservation,
    RecordVideo,
    RescaleAction,
    TimeLimit,
)
from stable_baselines3.common.monitor import Monitor

from .environments import ea
from .gpmpc.control_objects.gp_mpc_controller import GpMpcController
from .gpmpc.cost_functions.base_cost import QuadraticCostFunction
from .gpmpc.utils.utils import close_run, init_control, init_visu_and_folders
from .wrappers import (
    EAMpcEpisodeWithPlotting,
    LogTaskStatistics,
    PlotEpisode,
    RescaleObservation,
)


def init_graphics_and_controller(env, num_steps, params_controller_dict):
    """Initialize the graphics and the controller object.

    :param env: The environment object.
    :param num_steps: The number of steps to run the simulation.
    :param params_controller_dict: The dictionary containing the parameters of the
    controller.
    :return: The `live_plot_obj` and the `ctrl_obj`.

    """
    live_plot_obj = init_visu_and_folders(
        env=env, num_steps=num_steps, params_controller_dict=params_controller_dict
    )

    cost_function = QuadraticCostFunction(
        target_state=torch.tensor(
            params_controller_dict["controller"]["target_state_norm"]
        ),
        target_action=torch.tensor(
            params_controller_dict["controller"]["target_action_norm"]
        ),
        weight_state_matrix=torch.diag(
            torch.tensor(params_controller_dict["controller"]["weight_state"])
        ),
        weight_action_matrix=torch.diag(
            torch.tensor(params_controller_dict["controller"]["weight_action"])
        ),
    )

    ctrl_obj = GpMpcController(
        observation_space=env.observation_space,
        action_space=env.action_space,
        params_dict=params_controller_dict,
        cost_function=cost_function,
    )

    return live_plot_obj, ctrl_obj


def main(args):
    with open(args.config, "r") as file:
        params_controller_dict = yaml.safe_load(file)

    num_steps = params_controller_dict["num_steps_env"]
    num_repeat_actions = params_controller_dict["controller"]["num_repeat_actions"]
    random_actions_init = params_controller_dict["random_actions_init"]

    env = make_env(config=params_controller_dict["env"])
    live_plot_obj, ctrl_obj = init_graphics_and_controller(
        env, num_steps, params_controller_dict
    )

    (
        ctrl_obj,
        env,
        live_plot_obj,
        obs,
        action,
        cost,
        obs_prev_ctrl,
        obs_lst,
        actions_lst,
        rewards_lst,
    ) = init_control(
        ctrl_obj=ctrl_obj,
        env=env,
        live_plot_obj=live_plot_obj,
        random_actions_init=random_actions_init,
        num_repeat_actions=num_repeat_actions,
    )

    info_dict = None
    terminated = False
    truncated = False
    # Perform the control loop
    for iter_ctrl in range(random_actions_init, num_steps):
        # Temporary: use time.sleep to simulate the time needed for the control loop
        time.sleep(0.5)
        # time.sleep(0.5)

        # time_start = time.time()
        # Repeat the action for `num_repeat_actions` steps, then compute a new action
        if iter_ctrl % num_repeat_actions == 0:
            if info_dict is not None:
                predicted_state = info_dict["predicted states"][0]
                predicted_state_std = info_dict["predicted states std"][0]
                check_storage = True
            else:
                predicted_state = None
                predicted_state_std = None
                check_storage = False
            # If num_repeat_actions != 1, the gaussian process models predict that
            # much steps ahead. For iteration k, the memory holds
            # obs(k - step), action (k - step), obs(k), reward(k)
            # Add memory is put before compute action because it uses data from
            # the step before
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
            action, info_dict = ctrl_obj.compute_action(obs_mu=obs)
            if params_controller_dict["verbose"]:
                for key in info_dict:
                    print(key + ": " + str(info_dict[key]))

        if terminated or truncated:
            obs, _ = env.reset()
            # Reset the target state for the cost calculation
            new_target_state = new_target_state = env.get_wrapper_attr(
                "normalized_target_beam"
            )(
                min_observation=ctrl_obj.obs_space.low,
                max_observation=ctrl_obj.obs_space.high,
            )
            ctrl_obj.cost_function.set_target_state(torch.tensor(new_target_state))
        # perform action on the system
        obs_new, reward, terminated, truncated, _ = env.step(action)
        cost, cost_var = ctrl_obj.compute_cost_unnormalized(obs, action)
        try:
            if live_plot_obj is not None:
                live_plot_obj.update(
                    obs=obs, cost=cost, action=action, info_dict=info_dict
                )
        except Exception as e:
            print("An error occurred when plotting:", str(e))
        # set obs to previous control
        obs_prev_ctrl = obs
        obs = obs_new
        # print("time loop: " + str(time.time() - time_start) + " s\n")

    # Close the environment
    input("Press Enter to close the environment...")
    close_run(ctrl_obj=ctrl_obj, env=env)


def make_env(
    config: dict,
    record_video: bool = False,
    plot_episode: bool = False,
    log_task_statistics: bool = False,
) -> gym.Env:
    env = ea.TransverseTuning(
        backend="cheetah",
        backend_args={
            "incoming_mode": config["incoming_mode"],
            "misalignment_mode": config["misalignment_mode"],
            "max_misalignment": config["max_misalignment"],
            "generate_screen_images": plot_episode,
        },
        action_mode=config["action_mode"],
        magnet_init_mode=config["magnet_init_mode"],
        max_quad_setting=config["max_quad_setting"],
        max_quad_delta=config["max_quad_delta"],
        max_steerer_delta=config["max_steerer_delta"],
        target_beam_mode=config["target_beam_mode"],
        target_threshold=config["target_threshold"],
        threshold_hold=config["threshold_hold"],
        clip_magnets=config["clip_magnets"],
        beam_param_transform=config["beam_param_transform"],
        beam_param_combiner=config["beam_param_combiner"],
        beam_param_combiner_args=config["beam_param_combiner_args"],
        beam_param_combiner_weights=config["beam_param_combiner_weights"],
        magnet_change_transform=config["magnet_change_transform"],
        magnet_change_combiner=config["magnet_change_combiner"],
        magnet_change_combiner_args=config["magnet_change_combiner_args"],
        magnet_change_combiner_weights=config["magnet_change_combiner_weights"],
        final_combiner=config["final_combiner"],
        final_combiner_args=config["final_combiner_args"],
        final_combiner_weights=config["final_combiner_weights"],
        render_mode="rgb_array",
    )
    env = TimeLimit(env, config["max_episode_steps"])
    if plot_episode:
        env = PlotEpisode(
            env,
            save_dir=f"plots/{config['run_name']}",
            episode_trigger=lambda x: x % 5 == 0,  # Once per (5x) evaluation
            log_to_wandb=True,
        )
    if log_task_statistics:
        env = LogTaskStatistics(env)
    if config["normalize_observation"] and not config["running_obs_norm"]:
        env = RescaleObservation(env, 0, 1)
    if config["rescale_action"]:
        env = RescaleAction(env, 0, 1)
    env = FilterObservation(env, ["beam"])
    env = FlattenObservation(env)
    env = Monitor(env)
    if record_video:
        env = RecordVideo(
            env,
            video_folder=f"recordings/{config['run_name']}",
            episode_trigger=lambda x: x % 5 == 0,  # Once per (5x) evaluation
        )
    env = EAMpcEpisodeWithPlotting(env)
    return env


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/config_ea_direct.yaml")
    # To-do: Add an argument for saving the results
    args = parser.parse_args()
    main(args)
