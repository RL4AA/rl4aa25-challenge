import numpy as np
import pandas as pd
from gymnasium.wrappers import RescaleAction, TimeLimit
from xopt import VOCS

from .ea_optimize import (
    ARESeLog,
    BaseCallback,
    OptimizeFunctionCallback,
    TQDMWrapper,
    setup_callback,
)
from .environments import ea
from .wrappers import RecordEpisode
from .xopt_helper import (
    BOXoptCompatibleWrapper,
    XoptAgent,
    prepare_ARES_EA_data,
    rescale_magnet_values,
)


def optimize(
    target_mu_x,
    target_sigma_x,
    target_mu_y,
    target_sigma_y,
    target_mu_x_threshold=4e-5,
    target_mu_y_threshold=4e-5,
    target_sigma_x_threshold=4e-5,
    target_sigma_y_threshold=4e-5,
    max_steps=100,
    model_name="Xopt",
    method="EI",  # ["EI", "UCB", "ES", "RCDS"]
    optimizer_kwargs={
        "beta": 2.0,
        "proximal_weights": 0.2,
        "max_travel_distances": [0.1] * 5,
        "init_samples": 5,
    },
    logbook=False,
    data_log_dir=None,
    progress_bar=False,
    callback=BaseCallback(),
    rescale_action=(-1, 1),
    magnet_init_values=np.array([10, -10, 0, 10, 0]),
    set_to_best=True,  # set back to best found setting after opt.
    backend="cheetah",
):
    callback = setup_callback(callback)

    # Create the environment
    env = ea.TransverseTuning(
        backend=backend,
        backend_args={"generate_screen_images": True} if backend == "cheetah" else {},
        action_mode="direct",
        magnet_init_mode=magnet_init_values,
        target_beam_mode=np.array(
            [target_mu_x, target_sigma_x, target_mu_y, target_sigma_y]
        ),
        target_threshold=np.array(
            [
                target_mu_x_threshold,
                target_sigma_x_threshold,
                target_mu_y_threshold,
                target_sigma_y_threshold,
            ]
        ),
    )
    if max_steps is not None:
        env = TimeLimit(env, max_steps)
    if progress_bar:
        env = TQDMWrapper(env)
    if callback is not None:
        env = OptimizeFunctionCallback(env, callback)
    if data_log_dir is not None:
        env = RecordEpisode(env, save_dir=data_log_dir)
    if logbook:
        env = ARESeLog(env, agent_name=model_name)
    if rescale_action is not None:
        env = RescaleAction(env, rescale_action[0], rescale_action[1])

    env = BOXoptCompatibleWrapper(env, prepare_ARES_EA_data)

    if rescale_action is not None:
        vocs_variables = {
            "q1": [-1, 1],
            "q2": [-1, 1],
            "cv": [-1, 1],
            "q3": [-1, 1],
            "ch": [-1, 1],
        }
    else:
        vocs_variables = {
            "q1": [-72, 72],
            "q2": [-72, 72],
            "cv": [-6.1782e-3, 6.1782e-3],
            "q3": [-72, 72],
            "ch": [-6.1782e-3, 6.1782e-3],
        }

    vocs = VOCS(
        variables=vocs_variables,
        # objectives={"mae": "MINIMIZE"},
        objectives={"logmae": "MINIMIZE"},
        # constraints={"max_beamparam": ["LESS_THAN", 2e-3]},
    )

    xopt_agent = XoptAgent(
        env,
        vocs,
        method=method,
        action_order=["q1", "q2", "cv", "q3", "ch"],
        **optimizer_kwargs,
    )

    # Actual optimisation
    output, info = env.reset()
    if rescale_action is not None:
        init_magnet_values = rescale_magnet_values(
            env.unwrapped.backend.get_magnets(),
            env.env,
            min_action=rescale_action[0],
            max_action=rescale_action[1],
        )
    else:
        init_magnet_values = env.unwrapped.backend.get_magnets()
    init_sample = [
        {k: v for k, v in zip(["q1", "q2", "cv", "q3", "ch"], init_magnet_values)}
    ]
    xopt_agent.add_data(pd.DataFrame(init_sample), output)
    done = False
    while not done:
        action = xopt_agent.predict(n_samples=1)
        output, reward, done, truncated, info = env.step(action)
        xopt_agent.add_data(xopt_agent.last_sample, output)

    # Set back to best
    if set_to_best:
        best_sample = xopt_agent.xopt.data.iloc[
            xopt_agent.xopt.data[list(xopt_agent.vocs.objectives.keys())[0]].idxmin()
        ]
        best_action = best_sample.values[:5]
        env.step(best_action)
    env.close()
