import numpy as np
from gymnasium.wrappers import RescaleAction, TimeLimit

from .bayesopt import BayesianOptimizationAgent
from .ea_optimize import (
    ARESeLog,
    BaseCallback,
    OptimizeFunctionCallback,
    TQDMWrapper,
    setup_callback,
)
from .environments import ea
from .wrappers import BOOldCompatibility, RecordEpisode


def optimize(
    target_mu_x,
    target_sigma_x,
    target_mu_y,
    target_sigma_y,
    target_mu_x_threshold=3.3198e-6,
    target_mu_y_threshold=3.3198e-6,
    target_sigma_x_threshold=3.3198e-6,
    target_sigma_y_threshold=3.3198e-6,
    max_steps=100,
    model_name="BO",
    logbook=False,
    data_log_dir=None,
    progress_bar=False,
    callback=BaseCallback(),
    stepsize=0.1,  # comparable to RL env
    acquisition="EI",
    init_samples=5,
    filter_action=None,
    rescale_action=(-3, 3),  # Yes 3 is the value we chose
    magnet_init_values=np.array([10, -10, 0, 10, 0]),
    set_to_best=True,  # set back to best found setting after opt.
    mean_module=None,
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
        threshold_hold=1,
        unidirectional_quads=True,
    )
    if max_steps is not None:
        env = TimeLimit(env, max_steps)
    env = BOOldCompatibility(env)
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

    model = BayesianOptimizationAgent(
        env=env,
        filter_action=filter_action,
        stepsize=stepsize,
        init_samples=init_samples,
        acquisition=acquisition,
        mean_module=mean_module,
    )

    callback.env = env

    # Actual optimisation
    observation, _ = env.reset()
    reward = None
    done = False
    while not done:
        action = model.predict(observation, reward)
        observation, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

    # Set back to best
    if set_to_best:
        action = model.X[model.Y.argmax()].detach().numpy()
        env.step(action)

    env.close()
