import time
from typing import Callable, Dict, List, Optional, Union

import gymnasium as gym
import numpy as np
import pandas as pd
import torch
import xopt
from botorch.acquisition.proximal import ProximalAcquisitionFunction
from xopt.generators.bayesian.bayesian_generator import BayesianGenerator
from xopt.generators.bayesian.expected_improvement import ExpectedImprovementGenerator
from xopt.generators.bayesian.upper_confidence_bound import (
    TDUpperConfidenceBoundGenerator,
    UpperConfidenceBoundGenerator,
)
from xopt.generators.es.extremumseeking import ExtremumSeekingGenerator
from xopt.generators.rcds.rcds import RCDSGenerator
from xopt.vocs import VOCS


class BOXoptCompatibleWrapper(gym.Wrapper):
    """
    Wrapper to wrap the env as a black-box function for running optimization algorithms.
    """

    def __init__(self, env: gym.Env, prepare_data: Callable):
        super().__init__(env)
        try:
            self.env.action_mode = "direct"
        except AttributeError:
            raise AttributeError("The environment must have an action_mode attribute.")
        self.prepare_data = prepare_data

    def reset(self, seed=None, options=None):
        observation, info = super().reset(seed=seed, options=options)
        data_for_optimizer = self.prepare_data(self.env, observation)
        return data_for_optimizer, info

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        done = terminated or truncated
        data_for_optimizer = self.prepare_data(self.env, obs)
        return data_for_optimizer, reward, done, truncated, info

    def close(self):
        super().close()


class XoptAgent:
    """
    Provide an interface to Xopt (using Bayesian Optimisation or other methods) similar
    to a Stable Baselines3 RL agents.
    """

    def __init__(
        self,
        env: gym.Env,
        vocs: VOCS,
        method: str = "UCB",
        action_order: Optional[list] = None,
        init_samples: Optional[int] = 3,
        init_max_travel_distances: float = 0.05,
        **kwargs,
    ) -> None:
        self.env = env
        self.vocs = vocs
        self.action_order = (
            action_order if action_order is not None else self.vocs.variable_names
        )
        self.last_sample = None
        self.init_samples = init_samples
        self.init_max_travel_distances = init_max_travel_distances
        self.method = method
        if method == "UCB":
            self.generator = ProximalUCBGenerator(vocs=self.vocs, **kwargs)
        elif method == "EI":
            self.generator = ProximalEIGenerator(vocs=self.vocs, **kwargs)
        elif method == "TDUCB":
            self.generator = ProximalTDUCBGenerator(vocs=self.vocs, **kwargs)
        elif method == "ES":
            self.generator = ExtremumSeekingGenerator(vocs=self.vocs, **kwargs)
        elif method == "RCDS":
            self.generator = RCDSGenerator(vocs=self.vocs, **kwargs)
        else:
            raise NotImplementedError(f"Method {method} not implemented.")

        # Create Xopt object
        self.xopt = xopt.Xopt(
            generator=self.generator,
            evaluator=xopt.Evaluator(function=lambda: None),
            vocs=self.vocs,
        )

    def add_data(
        self,
        input_data: Union[pd.DataFrame, Dict[str, float]],
        output_data: Union[pd.DataFrame, Dict[str, float]],
    ):
        # Prepare the input_data for appending in the dataframe
        if not isinstance(input_data, pd.DataFrame):
            try:
                input_data = pd.DataFrame(input_data)
            except ValueError:
                input_data = pd.DataFrame(input_data, index=[0])
        output_data = pd.DataFrame(output_data, index=input_data.index)
        new_data = pd.concat([input_data, output_data], axis=1)
        self.xopt.add_data(new_data)

    def predict(self, n_samples: int = 1):
        assert n_samples == 1, "Only one sample at a time is supported now."
        if len(self.xopt.data) < self.init_samples and self.method in ["UCB", "EI"]:
            # For initialization, sample at smaller stepsizes around the intial sample

            normal_max_travel_distances = (
                self.xopt.generator.max_travel_distances
                if hasattr(self.xopt.generator, "max_travel_distances")
                else 1.0
            )
            # get length of action space
            n_actions = len(self.xopt.vocs.variables)
            self.xopt.generator.max_travel_distances = list(
                np.ones(n_actions) * self.init_max_travel_distances
            )
            new_samples = self.xopt.generator.generate(n_samples)
            self.xopt.generator.max_travel_distances = (
                normal_max_travel_distances  # reset to normal
            )
        else:
            new_samples = self.xopt.generator.generate(n_samples)
        self.last_sample = new_samples
        action = [float(new_samples[0][input]) for input in self.action_order]
        # action = new_samples[self.action_order].values[0]
        return action


def prepare_ARES_EA_data(env, observation: list) -> Dict:
    """
    Transform gym.Env observation to Xopt output format.
    """
    # get unwarpped observation
    # unwrapped_observation = env.unwrapped.observation(observation)
    unwrapped_observation = observation
    current_beam = unwrapped_observation["beam"]
    target_beam = unwrapped_observation["target"]
    # calculate the objective function, and other constraints from the observation
    output_data = {
        "mae": np.mean(np.abs(current_beam - target_beam)),
        "logmae": np.log(np.mean(np.abs(current_beam - target_beam))),
        "max_beamparam": np.max(np.abs(current_beam)),
        "time": time.time(),
    }
    return output_data


def rescale_magnet_values(
    magnet_values: np.ndarray,
    env: gym.Env,
    min_action: float = -1.0,
    max_action: float = 1.0,
):
    low = env.unwrapped.action_space.low
    high = env.unwrapped.action_space.high
    action = min_action + (max_action - min_action) * (magnet_values - low) / (
        high - low
    )
    return action


# Xopt generators
####################


class ProximalGenerator(BayesianGenerator):
    name = "proximal_wrapper"
    proximal_weights: Union[float, List[float], torch.Tensor, None] = None
    softplus_beta: float = 1.0

    def _get_acquisition(self, model):
        base_acq = super()._get_acquisition(model)
        # Convert to tensor if needed
        if self.proximal_weights is None:
            return base_acq
        elif isinstance(self.proximal_weights, float):
            self.proximal_weights = (
                torch.ones(model.train_inputs[0][0].shape[-1]) * self.proximal_weights
            )
        elif isinstance(self.proximal_weights, list):
            self.proximal_weights = torch.tensor(self.proximal_weights)

        return ProximalAcquisitionFunction(
            acq_function=base_acq,
            proximal_weights=self.proximal_weights,
            beta=self.softplus_beta,
        )


class ProximalUCBGenerator(UpperConfidenceBoundGenerator, ProximalGenerator):
    name = "proximal_ucb"
    __doc__ = """Implements Proximal Biasing Bayeisan Optimization using the Upper
            Confidence Bound acquisition function"""


class ProximalTDUCBGenerator(TDUpperConfidenceBoundGenerator, ProximalGenerator):
    name = "proximal_tducb"
    __doc__ = """Implements Proximal Biasing Bayeisan Optimization using the 
        Temporal Difference Upperconfidence Bound acquisition function"""


class ProximalEIGenerator(ExpectedImprovementGenerator, ProximalGenerator):
    name = "proximal_ei"
    __doc__ = """Implements Proximal Biasing Bayeisan Optimization using the
        Expected Improvement acquisition function"""
