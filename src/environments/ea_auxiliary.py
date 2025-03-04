from gymnasium import spaces
from gymnasium.wrappers import TimeLimit, FlattenObservation, RescaleAction

from ..reward import combiners, transforms
from ..type_aliases import CombinerLiteral, TransformLiteral
from ..wrappers import RecordEpisode, RescaleObservation
from . import ea


def make_eval_env(config: dict, ea):
    wrapper_config = config["env_wrapper"]
    env = ea.TransverseTuning(**config["env"])

    env = TimeLimit(env, max_episode_steps=wrapper_config["max_episode_steps"])
    env = RecordEpisode(env)
    if wrapper_config["normalize_observation"]:
        env = RescaleObservation(env, 0, 1)
    if wrapper_config["rescale_action"]:
        env = RescaleAction(env, -1, 1)
    env = FlattenObservation(env)
    return env


class TransverseTuning(ea.TransverseTuning):
    """
    Variant of the ARES EA transverse tuning environment with auxiliary taks added. The
    auxiliary tasks are:
     - Predicting the incoming beam parameters
     - Predicting the misalignments of the quadrupoles and the screen

    Magnets: AREAMQZM1, AREAMQZM2, AREAMCVM1, AREAMQZM3, AREAMCHM1
    Screen: AREABSCR1

    :param incoming_transform: Reward transform for the incoming beam parameters. Can be
        `"Linear"`, `"ClippedLinear"`, `"SoftPlus"`, `"NegExp"` or `"Sigmoid"`.
    :param incoming_combiner: Reward combiner for the incoming beam parameters. Can be
        `"Mean"`, `"Multiply"`, `"GeometricMean"`, `"Min"`, `"Max"`, `"LNorm"` or
        `"SmoothMax"`.
    :param incoming_combiner_args: Arguments for the incoming beam parameter combiner.
        NOTE that these may be different for different combiners.
    :param incoming_combiner_weights: Weights for the incoming beam parameter combiner.
    :param misalignment_transform: Reward transform for the misalignments. Can be
        `"Linear"`, `"ClippedLinear"`, `"SoftPlus"`, `"NegExp"` or `"Sigmoid"`.
    :param misalignment_combiner: Reward combiner for the misalignments. Can be
        `"Mean"`, `"Multiply"`, `"GeometricMean"`, `"Min"`, `"Max"`, `"LNorm"` or
        `"SmoothMax"`.
    :param misalignment_combiner_args: Arguments for the misalignment combiner. NOTE
        that these may be different for different combiners.
    :param misalignment_combiner_weights: Weights for the misalignment combiner.
    :param aux_combiner: Reward combiner for the auxiliary tasks. Can be `"Mean"`,
        `"Multiply"`, `"GeometricMean"`, `"Min"`, `"Max"`, `"LNorm"` or `"SmoothMax"`.
    :param aux_combiner_args: Arguments for the auxiliary combiner combining the
        incoming beam reward and misalignment reward. NOTE that these may be different
        for different combiners.
    :param aux_combiner_weights: Weights for the auxiliary combiner.
    :param combined_combiner: Reward combiner for the combined reward of taks and
        auxiliary reward. Can be `"Mean"`, `"Multiply"`, `"GeometricMean"`, `"Min"`,
        `"Max"`, `"LNorm"` or `"SmoothMax"`.
    :param combined_combiner_args: Arguments for the combined combiner combining the
        task reward and auxiliary reward. NOTE that these may be different for different
        combiners.
    :param combined_combiner_weights: Weights for the combined combiner.
    """

    def __init__(
        self,
        *args,
        incoming_transform: TransformLiteral = "Sigmoid",
        incoming_combiner: CombinerLiteral = "GeometricMean",
        incoming_combiner_args: dict = {},
        incoming_combiner_weights: list = [1] * 11,
        misalignment_transform: TransformLiteral = "Sigmoid",
        misalignment_combiner: CombinerLiteral = "GeometricMean",
        misalignment_combiner_args: dict = {},
        misalignment_combiner_weights: list = [1] * 8,
        aux_combiner: CombinerLiteral = "SmoothMax",
        aux_combiner_args: dict = {"alpha": -5},
        aux_combiner_weights: list = [1, 1],
        combined_combiner: CombinerLiteral = "SmoothMax",
        combined_combiner_args: dict = {"alpha": -5},
        combined_combiner_weights: list = [2, 1],
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)

        self.action_space = spaces.Dict(
            {
                "action": self.action_space,
                "incoming": self.backend.incoming_beam_space,
                "misalignments": self.backend.misalignment_space,
            }
        )

        # Setup auxiliary reward computation
        incoming_transform_cls = getattr(transforms, incoming_transform)
        incoming_combiner_cls = getattr(combiners, incoming_combiner)
        misalignment_transform_cls = getattr(transforms, misalignment_transform)
        misalignment_combiner_cls = getattr(combiners, misalignment_combiner)
        aux_combiner_cls = getattr(combiners, aux_combiner)
        combined_combiner_cls = getattr(combiners, combined_combiner)

        self._abs_transform = transforms.Abs()
        self._incoming_energy_transform = incoming_transform_cls(
            good=0.0,
            bad=self.backend.incoming_beam_space.high[0]
            - self.backend.incoming_beam_space.low[0],
        )
        self._incoming_mu_transform = incoming_transform_cls(
            good=0.0,
            bad=self.backend.incoming_beam_space.high[1]
            - self.backend.incoming_beam_space.low[1],
        )
        self._incoming_mup_transform = incoming_transform_cls(
            good=0.0,
            bad=self.backend.incoming_beam_space.high[2]
            - self.backend.incoming_beam_space.low[2],
        )
        self._incoming_sigma_transform = incoming_transform_cls(
            good=0.0,
            bad=self.backend.incoming_beam_space.high[5]
            - self.backend.incoming_beam_space.low[5],
        )
        self._incoming_sigmap_transform = incoming_transform_cls(
            good=0.0,
            bad=self.backend.incoming_beam_space.high[6]
            - self.backend.incoming_beam_space.low[6],
        )
        self._incoming_sigma_s_transform = incoming_transform_cls(
            good=0.0,
            bad=self.backend.incoming_beam_space.high[9]
            - self.backend.incoming_beam_space.low[9],
        )
        self._incoming_sigma_p_transform = incoming_transform_cls(
            good=0.0,
            bad=self.backend.incoming_beam_space.high[10]
            - self.backend.incoming_beam_space.low[10],
        )
        self._incoming_combiner = incoming_combiner_cls(**incoming_combiner_args)

        self._misalignment_transform = misalignment_transform_cls(
            good=0.0, bad=self.backend.max_misalignment * 2
        )
        self._misalignment_combiner = misalignment_combiner_cls(
            **misalignment_combiner_args
        )

        self._aux_combiner = aux_combiner_cls(**aux_combiner_args)
        self._combined_combiner = combined_combiner_cls(**combined_combiner_args)

        self._incoming_combiner_weights = incoming_combiner_weights
        self._misalignment_combiner_weights = misalignment_combiner_weights
        self._aux_combiner_weights = aux_combiner_weights
        self._combined_combiner_weights = combined_combiner_weights

    def _take_action(self, action: dict) -> None:
        """Take `action` according to the environment's configuration."""
        super()._take_action(action["action"])

        # Save extimates from auxiliary task for reward computation
        self._incoming_estimate = action["incoming"]
        self._misalignment_estimate = action["misalignments"]

    def _get_reward(self) -> float:
        """Compute reward as combination of task reward and auxiliary reward."""
        incoming_truth = self.backend.get_incoming_parameters()
        misalignments_truth = self.backend.get_misalignments()
        incoming_estimate = self._incoming_estimate
        misalignment_estimate = self._misalignment_estimate

        task_reward = super()._get_reward()

        incoming_energy_reward = self._incoming_energy_transform(
            self._abs_transform([incoming_estimate[0] - incoming_truth[0]])
        )
        incoming_mu_x_reward = self._incoming_mu_transform(
            self._abs_transform([incoming_estimate[1] - incoming_truth[1]])
        )
        incoming_mu_xp_reward = self._incoming_mup_transform(
            self._abs_transform([incoming_estimate[2] - incoming_truth[2]])
        )
        incoming_mu_y_reward = self._incoming_mu_transform(
            self._abs_transform([incoming_estimate[3] - incoming_truth[3]])
        )
        incoming_mu_yp_reward = self._incoming_mup_transform(
            self._abs_transform([incoming_estimate[4] - incoming_truth[4]])
        )
        incoming_sigma_x_reward = self._incoming_sigma_transform(
            self._abs_transform([incoming_estimate[5] - incoming_truth[5]])
        )
        incoming_sigma_xp_reward = self._incoming_sigmap_transform(
            self._abs_transform([incoming_estimate[6] - incoming_truth[6]])
        )
        incoming_sigma_y_reward = self._incoming_sigma_transform(
            self._abs_transform([incoming_estimate[7] - incoming_truth[7]])
        )
        incoming_sigma_yp_reward = self._incoming_sigmap_transform(
            self._abs_transform([incoming_estimate[8] - incoming_truth[8]])
        )
        incoming_sigma_s_reward = self._incoming_sigma_s_transform(
            self._abs_transform([incoming_estimate[9] - incoming_truth[9]])
        )
        incoming_sigma_p_reward = self._incoming_sigma_p_transform(
            self._abs_transform([incoming_estimate[10] - incoming_truth[10]])
        )
        incoming_reward = self._incoming_combiner(
            [
                incoming_energy_reward[0],
                incoming_mu_x_reward[0],
                incoming_mu_xp_reward[0],
                incoming_mu_y_reward[0],
                incoming_mu_yp_reward[0],
                incoming_sigma_x_reward[0],
                incoming_sigma_xp_reward[0],
                incoming_sigma_y_reward[0],
                incoming_sigma_yp_reward[0],
                incoming_sigma_s_reward[0],
                incoming_sigma_p_reward[0],
            ],
            weights=self._incoming_combiner_weights,
        )

        misalignment_reward = self._misalignment_combiner(
            self._misalignment_transform(
                self._abs_transform(misalignment_estimate - misalignments_truth)
            ),
            weights=self._misalignment_combiner_weights,
        )

        aux_reward = self._aux_combiner(
            [incoming_reward, misalignment_reward], weights=self._aux_combiner_weights
        )
        combined_reward = self._combined_combiner(
            [task_reward, aux_reward], weights=self._combined_combiner_weights
        )

        return combined_reward
