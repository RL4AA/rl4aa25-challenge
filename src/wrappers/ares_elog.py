import base64
import subprocess
from datetime import datetime
from io import BytesIO
from typing import Callable, Optional

import gymnasium as gym
import numpy as np
from gymnasium import logger

from ..eval.episode import Episode

# TODO Make record episode, plot episode and area ea elog wrapper based on one wrapper
#     base class.
# TODO Check that message formatting is correct for all sections. Plot is be correct.


def send_to_elog(
    author: str,
    title: str,
    severity: str,
    text: str,
    elog: str,
    image: Optional[bytes] = None,
) -> bool:
    """
    Send information to a supplied electronic logbook.

    :param author: The author of the log entry.
    :param title: The title of the log entry.
    :param severity: The severity of the log entry.
    :param text: The text of the log entry.
    :param elog: The name of the electronic logbook to send the log entry to.
    :return: True if the log entry was successfully sent, False otherwise.
    """

    # The DOOCS elog expects an XML string in a particular format. This string
    # is beeing generated in the following as an initial list of strings.
    succeded = True  # indicator for a completely successful job
    # list beginning
    elog_xml_string_list = ['<?xml version="1.0" encoding="ISO-8859-1"?>', "<entry>"]
    # author information
    elog_xml_string_list.append("<author>")
    elog_xml_string_list.append(author)
    elog_xml_string_list.append("</author>")
    # title information
    elog_xml_string_list.append("<title>")
    elog_xml_string_list.append(title)
    elog_xml_string_list.append("</title>")
    # severity information
    elog_xml_string_list.append("<severity>")
    elog_xml_string_list.append(severity)
    elog_xml_string_list.append("</severity>")
    # text information
    elog_xml_string_list.append("<text>")
    elog_xml_string_list.append(text)
    elog_xml_string_list.append("</text>")
    # image information
    if image:
        try:
            encodedImage = base64.b64encode(image)
            elog_xml_string_list.append("<image>")
            elog_xml_string_list.append(encodedImage.decode())
            elog_xml_string_list.append("</image>")
        except (
            Exception
        ) as e:  # make elog entry anyway, but return error (succeded = False)
            succeded = False
            print(f"When appending image, encounterd exception {e}")
    # list end
    elog_xml_string_list.append("</entry>")
    # join list to the final string
    elog_xml_string = "\n".join(elog_xml_string_list)
    # open printer process
    try:
        lpr = subprocess.Popen(
            ["/usr/bin/lp", "-o", "raw", "-d", elog],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
        )
        # send printer job
        lpr.communicate(elog_xml_string.encode("utf-8"))
    except Exception as e:
        print(f"When sending log entry to printer process, encounterd exception {e}")
        succeded = False
    return succeded


class ARESeLog(gym.Wrapper):
    """
    Wrapper to send a summary of transverse beam parameter optimsations at the ARES
    particle accelerator to the ARES eLog.

    NOTE: This wrapper is only compatible with ARES transverse beam parameter tuning
    environments.

    NOTE: A plot will only be saved if at least two steps have been taken in the
    episode.

    :param env: The environment that will be wrapped.
    :param agent_name: The name of the agent that is being used to optimise the beam.
    """

    def __init__(
        self,
        env: gym.Env,
        episode_trigger: Callable[[int], bool] = lambda _: True,
        agent_name: str = "unknown",
    ):
        super().__init__(env)

        self.agent_name = agent_name
        self.episode_id = 0
        self.is_recording = False
        self.episode_trigger = episode_trigger

    def reset(self, seed=None, options=None):
        observation, info = self.env.reset(seed=seed, options=options)

        if self.episode_trigger(self.episode_id):
            self.is_recording = True

            self.observations = [observation]
            self.rewards = []
            self.terminateds = []
            self.truncateds = []
            self.infos = []
            self.actions = []
            self.t_start = datetime.now()
            self.t_end = None
            self.steps_taken = 0
            self.step_start_times = []
            self.step_end_times = []

        return observation, info

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)

        if self.is_recording:
            self.observations.append(observation)
            self.rewards.append(reward)
            self.terminateds.append(terminated)
            self.truncateds.append(truncated)
            self.infos.append(info)
            self.actions.append(action)
            self.steps_taken += 1
            self.step_end_times.append(datetime.now())

            if terminated or truncated:
                self.t_end = datetime.now()
                self.send_episode_to_elog()
                self.is_recording = False

        if terminated or truncated:
            self.episode_id += 1

        return observation, reward, terminated, truncated, info

    def close(self):
        super().close()

        if self.is_recording:
            self.t_end = datetime.now()
            self.send_episode_to_elog()

    def send_episode_to_elog(self):
        """Send a summary report of the episode to the ARES eLog."""
        if len(self.observations) < 2:  # No data to plot
            logger.warn(
                f"Unable to save episode plot for {self.episode_id = } because the"
                " episode was too short."
            )
            return

        episode = Episode(
            observations=self.observations,
            rewards=self.rewards,
            terminateds=self.terminateds,
            truncateds=self.truncateds,
            infos=self.infos,
            actions=self.actions,
            t_start=self.t_start,
            t_end=self.t_end,
        )

        screen_name = episode.infos[0]["screen_name"]

        msg = self.create_text_message(episode)
        img = self.create_plot_jpg(episode)

        title = f"Beam Optimisation on {screen_name} using " + (
            "Bayesian Optimisation"
            if self.agent_name == "Bayesian Optimisation"
            else "Reinforcement Learning"
        )

        print(f"{title = }")
        print(f"{msg = }")

        send_to_elog(
            elog="areslog",
            author="Autonomous ARES",
            title=title,
            severity="NONE",
            text=msg,
            image=img,
        )

    def create_text_message(self, episode: Episode):
        """Create text message summarising the optimisation."""
        beam_before = episode.beam_parameters_before()
        beam_after = episode.beam_parameters_after()
        target_beam = episode.target
        final_deltas = beam_after - target_beam
        final_mae = episode.final_mae()
        target_threshold = (
            np.full(4, self.env.unwrapped.target_threshold)
            if isinstance(self.env.unwrapped.target_threshold, float)
            else self.env.unwrapped.target_threshold
        )
        if target_threshold is None:
            target_threshold = np.full(4, 4e-5)
        final_magnets = episode.magnet_history()[-1]
        steps_taken = len(episode)
        success = np.abs(beam_after - target_beam) < target_threshold
        agent_name = self.agent_name
        t_start = episode.t_start
        t_end = episode.t_end
        screen_name = episode.infos[0]["screen_name"]
        magnet_names = episode.infos[0]["magnet_names"]
        magnet_property_names = [
            "strength" if name[5] == "Q" else "kick" for name in magnet_names
        ]
        magnet_units = ["1/m^2" if name[5] == "Q" else "mrad" for name in magnet_names]

        algorithm = (
            "Bayesian Optimisation"
            if self.agent_name == "Bayesian Optimisation"
            else "Reinforcement Learning agent"
        )

        message = (
            f"{algorithm} optimised beam on {screen_name}\n"
            "\n"
            f"Agent: {agent_name}\n"
            f"Start time: {t_start}\n"
            f"Time taken: {t_end - t_start}\n"
            f"No. of steps: {steps_taken}\n"
            "\n"
            "Beam before:\n"
            f"    mu_x    = {beam_before[0] * 1e3: 5.4f} mm\n"
            f"    sigma_x = {beam_before[1] * 1e3: 5.4f} mm\n"
            f"    mu_y    = {beam_before[2] * 1e3: 5.4f} mm\n"
            f"    sigma_y = {beam_before[3] * 1e3: 5.4f} mm\n"
            "\n"
            "Beam after:\n"
            f"    mu_x    = {beam_after[0] * 1e3: 5.4f} mm\n"
            f"    sigma_x = {beam_after[1] * 1e3: 5.4f} mm\n"
            f"    mu_y    = {beam_after[2] * 1e3: 5.4f} mm\n"
            f"    sigma_y = {beam_after[3] * 1e3: 5.4f} mm\n"
            "\n"
            "Target beam:\n"
            f"    mu_x    = {target_beam[0] * 1e3: 5.4f} mm    (e = "
            f"{target_threshold[0] * 1e3:5.4f} mm) {';)' if success[0] else ':/'}\n"
            f"    sigma_x = {target_beam[1] * 1e3: 5.4f} mm    (e = "
            f"{target_threshold[1] * 1e3:5.4f} mm) {';)' if success[1] else ':/'}\n"
            f"    mu_y    = {target_beam[2] * 1e3: 5.4f} mm    (e = "
            f"{target_threshold[2] * 1e3:5.4f} mm) {';)' if success[2] else ':/'}\n"
            f"    sigma_y = {target_beam[3] * 1e3: 5.4f} mm    (e = "
            f"{target_threshold[3] * 1e3:5.4f} mm) {';)' if success[3] else ':/'}\n"
            "\n"
            "Result:\n"
            f"    |delta_mu_x|    = {abs(final_deltas[0]) * 1e3: 5.4f} mm\n"
            f"    |delta_sigma_x| = {abs(final_deltas[1]) * 1e3: 5.4f} mm\n"
            f"    |delta_mu_y|    = {abs(final_deltas[2]) * 1e3: 5.4f} mm\n"
            f"    |delta_sigma_y| = {abs(final_deltas[3]) * 1e3: 5.4f} mm\n"
            "\n"
            f"    MAE = {final_mae * 1e3: 5.4f} mm\n\nFinal magnet settings:"
        )

        # Append magnet settings dymanically based on the number and type of magnets
        for name, property_name, setting, unit in zip(
            magnet_names, magnet_property_names, final_magnets, magnet_units
        ):
            converted = setting * 1e3 if unit == "mrad" else setting
            message += f"\n    {name} {property_name} = {converted: 8.4f} {unit}"

        return message

    def create_plot_jpg(self, episode: Episode):
        """Create plot overview of the optimisation and return it as jpg bytes."""
        fig = episode.plot_summary()

        buf = BytesIO()
        fig.savefig(buf, dpi=300, format="jpg")
        buf.seek(0)
        img = bytes(buf.read())

        return img
