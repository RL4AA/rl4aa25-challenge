import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D


class EpisodeData:
    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.maes = []
        self.is_done = False

    def add_step(self, state, action, reward, mae):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.maes.append(mae)

    def end_episode(self):
        self.is_done = True


class EAMpcEpisodeWithPlotting(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.episodes = []
        self.current_episode = None
        self._setup_plotting()

    def _setup_plotting(self):
        self.fig, (self.ax1, self.ax2, self.ax3) = plt.subplots(
            3, 1, figsize=(6, 8), tight_layout=True
        )
        plt.ion()  # Interactive mode for live updates
        self.cumulative_step = 0
        self.n_states = self.env.observation_space.shape[0]
        # Here a hard hack to remove the target beam from the plot
        if self.n_states == 13:
            self.n_states = 9
        self.n_actions = self.env.action_space.shape[0]
        self.colors_states = plt.get_cmap("rainbow")(np.linspace(0, 1, self.n_states))
        self.colors_actions = plt.get_cmap("rainbow")(np.linspace(0, 1, self.n_actions))
        plt.show(block=False)

    def _update_plots(self):
        self.ax1.clear()
        self.ax2.clear()
        self.ax3.clear()

        cumulative_step = 0

        # Plot data for each completed episode
        for episode in self.episodes:
            cumulative_step = self._plot_episode(episode, cumulative_step)

        # Plot data for the current (incomplete) episode
        cumulative_step = self._plot_episode(self.current_episode, cumulative_step)

        self.ax1.set_title("Trajectories for Each Episode")
        self.ax1.set_xlabel("Cumulative Step")
        self.ax1.set_ylabel("State Value")
        self.ax1.grid()

        self.ax2.set_title("Actions for Each Episode")
        self.ax2.set_xlabel("Cumulative Step")
        self.ax2.set_ylabel("Action Value")
        self.ax2.grid()

        self.ax3.set_title("MAEs for Each Episode")
        self.ax3.set_xlabel("Cumulative Step")
        self.ax3.set_ylabel("MAE (mm)")
        self.ax3.set_yscale("log")
        self.ax3.grid()

        # Update legends
        legend_handles_states = [
            Line2D([], [], color=self.colors_states[i], label=f"State {i + 1}")
            for i in range(self.n_states)
        ]

        action_names = self.env.get_wrapper_attr("action_names")

        legend_handles_actions = [
            Line2D([], [], color=self.colors_actions[i], label=action_names[i])
            for i in range(self.n_actions)
        ]

        # self.ax1.legend(handles=legend_handles_states)
        # self.ax2.legend(handles=legend_handles_actions)
        self.ax1.legend(
            handles=legend_handles_states, loc="upper left", bbox_to_anchor=(1, 1)
        )
        self.ax2.legend(
            handles=legend_handles_actions, loc="upper left", bbox_to_anchor=(1, 1)
        )

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    # Function to plot data for an episode
    def _plot_episode(self, episode, start_step):
        if not episode:
            return start_step

        trajectory = (
            np.array(episode.states) if episode.states else np.zeros((0, self.n_states))
        )
        steps = range(start_step, start_step + len(trajectory))

        # Plot the trajectory
        for i in range(self.n_states):
            self.ax1.plot(steps, trajectory[:, i], color=self.colors_states[i])

        # Plot the actions
        for i in range(self.n_actions):
            action_values = [
                action[i] if action is not None and i < len(action) else np.nan
                for action in episode.actions
            ]
            self.ax2.plot(
                steps,
                action_values,
                color=self.colors_actions[i],
                ls="--",
                marker=".",
            )

        # Plot the MAEs in mm
        self.ax3.plot(steps, np.array(episode.maes) * 1e3)

        return start_step + len(trajectory)

    def step(self, action):
        observation, reward, done, _, info = self.env.step(action)

        if self.current_episode is None:
            self.current_episode = EpisodeData()

        mae = self._calculate_mae()
        self.current_episode.add_step(observation, action, reward, mae)

        if done:
            self.current_episode.end_episode()
            self.episodes.append(self.current_episode)
            self.current_episode = None

        self._update_plots()
        return observation, reward, done, False, info

    def reset(self, **kwargs):
        observation, info = self.env.reset(**kwargs)

        if self.current_episode is not None and not self.current_episode.is_done:
            self.current_episode.end_episode()
            self.episodes.append(self.current_episode)

        self.current_episode = EpisodeData()
        self.current_episode.add_step(
            observation, None, None, self._calculate_mae()
        )  # Initial state with no action or reward

        self._update_plots()
        return observation, info

    def _calculate_mae(self):
        """Return the MAE value"""
        target_beam = self.env.unwrapped._target_beam
        current_beam = self.env.unwrapped.backend.get_beam_parameters()

        return np.mean(np.abs(current_beam - target_beam))
