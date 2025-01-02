from abc import ABC, abstractmethod
from typing import Optional, Tuple

import torch


class BaseCostFunction(ABC):

    @abstractmethod
    def __call__(
        self,
        state_mu: torch.Tensor,
        state_var: torch.Tensor,
        action=None,
        terminal=False,
        **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        pass


class QuadraticCostFunction(BaseCostFunction):

    differentiable = True

    def __init__(
        self,
        target_state: torch.Tensor,
        target_action: torch.Tensor,
        weight_state_matrix: Optional[torch.Tensor] = None,
        weight_action_matrix: Optional[torch.Tensor] = None,
    ):
        self.target_state = target_state
        self.target_action = target_action
        self.target_state_action = torch.cat([target_state, target_action])
        if weight_state_matrix is None:
            self.weight_state_matrix = torch.eye(target_state.shape[0])
        else:
            self.weight_state_matrix = weight_state_matrix
        if weight_action_matrix is None:
            self.weight_action_matrix = torch.eye(target_action.shape[0])
        else:
            self.weight_action_matrix = weight_action_matrix
        self.weight_state_action_matrix = torch.block_diag(
            self.weight_state_matrix, self.weight_action_matrix
        )

    def __call__(
        self,
        state_mu: torch.Tensor,
        state_var: torch.Tensor,
        action=None,
        terminal=False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if terminal:
            return self.compute_cost_state(state_mu, state_var)
        return self.compute_cost_state_action(state_mu, state_var, action)

    def compute_cost_state_action(
        self,
        state_mu: torch.Tensor,
        state_var: torch.Tensor,
        action=None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute the quadratic cost of one state distribution or a trajectory of states
        distributions given the mean value and variance of states (observations), the
        weight matrix, and target state.
        The state, state_var and action must be normalized.
        If reading directly from the gym env observation,
        this can be done with the gym env action space and observation space.
        See an example of normalization in the add_points_memory function.
        Args:
            state_mu (torch.Tensor): normalized mean value of the state or observation
                distribution (elements between 0 and 1). dim=(Ns) or dim=(Np, Ns)
            state_var (torch.Tensor): normalized variance matrix of the state or
            observation distribution (elements between 0 and 1)
                dim=(Ns, Ns) or dim=(Np, Ns, Ns)
            action (torch.Tensor): normed actions. (elements between 0 and 1).
                dim=(Na) or dim=(Np, Na)

            Np: length of the prediction trajectory. (=self.len_horizon)
            Na: dimension of the gym environment actions
            Ns: dimension of the gym environment states

        Returns:
            cost_mu (torch.Tensor): mean value of the cost distribution.
                dim=(1) or dim=(Np)
            cost_var (torch.Tensor): variance of the cost distribution.
                dim=(1) or dim=(Np)
        """

        if state_var.ndim == 3:
            error = torch.cat((state_mu, action), 1) - self.target_state_action
            state_action_var = torch.cat(
                (
                    torch.cat(
                        (
                            state_var,
                            torch.zeros(
                                (
                                    state_var.shape[0],
                                    state_var.shape[1],
                                    action.shape[1],
                                )
                            ),
                        ),
                        2,
                    ),
                    torch.zeros(
                        (
                            state_var.shape[0],
                            action.shape[1],
                            state_var.shape[1] + action.shape[1],
                        )
                    ),
                ),
                1,
            )
        else:
            error = torch.cat((state_mu, action), 0) - self.target_state_action
            state_action_var = torch.block_diag(
                state_var, torch.zeros((action.shape[0], action.shape[0]))
            )
        cost_mu = (
            torch.diagonal(
                torch.matmul(state_action_var, self.weight_state_action_matrix),
                dim1=-1,
                dim2=-2,
            ).sum(-1)
            + torch.matmul(
                torch.matmul(
                    error[..., None].transpose(-1, -2), self.weight_state_action_matrix
                ),
                error[..., None],
            ).squeeze()
        )
        TS = self.weight_state_action_matrix @ state_action_var
        cost_var_term1 = torch.diagonal(2 * TS @ TS, dim1=-1, dim2=-2).sum(-1)
        cost_var_term2 = TS @ self.weight_state_action_matrix
        cost_var_term3 = (
            4 * error[..., None].transpose(-1, -2) @ cost_var_term2 @ error[..., None]
        ).squeeze()
        cost_var = cost_var_term1 + cost_var_term3
        return cost_mu, cost_var

    def compute_cost_state(
        self,
        state_mu: torch.Tensor,
        state_var: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        error = state_mu - self.target_state
        cost_mu = torch.trace(
            torch.matmul(state_var, self.weight_state_matrix)
        ) + torch.matmul(torch.matmul(error.t(), self.weight_state_matrix), error)
        TS = self.weight_state_matrix @ state_var
        cost_var_term1 = torch.trace(2 * TS @ TS)
        cost_var_term2 = 4 * error.t() @ TS @ self.weight_state_matrix @ error
        cost_var = cost_var_term1 + cost_var_term2
        return cost_mu, cost_var

    def set_target_state(self, target_state: torch.Tensor):
        self.target_state = target_state
        self.target_state_action = torch.cat([target_state, self.target_action])
