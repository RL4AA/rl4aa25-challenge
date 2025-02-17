import gpytorch
import torch


class ExactGPModelMonoTask(gpytorch.models.ExactGP):
    """
    A single-task Exact Gaussian Process Model using GPyTorch.

    Parameters:
    - train_x (Tensor): Training input data.
    - train_y (Tensor): Training target data.
    - dim_input (int): Dimensionality of the input features.
    - likelihood (gpytorch.likelihoods.Likelihood, optional):
        Gaussian likelihood function. Defaults to GaussianLikelihood.
    - kernel (gpytorch.kernels.Kernel, optional):
        Covariance kernel. Defaults to RBFKernel with ARD.
    """

    def __init__(
        self,
        train_x: torch.Tensor,
        train_y: torch.Tensor,
        likelihood: gpytorch.likelihoods.Likelihood = None,
        kernel: gpytorch.kernels.Kernel = None,
        dim_input: int = None,
    ):
        likelihood = (
            likelihood
            if likelihood is not None
            else gpytorch.likelihoods.GaussianLikelihood()
        )
        super().__init__(train_x, train_y, likelihood)

        dim_input = train_x.shape[-1] if train_x is not None else dim_input

        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = (
            kernel
            if kernel is not None
            else gpytorch.kernels.ScaleKernel(
                gpytorch.kernels.RBFKernel(ard_num_dims=dim_input)
            )
        )

    def forward(self, x):
        """
        Performs a forward pass through the model.

        Parameters:
        - x (Tensor): Input data for prediction.

        Returns:
        - MultivariateNormal: Predictive distribution.
        """
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class BatchIndependentMultitaskGPModel(gpytorch.models.ExactGP):
    def __init__(
        self,
        train_x: torch.Tensor,
        train_y: torch.Tensor,
        likelihood: gpytorch.likelihoods.Likelihood = None,
    ):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(ard_num_dims=train_x.shape[-1])
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultitaskMultivariateNormal.from_batch_mvn(
            gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
        )
