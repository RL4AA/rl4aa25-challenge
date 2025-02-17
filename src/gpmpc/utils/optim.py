import time

import gpytorch
import torch

from src.gpmpc.utils.utils import create_models


def train_models_torch_optim(
    queue,
    train_inputs,
    train_targets,
    parameters,
    constraints_hyperparams,
    lr_train,
    num_iter_train,
    clip_grad_value,
    print_train=False,
    step_print_train=25,
    num_cores_train=1,
    verbose=False,
):
    """
    Train the gaussian process models hyper-parameters such that the marginal-log
    likelihood for the predictions of the points in memory is minimized.
    This function is launched in parallel of the main process, which is why a queue
    is used to transfer information back to the main process and why the gaussian
    process models are reconstructed using the points in memory and hyper-parameters
    (they cant be sent directly as argument).
    If an error occurs, returns the parameters sent as init values
    (hyper-parameters obtained by the previous training process)
    Args:
            queue (multiprocessing.queues.Queue): queue object used to transfer
            information to the main process

            train_inputs (torch.Tensor): input data-points of the gaussian process
            models (concat(obs, actions)). Dim=(Np, Ns + Na)

            train_targets (torch.Tensor): targets data-points of the gaussian
            process models (obs_new - obs). Dim=(Np, Ns)

            parameters (list of OrderedDict): contains the hyper-parameters of the
            models used as init values. They are obtained by using
            [model.state_dict() for model in models] where models is a list
            containing gaussian process models of the gpytorch library:
            gpytorch.models.ExactGP

            constraints_hyperparams (dict): Constraints on the hyper-parameters.
                See parameters.md for more information

            lr_train (float): learning rate of the training

            num_iter_train (int): number of iteration for the training optimizer

            clip_grad_value (float): value at which the gradient are clipped,
                so that the training is more stable

            print_train (bool): weither to print the information during training.
                default=False

            step_print_train (int): If print_train is True, only print the
                information every step_print_train iteration
    """

    torch.set_num_threads(num_cores_train)
    start_time = time.time()
    # create models, which is necessary since this function is used in a parallel
    # process that do not share memory with the principal process
    models = create_models(
        train_inputs, train_targets, parameters, constraints_hyperparams
    )
    best_outputscales = [model.covar_module.outputscale.detach() for model in models]
    best_noises = [model.likelihood.noise.detach() for model in models]
    best_lengthscales = [
        model.covar_module.base_kernel.lengthscale.detach() for model in models
    ]
    previous_losses = torch.empty(len(models))

    for model_idx in range(len(models)):
        output = models[model_idx](models[model_idx].train_inputs[0])
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(
            models[model_idx].likelihood, models[model_idx]
        )
        previous_losses[model_idx] = -mll(output, models[model_idx].train_targets)

    best_losses = previous_losses.detach().clone()
    # Random initialization of the parameters showed better performance than
    # just taking the value from the previous iteration as init values.
    # If parameters found at the end do not better performance than previous iter,
    # return previous parameters
    for model_idx in range(len(models)):
        models[model_idx].covar_module.outputscale = models[
            model_idx
        ].covar_module.raw_outputscale_constraint.lower_bound + torch.rand(
            models[model_idx].covar_module.outputscale.shape
        ) * (
            models[model_idx].covar_module.raw_outputscale_constraint.upper_bound
            - models[model_idx].covar_module.raw_outputscale_constraint.lower_bound
        )

        models[model_idx].covar_module.base_kernel.lengthscale = models[
            model_idx
        ].covar_module.base_kernel.raw_lengthscale_constraint.lower_bound + (
            models[
                model_idx
            ].covar_module.base_kernel.raw_lengthscale_constraint.upper_bound
            - models[
                model_idx
            ].covar_module.base_kernel.raw_lengthscale_constraint.lower_bound
        ) * torch.rand(
            models[model_idx].covar_module.base_kernel.lengthscale.shape
        )

        models[model_idx].likelihood.noise = models[
            model_idx
        ].likelihood.noise_covar.raw_noise_constraint.lower_bound + torch.rand(
            models[model_idx].likelihood.noise.shape
        ) * (
            models[model_idx].likelihood.noise_covar.raw_noise_constraint.upper_bound
            - models[model_idx].likelihood.noise_covar.raw_noise_constraint.lower_bound
        )
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(
            models[model_idx].likelihood, models[model_idx]
        )
        models[model_idx].train()
        models[model_idx].likelihood.train()
        optimizer = torch.optim.LBFGS(
            [
                {
                    "params": models[model_idx].parameters()
                },  # Includes GaussianLikelihood parameters
            ],
            lr=lr_train,
            line_search_fn="strong_wolfe",
        )
        try:
            for i in range(num_iter_train):

                def closure():
                    optimizer.zero_grad()
                    # Output from the model
                    output = models[model_idx](models[model_idx].train_inputs[0])
                    # Calculate loss and backpropagation gradients
                    loss = -mll(output, models[model_idx].train_targets)
                    loss.backward()
                    torch.nn.utils.clip_grad_value_(
                        models[model_idx].parameters(), clip_grad_value
                    )
                    return loss

                loss = optimizer.step(closure)
                if print_train and i % step_print_train == 0:
                    lengthscale = (
                        models[model_idx]
                        .covar_module.base_kernel.lengthscale.detach()
                        .numpy()
                    )
                    print(
                        f"Iter {i + 1}/{num_iter_train} - \nLoss: {loss.item():.5f}"
                        "   output_scale: "
                        f"{models[model_idx].covar_module.outputscale.item():.5f}"
                        f"   lengthscale: {str(lengthscale)}"
                        "    noise: "
                        f"{models[model_idx].likelihood.noise.item() ** 0.5:.5f}"
                    )

                if loss < best_losses[model_idx]:
                    best_losses[model_idx] = loss.item()
                    best_lengthscales[model_idx] = models[
                        model_idx
                    ].covar_module.base_kernel.lengthscale
                    best_noises[model_idx] = models[model_idx].likelihood.noise
                    best_outputscales[model_idx] = models[
                        model_idx
                    ].covar_module.outputscale

        except Exception as e:
            print(e)

        if verbose:
            print(
                "training process - model %d - time train %f - output_scale: %s "
                "- lengthscales: %s - noise: %s"
                % (
                    model_idx,
                    time.time() - start_time,
                    str(best_outputscales[model_idx].detach().numpy()),
                    str(best_lengthscales[model_idx].detach().numpy()),
                    str(best_noises[model_idx].detach().numpy()),
                )
            )

    if verbose:
        print(
            "training process - previous marginal log likelihood: %s "
            "- new marginal log likelihood: %s"
            % (
                str(previous_losses.detach().numpy()),
                str(best_losses.detach().numpy()),
            )
        )

    params_dict_list = []
    for model_idx in range(len(models)):
        params_dict_list.append(
            {
                "covar_module.base_kernel.lengthscale": best_lengthscales[model_idx]
                .detach()
                .numpy(),
                "covar_module.outputscale": best_outputscales[model_idx]
                .detach()
                .numpy(),
                "likelihood.noise": best_noises[model_idx].detach().numpy(),
            }
        )
    queue.put(params_dict_list)
