"""Various sampling methods."""
import functools
import os
import torch
import numpy as np
import evaluation
import sde_lib
import datasets
import logging
from models import utils as mutils
import wandb
import copy
import matplotlib.pyplot as plt
from torchvision.utils import make_grid, save_image
import torchvision
from tqdm import tqdm
import lpips
import jax


def get_sampling_fn(sde, shape, inverse_scaler, model=None, scaler=None, dataset=None, save_folder=None, return_t_steps=False, device="cuda"):
    """
    Get a sampling function for generating samples from a stochastic differential equation (SDE).

    Args:
        sde (object): Stochastic differential equation object.
        shape (tuple): Shape of the samples to be generated.
        inverse_scaler (callable): Inverse scaling function to transform generated samples.
        model (object, optional): Neural network model for conditional sampling.
        scaler (callable, optional): Scaling function to preprocess input data.
        dataset (object, optional): Dataset object for sampling.
        save_folder (str, optional): Directory path for storing intermediate results.
        return_t_steps (bool, optional): If True, return the integration time steps along with the sampling function.
        device (str, optional): Device for computation ('cuda' or 'cpu').

    Returns:
        tuple or callable: If return_t_steps is True, returns a tuple containing the sampling function and log information.
                          If return_t_steps is False, returns only the sampling function.
    """
    # Extract relevant information from sde
    disc_name = sde.discretizing_method
    solver_name = sde.solver

    # Map solver_name to the corresponding solver function
    solver_mapping = {
        "euler": sde.euler_solver,
        "heun2": sde.heun2_solver,
        "rk45": sde.rk45_solver
    }
    solver = solver_mapping.get(solver_name, None)

    if solver is None:
        raise ValueError(f"Solver '{solver_name}' not supported.")

    
    # Get the disc function and log from helper function
    if disc_name == "uniform":
        disc, log = get_uniform_disc(sde.T_start, sde.T_end, sde.num_steps)
    elif disc_name == "edm":
        disc, log = get_edm_disc(sde)
    elif "bellman" in disc_name:
        disc, log = get_bellman_disc(sde, model, shape, scaler, inverse_scaler, dataset, save_folder, device)

    # Define the sampling function
    def sampling_function(model, z=None, class_labels=None):
        """
        Generate samples using the specified model and SDE solver.

        Args:
            model (object): Neural network model for generating samples.
            z (torch.Tensor, optional): Initial sample. If None, it is generated from the SDE.
            class_labels (torch.Tensor, optional): Conditional class labels for conditional sampling.

        Returns:
            tuple: A tuple containing the generated sample and the number of function evaluations (nfe).
        """
        with torch.no_grad():
            # Initial sample
            if z is None:
                z0 = sde.get_z0(torch.zeros(shape, device=device), train=False).to(device)
                x = z0.clone()
            else:
                x = z

            try:
                # Generate random class labels if model has label_dim
                if model.label_dim:
                    class_labels = torch.eye(model.label_dim, device=device)[torch.randint(model.label_dim, size=[sampling_batch_size], device=device)] 
            except AttributeError:
                pass

            # Obtain t_steps using disc function
            t_steps = disc(model, x)
            # Apply solver to evolve the sample
            x, nfe = solver(x, model, t_steps, class_labels=class_labels)
            # Inverse scaling to get the final sample
            x = inverse_scaler(x)
            return x, nfe


    # Return the sampling function and log, if requested
    if return_t_steps:
        return sampling_function, log
    else:
        return sampling_function

@torch.no_grad()
def get_uniform_disc(T_start, T_end, num_steps):
    t_steps = torch.arange(num_steps + 1) / num_steps * (T_end - T_start) + T_start
    def uniform_disc(model, z):
        return t_steps
    return uniform_disc, (t_steps,)

@torch.no_grad()
def get_edm_disc(sde, num_steps=None):
    T_start = sde.T_start
    T_end = sde.T_end
    if num_steps is None:
        num_steps = sde.num_steps
    step_indices = torch.arange(num_steps)
    if num_steps == 1:
        sigma_steps = step_indices + sde.edm_config.sigma_max
    else:
        sigma_steps = (sde.edm_config.sigma_max ** (1 / sde.edm_config.rho) + step_indices / (num_steps - 1) * (sde.edm_config.sigma_min ** (1 / sde.edm_config.rho) - sde.edm_config.sigma_max ** (1 / sde.edm_config.rho))) ** sde.edm_config.rho

    try:
        t_steps = sde.edm_config.sigma_inv(model.round_sigma(sigma_steps))
    except:
        t_steps = sde.edm_config.sigma_inv(sigma_steps)
    t_steps = torch.cat([t_steps, torch.full_like(t_steps[:1], sde.T_end)])  # t_N = 0

    def edm_disc(model, z):
        return t_steps
    return edm_disc, (t_steps,)

@torch.no_grad()
def get_bellman_disc(sde, model, shape, scaler=None, inverse_scaler=None, dataset=None, save_folder=None, device="cuda"):
    """
    Get the Bellman discretization function and associated time steps.

    Parameters:
    - sde: Stochastic Differential Equation object containing the SDE configuration.
    - model: The denoising model
    - shape: Tuple representing the shape of the data.
    - scaler: Optional data scaler function. It is only needed if you use from_image option.
    - inverse_scaler: Optional inverse data scaler function. It is only needed if you use from_image option.
    - dataset: Optional dataset object for collecting samples. It is only needed if you use from_image option.
    - save_folder: Optional folder path for saving cost matrix files.
    - device: Device on which the computations should be performed (default is "cuda").

    Returns:
    - bellman_disc: Bellman discretization function that takes a model and initial state as inputs and returns time steps.
    - t_steps: Tuple of time steps associated with the Bellman discretization.
    """
    bellman_config = sde.bellman_config
    K_max = bellman_config.K_max
    sampling_num_samples = bellman_config.num_samples
    sampling_batch_size = bellman_config.batch_size
    stack = bellman_config.stack_batch
    cost_matrix_path = bellman_config.cost_matrix_path
    from_image = bellman_config.from_image
    loss = bellman_config.loss

    # Extract SDE configuration parameters
    num_steps = sde.num_steps
    sampling_shape = tuple(shape)
    sampling_shape = (sampling_batch_size,) + sampling_shape[1:]
    base_disc_name = sde.discretizing_method.split('_')[1]

    # Get time steps based on the discretization method
    if base_disc_name == "uniform":
        _, t_steps = get_uniform_disc(sde.T_start, sde.T_end, K_max)
        t_steps = t_steps[0]
    elif base_disc_name == "edm":
        _, t_steps = get_edm_disc(sde, K_max)
        t_steps = t_steps[0]
    t_steps = t_steps.to(device)

    # Initialize cost matrix
    cost_matrix = np.zeros((K_max + 1, K_max + 1))

    # Check if the cost matrix file already exists
    if not os.path.exists(cost_matrix_path):
        # If not, create the save folder if it doesn't exist
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)
        # Generate the cost matrix file path
        cost_matrix_path = os.path.join(save_folder, "cost_matrix_{}_{}_{}.npy".format(sde.discretizing_method, K_max, sampling_num_samples))

    # Check if the cost matrix file still does not exist
    if not os.path.exists(cost_matrix_path):
        # If not, calculate Bellman Stepsizes
        logging.info("Calculating Bellman Stepsizes")

        # Define cost function based on the chosen loss
        if loss == "l2_pixel":
            def cost_fn(sp1, sp2):
                return torch.sum(torch.square(sp1 - sp2))
        elif bellman_loss == "lpips":
            loss_fn = lpips.LPIPS(net='vgg')
            def cost_fn(sp1, sp2):
                img1 = torch.clip(inverse_scaler(sp1), 0.0, 1.0)
                img2 = torch.clip(inverse_scaler(sp2), 0.0, 1.0)
                img1 = 2. * img1 - 1.0
                img2 = 2. * img2 - 1.0
                return torch.sum(loss_fn(img1, img2))
        else:
            assert False, 'Not implemented'

        # Define a function to take samples
        if from_image:
            ds = dataset.batch(sampling_shape[0])
            ite = iter(ds)
            def take_samples(nsp):
                nonlocal t_steps
                batch = jax.tree_map(lambda x: x._numpy(), next(ite))
                batch = torch.from_numpy(batch['image']).to(device).float()
                batch = batch.reshape(sampling_shape)
                batch = batch.permute(0, 3, 1, 2)
                batch = scaler(batch)
                reversed_t_steps = torch.flip(t_steps, dims=(0,))
                class_labels = None
                try:
                    if model.label_dim:
                        class_labels = torch.eye(model.label_dim, device=device)[torch.randint(model.label_dim, size=[sampling_batch_size], device=device)][0:nsp]
                except:
                    pass
                return sde.euler_solver(batch[0:nsp], model, reversed_t_steps, class_labels=class_labels, return_trans=True)
        else:
            def take_samples(nsp):
                nonlocal t_steps
                x0 = sde.get_z0(torch.zeros(sampling_shape, device=device), train=False).to(device)
                class_labels = None
                try:
                    if model.label_dim:
                        class_labels = torch.eye(model.label_dim, device=device)[torch.randint(model.label_dim, size=[sampling_batch_size], device=device)][0:nsp]
                except:
                    pass
                return sde.euler_solver(x0[0:nsp], model, t_steps, class_labels=class_labels, return_trans=True)

        # Iterate to accumulate samples and calculate the cost matrix
        cnt = 0
        while cnt < sampling_num_samples:
            logging.info("Samples: %d" % (cnt))
            nsp = min(sampling_batch_size, sampling_num_samples - cnt)
            x1, mid_steps, v_ts, nfe = take_samples(nsp)

            for j in tqdm(range(K_max + 1)):
                targets = mid_steps[j].to(device)

                for start in range(0, j, stack):
                    end = min(j, start + stack)
                    estimates = (
                    torch.stack([mid_steps[i].to(device) for i in range(start, end)]) +
                    torch.stack([v_ts[i].to(device) for i in range(start, end)]) *
                    torch.stack([(t_steps[j] - t_steps[i]) for i in range(start, end)]).view(-1, 1, 1, 1, 1)
                    )
                    for i in range(start, end):
                        cost_matrix[i][j] += cost_fn(estimates[i - start], targets)
            cnt += nsp
            save_path = os.path.join(save_folder,
                                        os.path.join(save_folder, "cost_matrix_{}_{}_{}.npy".format(sde.discretizing_method, K_max, cnt)))
            saved_cost_matrix = cost_matrix / cnt
            tracking, _ = find_minimum_cost_path(saved_cost_matrix, num_steps + 1, "dp")
            print(tracking)
            np.save(save_path, saved_cost_matrix)

    # Load the existing cost matrix file
    cost_matrix = np.load(cost_matrix_path)
    tracking, _ = find_minimum_cost_path(cost_matrix, num_steps + 1, "dp")
    t_steps = t_steps[tracking]

    # Define the Bellman discretization function
    def bellman_disc(model, z):
        return t_steps

    # Return the Bellman discretization function and the associated time steps
    return bellman_disc, (t_steps,)

    
def find_minimum_cost_path(c, K, method):
    """
    Find the path with the minimum cost from point 0 to K_max using a given cost matrix.

    Description:
      If we use K_max discrete intervals for approximation, the matrix c should be a (K_max + 1) * (K_max + 1) matrix
      0  1  2  3  4 ... K_max.
      Want to find the path from 0 to K_max that incur the cheapest cost by using exactly K steps.
      The path begins at 0 and ends at K_max.
      For example for K = 2, K_max = 100: 0 -> 23 -> 100.

    Args:
        c: The cost matrix representing the cost of transitions between points.
        K: The number of evaluations, indicating the desired number of steps in the path.
        method: The method for finding the minimum cost path, where "ip" refers to integer programming and "dp" refers to dynamic programming. 
                It is recommended to use the "dp" method.

    Returns:
        A tuple containing:
            - The list of stopping points representing the path, starting from 0 and ending at K_max.
            - The total cost of the computed path.
    """
    # Dynamic Programming Method
    if method == "dp":
        # Adjust K to account for the sink node
        K = K - 1  
        n = c.shape[0]  # Number of stopping points
        assert n >= K + 1 

        # Base case: if there are as many stopping points as K steps, return the simple path
        if n == K + 1:
            tracking = [i for i in range(0, n)]
            total_cost = sum(c[i][i+1] for i in range(n-1))
            return tracking, total_cost

        # Initialize the dynamic programming table with infinity
        dp = np.full((n, K + 1), float('inf'))

        # Initialize the base case (when the budget is 1)
        dp[:, 1] = c[:, -1]

        # Perform dynamic programming to compute the minimum cost
        for k in range(2, K + 1):
            for i in range(n-1):
                for j in range(i+1, n-1):
                    dp[i][k] = min(dp[i][k], c[i][j] + dp[j][k - 1])

        # Reconstruct the path by backtracking
        tracking = [0]  # Start from t = 0
        cur = 0
        for k in range(K, 1, -1):
            for j in range(cur + 1, n):
                if dp[cur][k] == c[cur][j] + dp[j][k - 1]:
                    tracking.append(j)
                    cur = j
                    break
        tracking.append(n-1)
        total_cost = dp[0][K]
        return tracking, total_cost

    # Integer Programming Method
    if method == "ip":
        n = c.shape[0]  # Number of stopping points
        assert n >= K

        # Define binary decision variables for edges using CVXPY
        x = cp.Variable((n, n), boolean=True)

        # Set up constraints for the integer programming problem
        row_sums = cp.sum(x, axis=1)
        col_sums = cp.sum(x, axis=0)
        constraints = [
            row_sums <= 1,      # Each point has at most one outgoing edge
            col_sums <= 1,      # Each point has at most one incoming edge
            cp.sum(x) == K - 1  # Exactly K edges
        ]

        # Flow conservation constraints
        constraints.extend([
            row_sums[0] == 1,
            col_sums[0] == 0,
            row_sums[-1] == 0,
            col_sums[-1] == 1
        ])
        constraints.extend([row_sums[i] == col_sums[i]
                            for i in range(1, n - 1)])
        for i in range(n):
            for j in range(0, i+1):
                constraints.append(x[i, j] == 0)

        # Set up the objective function for the integer programming problem
        objective = cp.Minimize(cp.sum(cp.multiply(x, c)))
        problem = cp.Problem(objective, constraints)
        problem.solve()

        # Reconstruct the path by following the edges
        tracking = [0]  # Start from t = 0
        cur = 0
        while cur != n - 1:
            for i in range(n):
                if x.value[cur][i]:
                    tracking.append(i)
                    cur = i
                    break

        total_cost = problem.value
        return tracking, total_cost
