import abc
import torch
import numpy as np
from models import utils as mutils
import dnnlib
from torch_utils import distributed as dist
import pickle
import tqdm
import PIL.Image


class SDE(abc.ABC):
    """Base class for Stochastic Differential Equations."""
    
    def __init__(self):
        super().__init__()

    @property
    @abc.abstractmethod
    def T_start(self):
        """Property representing the start time."""
        pass

    @property
    @abc.abstractmethod
    def T_end(self):
        """Property representing the end time."""
        pass

    @abc.abstractmethod
    def noise_shift(self, model, x, t, num_steps):
        """Abstract method for noise shifting at each step."""
        pass

    @abc.abstractmethod
    def scale_noise(self, x, t):
        """Abstract method for scaling noise."""
        pass

    @abc.abstractmethod
    def wrap_model(self, model):
        """Abstract method for wrapping the model (the wrapped model f(x) satisfied dy = f(x) dx)."""
        pass

    def euler_solver(self, x_init, model, t_steps, class_labels=None, return_trans=False):
        """
        Euler method for solving the Ordinary Differential Equation (ODE).

        Args:
            x_init (torch.Tensor): Initial state.
            model: Model used for the simulation.
            t_steps (torch.Tensor): Time steps.
            class_labels: Labels for the model (if applicable).
            return_trans (bool): Whether to return intermediate results.

        Returns:
            Tuple containing the final state and optional intermediate results.
        """
        # Get the shape and device from the initial state
        shape = x_init.shape
        device = x_init.device
        
        # Calculate the number of steps
        num_steps = t_steps.shape[0] - 1

        # Wrap the model
        model_fn = self.wrap_model(model=model)

        # Initialize the next time step and the next state
        t_next = t_steps[0]
        x_next = self.scale_noise(x=x_init.detach().clone(), t=t_next)

        # Initialize lists to store intermediate results if required
        if return_trans:
            mid_steps = []
            v_ts = []
            mid_steps.append(x_next.clone().cpu())

        # Iterate over time steps using Euler method
        for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])):  # 0, ..., N-1
            x_cur = x_next
            
            # Apply noise shifting
            x_cur, t_cur = self.noise_shift(model=model, x=x_cur, t=t_cur, num_steps=num_steps)
            
            # Calculate the time step size
            h = t_next - t_cur
            
            # Evaluate the model function at the current state and time
            d_cur = model_fn(x_cur, t_cur, class_labels)
            
            # Update the next state using Euler method
            x_next = x_cur + h * d_cur
            
            # Store intermediate results if required
            if return_trans:  
                mid_steps.append(x_next.clone().cpu())  
                v_ts.append(d_cur.clone().cpu())

        # Return the final state and optional intermediate results
        if return_trans:
            return x_next, mid_steps, v_ts, num_steps
        return x_next, num_steps

  
    def heun2_solver(self, x_init, model, t_steps, alpha=1.0, class_labels=None, return_trans=False):
      """
      Heun2 method for solving the Ordinary Differential Equation (ODE).

      Args:
          x_init (torch.Tensor): Initial state.
          model: Model used for the simulation.
          t_steps (torch.Tensor): Time steps.
          alpha (float): Parameter for the method.
          class_labels: Labels for the model (if applicable).
          return_trans (bool): Whether to return intermediate results.

      Returns:
          Tuple containing the final state and optional intermediate results.
      """
      # Get the shape and device from the initial state
      shape = x_init.shape
      device = x_init.device
      
      # Calculate the number of steps
      num_steps = t_steps.shape[0] - 1

      # Wrap the model
      model_fn = self.wrap_model(model=model)

      # Initialize the next time step and the next state
      t_next = t_steps[0]
      x_next = self.scale_noise(x=x_init.detach().clone(), t=t_next)

      # Initialize lists to store intermediate results if required
      if return_trans:
          mid_steps = []
          v_ts = []
          mid_steps.append(x_next.clone())

      # Iterate over time steps using Heun2 method
      for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])):  # 0, ..., N-1
          x_cur = x_next
          
          # Apply noise shifting
          x_cur, t_cur = self.noise_shift(model=model, x=x_cur, t=t_cur, num_steps=num_steps)
          
          # Calculate the time step size
          h = t_next - t_cur
          
          # Evaluate the model function at the current state and time
          d_cur = model_fn(x_cur, t_cur, class_labels)
          
          # Predict the next state using the Heun2 method
          x_prime = x_cur + alpha * h * d_cur
          t_prime = t_cur + alpha * h
          d_prime = model_fn(x_prime, t_prime, class_labels)
          d_cur = (1 - 1 / (2 * alpha)) * d_cur + 1 / (2 * alpha) * d_prime
          x_next = x_cur + h * d_cur
          
          # Store intermediate results if required
          if return_trans:
              mid_steps.append(x_next.clone())  
              v_ts.append(d_cur.clone())

      # Return the final state and optional intermediate results
      if return_trans:
          return x_next, mid_steps, v_ts, num_steps * 2
      return x_next, num_steps * 2

  
    def rk45_solver(self, x_init, model, t_steps, rtol=1e-5, atol=1e-5, method='RK45', class_labels=None, return_trans=False):
      """
      RK45 method for solving the Ordinary Differential Equation (ODE).

      Args:
          x_init (torch.Tensor): Initial state.
          model: Model used for the simulation.
          t_steps (torch.Tensor): Time steps.
          rtol (float): Relative tolerance for the solver.
          atol (float): Absolute tolerance for the solver.
          method (str): Integration method for scipy's solve_ivp.
          class_labels: Labels for the model (if applicable).
          return_trans (bool): Whether to return intermediate results.

      Returns:
          Tuple containing the final state and the number of function evaluations.
      """
      from models.utils import from_flattened_numpy, to_flattened_numpy, get_score_fn  
      from scipy import integrate
      # Get the shape and device from the initial state
      shape = x_init.shape
      device = x_init.device
      
      # Wrap the model
      model_fn = self.wrap_model(model=model)

      # Initialize the next time step and the next state
      t_next = t_steps[0]
      x_next = self.scale_noise(x=x_init.detach().clone(), t=t_next)

      # Define the ODE function for the solver
      def ode_func(t, x):
          x = from_flattened_numpy(x, shape).to(device).type(torch.float32)
          drift = model_fn(x, t, class_labels)
          return to_flattened_numpy(drift)

      # Use the black-box ODE solver for the probability flow ODE
      solution = integrate.solve_ivp(ode_func, (t_steps[0], t_steps[-1]), to_flattened_numpy(x_next),
                                    rtol=rtol, atol=atol, method=method)

      # Extract the final state from the solution
      x = torch.tensor(solution.y[:, -1]).reshape(shape).to(device).type(torch.float32)

      # Get the number of function evaluations
      nfe = solution.nfev

      # Print the number of function evaluations (optional)
      print('Number of Function Evaluations (NFE):', nfe)

      # Return the final state and the number of function evaluations
      return x, nfe

       
class EDM(SDE):
    def __init__(self, mode="eval", init_type='gaussian',
                 num_steps=18, discretizing_method="edm",
                 solver="heun2", reflow_t_schedule='uniform',
                 reflow_loss='l2', redress_loss='l2',
                 edm_config=None, bellman_config=None):
        super().__init__()
        self.mode = mode
        self.init_type = init_type
        self.discretizing_method = discretizing_method
        self.solver = solver
        self.num_steps = num_steps
        self.edm_config = edm_config
        self.bellman_config = bellman_config
        self.sigma_min = self.edm_config.sigma_min
        self.sigma_max = self.edm_config.sigma_max
        self.rho = self.edm_config.rho
        self.S_churn = self.edm_config.S_churn
        self.S_min = self.edm_config.S_min
        self.alpha = self.edm_config.alpha
        self.S_max = self.edm_config.S_max
        self.S_noise = self.edm_config.S_noise
        self.sigma = self.edm_config.sigma
        self.sigma_deriv = self.edm_config.sigma_deriv
        self.sigma_inv = self.edm_config.sigma_inv
        self.s = self.edm_config.s
        self.s_deriv = self.edm_config.s_deriv
        # Setup additional parameters based on the mode
        if self.mode == "reflow":
            self.reflow_t_schedule = reflow_t_schedule
            self.reflow_loss = reflow_loss
            if 'lpips' in reflow_loss:
                self.lpips_model = lpips.LPIPS(net='vgg')
                self.lpips_model = self.lpips_model.cuda()
                for p in self.lpips_model.parameters():
                    p.requires_grad = False

        if self.mode == "redress":
            self.redress_loss = redress_loss
            if 'lpips' in redress_loss:
                self.lpips_model = lpips.LPIPS(net='vgg')
                self.lpips_model = self.lpips_model.cuda()
                for p in self.lpips_model.parameters():
                    p.requires_grad = False

    @property
    def T_start(self):
        return self.sigma_inv(self.sigma_max)

    @property
    def T_end(self):
        return 0.

    def noise_shift(self, model, x, t, num_steps=None):
        if num_steps is None:
            num_steps = self.num_steps

        gamma = min(self.S_churn / num_steps, np.sqrt(2) - 1) if self.S_min <= self.sigma(t) <= self.S_max else 0
        t_hat = self.sigma_inv(model.module.round_sigma(self.sigma(t) + gamma * self.sigma(t)))
        x_hat = self.s(t_hat) / self.s(t) * x + (self.sigma(t_hat) ** 2 - self.sigma(t) ** 2).clip(min=0).sqrt() * \
                self.s(t_hat) * self.S_noise * torch.randn_like(x)
        return x_hat, t_hat

    def scale_noise(self, x, t):
        return x * (self.sigma(t) * self.s(t))

    def wrap_model(self, model, train=False):
        eval_model = mutils.get_model_fn(model, train=train)

        def wrapped_model(x, t, class_labels=None):
            vec_t = torch.ones(x.shape[0], device=x.device) * self.sigma(t)
            denoised = eval_model(x / self.s(t), vec_t, class_labels).to(torch.float64)
            term_x = (self.sigma_deriv(t) / self.sigma(t) + self.s_deriv(t) / self.s(t)).view(-1, 1, 1,
                                                                                             1).repeat(1, x.shape[1],
                                                                                                      x.shape[2],
                                                                                                      x.shape[3])
            term_denoised = (self.sigma_deriv(t) * self.s(t) / self.sigma(t)).view(-1, 1, 1,
                                                                                    1).repeat(1, x.shape[1],
                                                                                             x.shape[2], x.shape[3])
            return term_x.to(x.device) * x - term_denoised.to(x.device) * denoised

        return wrapped_model

    def heun2_solver(self, x_init, model, t_steps, alpha=None, class_labels=None, return_trans=False):
        """
        Heun2 solver for the Stochastic Differential Equation (SDE).

        Args:
            x_init (torch.Tensor): Initial state.
            model: Model used for the simulation.
            t_steps (torch.Tensor): Time steps.
            alpha (float): Parameter for the solver.
            class_labels: Labels for the model (if applicable).
            return_trans (bool): Whether to return intermediate results.

        Returns:
            Tuple containing the final state and optional intermediate results.
        """
        shape = x_init.shape
        device = x_init.device
        num_steps = t_steps.shape[0] - 1

        if alpha is None:
            alpha = self.alpha

        # Wrap the model to have the form dx = vdt (integral from )
        model_fn = self.wrap_model(model)

        # Initialize the next time step and the next state
        t_next = t_steps[0]
        x_next = self.scale_noise(x=x_init.detach().clone(), t=t_next)

        # Initialize lists to store intermediate results if required
        if return_trans:
            mid_steps = []
            v_ts = []
            mid_steps.append(x_next.clone())

        # Iterate over time steps using Heun2 method
        for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])):  # 0, ..., N-1
            x_cur = x_next

            # Increase noise temporarily.
            x_hat, t_hat = self.noise_shift(model, x_cur, t_cur)
            # Euler step.
            h = t_next - t_hat
            d_cur = model_fn(x_hat, t_hat, class_labels)

            # Apply 2nd order correction.
            if i == num_steps - 1:
                pass
            else:
                x_prime = x_hat + alpha * h * d_cur
                t_prime = t_hat + alpha * h
                d_prime = model_fn(x_prime, t_prime, class_labels)
                d_cur = (1 - 1 / (2 * alpha)) * d_cur + 1 / (2 * alpha) * d_prime

            x_next = x_hat + h * d_cur

            # Store intermediate results if required
            if return_trans:
                mid_steps.append(x_next.detach().clone())
                v_ts.append(d_cur.detach().clone())

        # Return the final state and optional intermediate results
        if return_trans:
            return x_next, mid_steps, v_ts, num_steps * 2 - 1
        return x_next, num_steps * 2 - 1

    def get_z0(self, batch, train=True):
        n, c, h, w = batch.shape
        if self.init_type == 'gaussian':
            # Standard Gaussian
            cur_shape = (n, c, h, w)
            return torch.randn(cur_shape)
        else:
            raise NotImplementedError("INITIALIZATION TYPE NOT IMPLEMENTED")



class RectifiedFlow(SDE):
    def __init__(self, mode="eval", eps=1e-3,
                 init_type='gaussian', noise_scale=1.0,
                 reflow_t_schedule='uniform', reflow_loss='l2', redress_loss='l2',
                 discretizing_method=None, solver="rk45", num_steps=None,
                 edm_config=None, bellman_config=None):
        super().__init__()
        self.mode = mode
        self.num_steps = num_steps if num_steps is not None else 10  # Default value if not provided
        self.init_type = init_type
        self.noise_scale = noise_scale
        self.discretizing_method = discretizing_method
        self.solver = solver
        self.eps = eps
        self.edm_config = edm_config
        self.bellman_config = bellman_config

        if self.mode == 'reflow':
            self.reflow_t_schedule = reflow_t_schedule
            self.reflow_loss = reflow_loss
            if 'lpips' in reflow_loss:
                self.lpips_model = lpips.LPIPS(net='vgg')
                self.lpips_model = self.lpips_model.cuda()
                for p in self.lpips_model.parameters():
                    p.requires_grad = False

        if self.mode == "redress":
            self.redress_loss = redress_loss
            if 'lpips' in redress_loss:
                self.lpips_model = lpips.LPIPS(net='vgg')
                self.lpips_model = self.lpips_model.cuda()
                for p in self.lpips_model.parameters():
                    p.requires_grad = False

    @property
    def T_start(self):
        return 0.

    @property
    def T_end(self):
        return 1.

    @property
    def T(self):
        return 1.

    def noise_shift(self, model, x, t, num_steps=None):
        return x, t

    def scale_noise(self, x, t):
        return x

    def wrap_model(self, model, train=False):
        eval_model = mutils.get_model_fn(model, train=train)

        def wrapped_model(x, t, class_labels=None):
            vec_t = torch.ones(x.shape[0], device=x.device) * (t * (self.T_end - self.T_start - self.eps) + self.eps) * 999
            return eval_model(x, vec_t, class_labels)

        return wrapped_model

    def get_z0(self, batch, train=True):
        n, c, h, w = batch.shape

        if self.init_type == 'gaussian':
            # Standard Gaussian
            cur_shape = (n, c, h, w)
            return torch.randn(cur_shape) * self.noise_scale
        else:
            raise NotImplementedError("INITIALIZATION TYPE NOT IMPLEMENTED")