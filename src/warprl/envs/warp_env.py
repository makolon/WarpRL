import os
import sys
from abc import abstractmethod

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import torch
import warp as wp  # type: ignore
import warp.sim  # type: ignore
import warp.sim.render  # type: ignore

try:
    from pxr import Usd  # type: ignore # noqa: F401
except ModuleNotFoundError:
    print("No pxr package")

from gym import spaces  # type: ignore


def jacobian(output, input, max_out_dim=None):
    """Computes the jacobian of output tensor with respect to the input"""
    num_envs, input_dim = input.shape
    output_dim = output.shape[1]
    if max_out_dim:
        output_dim = min(output_dim, max_out_dim)
    jacobians = torch.zeros((num_envs, output_dim, input_dim), dtype=torch.float32)
    for out_idx in range(output_dim):
        select_index = torch.zeros(output.shape[1])
        select_index[out_idx] = 1.0
        e = torch.tile(select_index, (num_envs, 1)).cuda()
        try:
            (grad,) = torch.autograd.grad(
                outputs=output, inputs=input, grad_outputs=e, retain_graph=True
            )
            jacobians[:, out_idx, :] = grad.view(num_envs, input_dim)
        except RuntimeError as err:
            print(f"WARN: Couldn't compute jacobian for {out_idx} index")
            print(err)


class ForwardSimulation(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        joint_q,
        joint_qd,
        action,
        model,
        state_0,
        control,
        integrator,
        sim_substeps,
        sim_dt,
    ):
        """
        Forward pass for the simulation step.
        """
        # save inputs for backward pass
        ctx.model = model
        ctx.state_0 = state_0
        ctx.control = control
        ctx.integrator = integrator
        ctx.sim_substeps = sim_substeps
        ctx.sim_dt = sim_dt

        # gradient tracking for the action
        ctx.joint_q = wp.from_torch(joint_q)
        ctx.joint_qd = wp.from_torch(joint_qd)
        ctx.action = wp.from_torch(action)

        # assign inputs to Warp state variables
        # NOTE: this is necessary to ensure that the integrator uses the correct values
        ctx.state_0.joint_q = ctx.joint_q  # current joint positions
        ctx.state_0.joint_qd = ctx.joint_qd  # current joint velocities
        ctx.control.joint_act = ctx.action  # civen joint action

        # prepare state for saving the simulation result
        state_1 = model.state()

        # prepare a Warp Tape for gradient tracking
        ctx.tape = wp.Tape()

        with ctx.tape:
            # Simulate forward
            state_0.clear_forces()
            wp.sim.collide(model, state_0)
            integrator.simulate(model, state_0, state_1, sim_dt, control)

        # save the state for the backward pass
        ctx.state_1 = state_1

        # return the joint positions and velocities
        return wp.to_torch(state_1.joint_q), wp.to_torch(state_1.joint_qd)

    @staticmethod
    def backward(ctx, grad_joint_q, grad_joint_qd):
        """
        Backward pass for gradient computation.
        """
        # assign gradients to Warp state variables
        ctx.state_1.joint_q.grad = wp.from_torch(grad_joint_q, dtype=wp.float32)
        ctx.state_1.joint_qd.grad = wp.from_torch(grad_joint_qd, dtype=wp.float32)

        # backpropagate through the Warp simulation
        ctx.tape.backward()

        # return adjoint w.r.t. inputs
        return (
            wp.to_torch(ctx.tape.gradients[ctx.joint_q]),  # joint_q
            wp.to_torch(ctx.tape.gradients[ctx.joint_qd]),  # joint_qd
            wp.to_torch(ctx.tape.gradients[ctx.action]),  # action
            None,  # state_0
            None,  # control
            None,  # model
            None,  # integrator
            None,  # sim_substeps
            None,  # sim_dt
        )


class WarpEnv:
    def __init__(
        self,
        num_envs=1,
        num_agts=1,
        num_obs=1,
        num_act=1,
        fps=60,
        substeps=16,
        episode_length=100,
        MM_caching_frequency=1,
        no_grad=True,
        visualize=True,
        nan_state_fix=False,
        jacobian_norm=None,
        stochastic_init=False,
        jacobian=False,
        device="cuda:0",
        renderer_type="opengl",
        integrator_type="featherstone",
    ):
        self.no_grad = no_grad
        wp.config.no_grad = self.no_grad

        # if true resets all envs on earfly termination
        self.nan_state_fix = nan_state_fix
        self.jacobian = jacobian
        self.jacobian_norm = jacobian_norm
        self.stochastic_init = stochastic_init

        self.episode_length = episode_length
        self.max_episode_steps = episode_length
        self.device = device
        self.visualize = visualize
        self.renderer_type = renderer_type
        self.integrator_type = integrator_type
        self.MM_caching_frequency = MM_caching_frequency

        self.num_environments = num_envs
        self.num_agents = num_agts
        self.num_observations = num_obs
        self.num_actions = num_act

        # initialize simulation time
        self.sim_time = 0.0
        self.frame_dt = 1.0 / fps
        self.sim_substeps = substeps
        self.sim_dt = self.frame_dt / self.sim_substeps

        # initialize observation and action space
        self.obs_space = spaces.Box(
            np.ones(self.num_observations) * -np.Inf,
            np.ones(self.num_observations) * np.Inf,
        )
        self.act_space = spaces.Box(
            np.ones(self.num_actions) * -1.0, np.ones(self.num_actions) * 1.0
        )

        # create buffers
        self.progress_buf = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.long, requires_grad=False
        )

    def initialize_buffers(self, env_ids):
        # initialize the progress buffer
        self.progress_buf[env_ids] = 0

    def forward(self, action):
        """
        Execute the forward simulation step using the ForwardSimulation class.
        """
        joint_q = wp.to_torch(self.state_0.joint_q)
        joint_qd = wp.to_torch(self.state_0.joint_qd)

        # NOTE: action must be a 1D torch tensor with shape (num_actions * num_envs)
        joint_q_next, joint_qd_next = ForwardSimulation.apply(
            joint_q,
            joint_qd,
            action,
            self.model,
            self.state_0,
            self.control,
            self.integrator,
            self.sim_substeps,
            self.sim_dt,
        )

        # update the state_0 with the new joint positions and velocities
        self.state_0.joint_q = wp.from_torch(joint_q_next)
        self.state_0.joint_qd = wp.from_torch(joint_qd_next)

        return joint_q_next, joint_qd_next

    def step(self, action):
        """
        Perform one simulation step with the given action.
        """
        # set and clip the action
        action = torch.clip(action, -1.0, 1.0).to(self.device)
        action = self.set_action(action)

        # forward simulation
        for _ in range(self.sim_substeps):
            joint_q_next, joint_qd_next = self.forward(action)

        # compute observations, rewards, and terminations
        # NOTE: these functions must be differentiable for joint_q_next, joint_qd_next, and self.action.
        obs_buf = self.get_observation(joint_q_next, joint_qd_next, action)
        reward_buf = self.calculate_reward(obs_buf)
        termination_buf = self.compute_termination(obs_buf)
        self.progress_buf += 1

        # reset environments that are done
        reset_env_ids = self.reset_idx(termination_buf, self.progress_buf)
        if len(reset_env_ids) > 0:
            self.reset(reset_env_ids)

        # render the environment
        self.render()

        # prepare extras for debugging
        extras = {
            "obs_before_reset": obs_buf,
            "termination": termination_buf,
            "truncation": self.progress_buf >= self.episode_length,
            "primal": self.primal,
        }

        return obs_buf, reward_buf, termination_buf, extras

    def reset(self, env_ids=None, grads=False, force_reset=False):
        if grads:
            """
            This function starts collecting a new trajectory from the
            current states but cuts off the computation graph to the
            previous states. It has to be called every time the algorithm
            starts an episode and it returns the observation vectors.

            Note: force_reset resets all envs and is here for
            compatibility with rl_games.
            """
            self.clear_grad()
            obs_buf = self.get_observation(
                self.state_0.joint_q, self.state_0.joint_qd, self.control.joint_act
            )
            return obs_buf

        if env_ids is None or force_reset:
            # reset all environemnts
            env_ids = torch.arange(self.num_envs, dtype=torch.int32, device=self.device)

        if env_ids is not None:
            # reset the environments
            self.static_init_func(env_ids)
            self.initialize_buffers(env_ids)
            obs_buf = self.get_observation(
                self.state_0.joint_q, self.state_0.joint_qd, self.control.joint_act
            )

        return obs_buf

    def reset_idx(self, done, progress):
        # generate environment IDs as a Torch tensor for indexing
        env_ids = torch.arange(self.num_envs, dtype=torch.int32, device=self.device)

        # check for truncation based on episode length
        truncation = progress > (self.episode_length - 1)

        # perform element-wise OR operation on tensors
        termination = done | truncation

        # convert filtered_env_ids back to Warp array if needed
        return env_ids[termination]

    def setup_visualizer(self, logdir: str = None):
        if self.visualize:
            filename = f"{logdir}/{self.__class__.__name__}_{self.num_envs}.usd"
            if self.renderer_type == "opengl":
                self.renderer = wp.sim.render.SimRendererOpenGL(self.model, filename)
            elif self.renderer_type == "usd":
                self.renderer = wp.sim.render.SimRendererUsd(self.model, filename)
            else:
                raise ValueError(
                    f"The speficied renderer type {self.renderer_type} does not exist."
                )
            self.renderer.draw_points = True
            self.renderer.draw_springs = True
            self.renderer.draw_shapes = True
            self.render_time = 0.0

    def render(self):
        if self.visualize:
            if self.renderer is None:
                return

            self.renderer.begin_frame(self.sim_time)
            self.renderer.render(self.state_0)
            self.renderer.end_frame()

    def clear_grad(self, checkpoint=None):
        with torch.no_grad():
            if checkpoint is None:
                checkpoint = {}
                checkpoint["joint_q"] = wp.to_torch(self.state_0.joint_q).clone()
                checkpoint["joint_qd"] = wp.to_torch(self.state_0.joint_qd).clone()
                checkpoint["joint_act"] = wp.to_torch(self.control.joint_act).clone()
                checkpoint["progress_buf"] = self.progress_buf.clone()

            self.state_0 = self.model.state()
            self.control = self.model.control()
            self.state_0.joint_q = wp.from_torch(checkpoint["joint_q"])
            self.state_0.joint_qd = wp.from_torch(checkpoint["joint_qd"])
            self.control.joint_act = wp.from_torch(checkpoint["joint_act"])
            self.progress_buf = checkpoint["progress_buf"]

    def get_checkpoint(self):
        checkpoint = {}
        checkpoint["joint_q"] = wp.to_torch(self.state_0.joint_q).clone()
        checkpoint["joint_qd"] = wp.to_torch(self.state_0.joint_qd).clone()
        checkpoint["joint_act"] = wp.to_torch(self.control.joint_act).clone()
        checkpoint["progress_buf"] = self.progress_buf.clone()

        return checkpoint

    @abstractmethod
    def static_init_func(self, env_ids):
        pass

    @abstractmethod
    def get_observation(self, joint_q, joint_qd, action):
        pass

    @abstractmethod
    def set_action(self, action):
        pass

    @abstractmethod
    def calculate_reward(self, obs):
        pass

    @abstractmethod
    def compute_termination(self, obs):
        pass

    @property
    def observation_space(self):
        return self.obs_space

    @property
    def action_space(self):
        return self.act_space

    @property
    def num_envs(self):
        return self.num_environments

    @property
    def num_agts(self):
        return self.num_agents

    @property
    def num_acts(self):
        return self.num_actions

    @property
    def num_obs(self):
        return self.num_observations
