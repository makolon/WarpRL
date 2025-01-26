import os

import numpy as np
import torch
import warp as wp  # type: ignore
import warp.sim  # type: ignore

from warprl.envs.warp_env import WarpEnv
from warprl.utils import torch_utils as tu
from warprl.utils.env_utils import compute_env_offsets
from warprl.utils.gain_utils import print_gain

np.set_printoptions(precision=5, linewidth=256, suppress=True)


@wp.func
def xu_func(
    x: wp.float32,
    term: wp.float32,
    tol: wp.float32,
    pos_scale: wp.float32,
    neg_scale: wp.float32,
):
    x = x - term - tol
    x = wp.clamp(x, -1.0, tol)
    x = wp.select(x < 0.0, neg_scale * x * x, x)
    x = wp.select(x >= 0.0, pos_scale * x, x)
    return x


class HumanoidEnv(WarpEnv):
    def __init__(
        self,
        visualize=True,
        device="cuda:0",
        num_envs=128,
        num_agts=1,
        fps=60,
        substeps=16,
        episode_length=1000,
        logdir=None,
        no_grad=True,
        stochastic_init=False,
        MM_caching_frequency=1,
        early_termination=True,
        nan_state_fix=True,  # humanoid env needs this
        debug_gain=False,
        jacobian=False,
        jacobian_norm=None,
        density=1000.0,
        stiffness=5.0,
        damping=0.1,
        contact_ke=1.0e4,
        contact_kd=1.0e2,
        contact_kf=1.0e2,
        contact_ka=0.0,
        contact_mu=1.0,
        contact_restitution=0.1,
        contact_thickness=0.0,
        limit_ke=1.0e2,
        limit_kd=1.0e1,
        armature=0.007,
        armature_scale=10.0,
        action_scale=0.0,
        action_offset=0.0,
        motor_strength=100.0,
        motor_damping=1.0,
        action_penalty=-0.002,
        heading_rew_scale=1.0,
        height_rew_scale=10.0,
        height_rew_type="xu",
        joint_vel_obs_scaling=0.1,
        termination_height=0.74,
        termination_tolerance=0.1,
        up_rew_scale=0.1,
        renderer_type="opengl",
        integrator_type="xpbd",
    ):
        num_obs = 76
        num_act = 21

        super(HumanoidEnv, self).__init__(
            num_envs,
            num_agts,
            num_obs,
            num_act,
            fps,
            substeps,
            episode_length,
            MM_caching_frequency,
            no_grad,
            visualize,
            nan_state_fix,
            jacobian_norm,
            stochastic_init,
            jacobian,
            device,
            renderer_type,
            integrator_type,
        )

        # simulation parameters
        self.early_termination = early_termination
        self.density = density
        self.stiffness = stiffness
        self.damping = damping
        self.contact_ke = contact_ke
        self.contact_kd = contact_kd
        self.contact_kf = contact_kf
        self.contact_ka = contact_ka
        self.contact_mu = contact_mu
        self.contact_restitution = contact_restitution
        self.contact_thickness = contact_thickness
        self.limit_ke = limit_ke
        self.limit_kd = limit_kd
        self.armature = armature
        self.armature_scale = armature_scale

        # other parameters
        self.debug_gain = debug_gain
        self.action_scale = action_scale
        self.action_offset = action_offset
        self.motor_strength = motor_strength
        self.motor_damping = motor_damping
        self.action_penalty = action_penalty
        self.heading_rew_scale = heading_rew_scale
        self.height_rew_scale = height_rew_scale
        self.height_rew_type = height_rew_type
        self.joint_vel_obs_scaling = joint_vel_obs_scaling
        self.termination_height = termination_height
        self.termination_tolerance = termination_tolerance
        self.up_rew_scale = up_rew_scale

        # initialize simulation
        self.init_sim()

        # setup visualizer
        self.setup_visualizer(logdir)

    def init_sim(self):
        # set up model
        self.builder = wp.sim.ModelBuilder()
        self.num_joint_q = 28
        self.num_joint_qd = 27
        self.num_joint_act = 21

        articulation_builder = wp.sim.ModelBuilder()
        asset_folder = os.path.join(os.path.dirname(__file__), "assets")
        wp.sim.parse_mjcf(
            os.path.join(asset_folder, "humanoid.xml"),
            articulation_builder,
            density=self.density,
            stiffness=self.stiffness,
            damping=self.damping,
            contact_ke=self.contact_ke,
            contact_kd=self.contact_kd,
            contact_kf=self.contact_kf,
            contact_ka=self.contact_ka,
            contact_mu=self.contact_mu,
            contact_restitution=self.contact_restitution,
            contact_thickness=self.contact_thickness,
            limit_ke=self.limit_ke,
            limit_kd=self.limit_kd,
            armature=self.armature,
            armature_scale=self.armature_scale,
            enable_self_collisions=False,
            up_axis="y",
        )
        articulation_builder.joint_q[:7] = [
            0.0,
            1.35,
            0.0,
            *wp.quat_from_axis_angle((1.0, 0.0, 0.0), -np.pi * 0.5),
        ]

        offsets = compute_env_offsets(self.num_envs, env_offset=(3.0, 0.0, 3.0))
        for i in range(self.num_envs):
            self.builder.add_builder(
                articulation_builder, xform=wp.transform(offsets[i], wp.quat_identity())
            )
            self.builder.joint_axis_mode = [wp.sim.JOINT_MODE_TARGET_POSITION] * len(
                self.builder.joint_axis_mode
            )

        # set ground plane
        self.builder.set_ground_plane(ke=1e5, kd=1e3, kf=1e3, mu=1.0, restitution=0.1)

        # finalize model
        self.model = self.builder.finalize(device=self.device, requires_grad=True)
        self.model.gravity = wp.vec3f(0.0, -9.81, 0.0)
        self.model.joint_attach_ke = 1e3  # for semi-implicit integrator
        self.model.joint_attach_kd = 1e1  # for semi-implicit integrator
        self.model.joint_target_ke = wp.from_numpy(
            np.tile(self.motor_strength, (self.num_envs)),
            dtype=wp.float32,
            device=self.device,
        )
        self.model.joint_target_kd = wp.from_numpy(
            np.tile(self.motor_damping, (self.num_envs)),
            dtype=wp.float32,
            device=self.device,
        )

        if self.debug_gain:
            print_gain(self.model)

        # set up integrator
        if self.integrator_type == "xpbd":
            self.integrator = wp.sim.XPBDIntegrator()
        elif self.integrator_type == "semi-implicit":
            self.integrator = wp.sim.SemiImplicitIntegrator()
        else:
            self.integrator = wp.sim.FeatherstoneIntegrator(self.model)

        # set up state and control
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()

        # initialize numpy.ndarray target and start positions / rotations
        target_pos = np.tile(np.array([200.0, 0.0, 0.0]), (self.num_envs, 1))
        joint_q = self.state_0.joint_q.numpy().reshape(self.num_envs, self.num_joint_q)
        joint_qd = self.state_0.joint_qd.numpy().reshape(
            self.num_envs, self.num_joint_qd
        )
        joint_act = self.control.joint_act.numpy().reshape(
            self.num_envs, self.num_joint_act
        )

        # initialize numpy.ndarray joint state
        start_pos, start_rot = joint_q[:, :3], joint_q[:, 3:7]
        start_joint_q = joint_q[:, 7:]
        start_joint_qd = joint_qd
        start_joint_act = joint_act

        # initialize target and start positions / rotations
        self.target_pos = torch.from_numpy(target_pos.astype(np.float32)).to(
            self.device
        )
        self.start_pos = torch.from_numpy(start_pos.astype(np.float32)).to(self.device)
        self.start_rot = torch.from_numpy(start_rot.astype(np.float32)).to(self.device)

        # initialize joint state
        self.start_joint_q = torch.from_numpy(start_joint_q.astype(np.float32)).to(
            self.device
        )
        self.start_joint_qd = torch.from_numpy(start_joint_qd.astype(np.float32)).to(
            self.device
        )
        self.start_joint_act = torch.from_numpy(start_joint_act.astype(np.float32)).to(
            self.device
        )

        # basis vectors
        self.basis_vec0 = torch.tensor([1.0, 0.0, 0.0], device=self.device).repeat(
            (self.num_envs, 1)
        )
        self.basis_vec1 = torch.tensor([0.0, 1.0, 0.0], device=self.device).repeat(
            (self.num_envs, 1)
        )

        if self.model.ground:
            wp.sim.collide(self.model, self.state_0)

    def static_init_func(self, env_ids):
        joint_q = wp.to_torch(self.state_0.joint_q).clone()
        joint_qd = wp.to_torch(self.state_0.joint_qd).clone()
        joint_act = wp.to_torch(self.control.joint_act).clone()

        # iterate over each environment
        for env_id in env_ids:
            # calculate the base indices for this environment
            q_base = env_id * self.num_joint_q
            qd_base = env_id * self.num_joint_qd
            act_base = env_id * self.num_joint_act

            # initialize joint positions / velocities / actions
            joint_q[q_base : q_base + 3] = self.start_pos[env_id]
            joint_q[q_base + 3 : q_base + 7] = self.start_rot[env_id]
            joint_q[q_base + 7 : q_base + self.num_joint_q] = self.start_joint_q[env_id]
            joint_qd[qd_base : qd_base + self.num_joint_qd] = self.start_joint_qd[
                env_id
            ]
            joint_act[act_base : act_base + self.num_joint_act] = self.start_joint_act[
                env_id
            ]

        self.state_0.joint_q = wp.from_torch(joint_q, dtype=wp.float32)
        self.state_0.joint_qd = wp.from_torch(joint_qd, dtype=wp.float32)
        self.control.joint_act = wp.from_torch(joint_act, dtype=wp.float32)

    def set_action(self, action):
        # flatten the action tensor to 1D
        action = action.flatten()
        action = action * self.action_scale
        return action

    def get_observation(self, joint_q, joint_qd, joint_act):
        # extract joint positions and velocities
        if isinstance(joint_q, wp.array):
            joint_q = wp.to_torch(joint_q, requires_grad=True)
        if isinstance(joint_qd, wp.array):
            joint_qd = wp.to_torch(joint_qd, requires_grad=True)
        if isinstance(joint_act, wp.array):
            joint_act = wp.to_torch(joint_act, requires_grad=True)

        joint_q = joint_q.view(self.num_envs, self.num_joint_q)
        joint_qd = joint_qd.view(self.num_envs, self.num_joint_qd)
        joint_act = joint_act.view(self.num_envs, self.num_joint_act)

        torso_pos = joint_q[:, :3].clone()
        torso_rot = joint_q[:, 3:7].clone()
        lin_vel = joint_qd[:, 3:6].clone()
        ang_vel = joint_qd[:, 0:3].clone()

        # convert the linear velocity of the torso from twist representation to the velocity of the center of mass in world frame
        lin_vel = lin_vel - torch.cross(torso_pos, ang_vel, dim=-1)

        to_target = self.target_pos + self.start_pos - torso_pos
        to_target[:, 1] = 0.0

        target_dirs = tu.normalize(to_target)
        inv_start_rot = tu.quat_conjugate(self.start_rot)
        torso_quat = tu.quat_mul(torso_rot, inv_start_rot)

        up_vec = tu.quat_rotate(torso_quat, self.basis_vec1)
        heading_vec = tu.quat_rotate(torso_quat, self.basis_vec0)

        return torch.cat(
            [
                torso_pos[:, 1:2],  # 0
                torso_rot,  # 1:5
                lin_vel,  # 5:8
                ang_vel,  # 8:11
                joint_q.view(self.num_envs, -1)[:, 7:],  # 11:32
                self.joint_vel_obs_scaling
                * joint_qd.view(self.num_envs, -1)[:, 6:],  # 32:53
                up_vec[:, 1:2],  # 54
                (heading_vec * target_dirs).sum(dim=-1).unsqueeze(-1),  # 55
                joint_act[:, :].clone(),  # 55:74
            ],
            dim=-1,
        )

    def calculate_reward(self, obs, action):
        up_reward = self.up_rew_scale * obs[:, 53]
        heading_reward = self.heading_rew_scale * obs[:, 54]
        height_reward = xu_func(
            obs[:, 0],
            self.termination_height,
            self.termination_tolerance,
            self.height_rew_scale,
            -wp.float32(200.0),
        )

        progress_reward = obs[:, 5]
        self.primal = progress_reward.detach()

        return progress_reward + up_reward + heading_reward + height_reward

    def compute_termination(self, obs):
        termination = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        if self.early_termination:
            termination = obs[:, 0] < self.termination_height
        return termination.int()
