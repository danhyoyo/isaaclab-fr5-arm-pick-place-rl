"""Three policy actions move the TCP; the IK controller maintains a downward grasp."""
import torch
from isaaclab.envs.mdp.actions.task_space_actions import DifferentialInverseKinematicsAction
from .constants import DOWN_QUAT, WORKSPACE_LOW, WORKSPACE_HIGH, ROBOT_BASE_HEIGHT


class CartesianDeltaAction(DifferentialInverseKinematicsAction):
    """Convert three Cartesian policy deltas into a persistent TCP pose command.

    The policy controls ``[delta_x, delta_y, delta_z]``.  This class integrates
    those deltas into a target position and leaves the joint-level conversion to
    Isaac Lab's differential IK controller.
    """

    def __init__(self, cfg, env):
        # Let the Isaac Lab parent create the action buffers and IK controller.
        super().__init__(cfg, env)

        # One persistent TCP target is kept for every parallel environment.
        # Shape: [num_envs, 3], with columns [x, y, z].
        self._target_position = torch.zeros(self.num_envs, 3, device=self.device)

        # After reset, initialize each target from the TCP's actual pose on the
        # first action so the robot does not jump to an arbitrary target.
        self._needs_initialization = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)

    @property
    def action_dim(self):
        # This action term consumes only the three Cartesian position values.
        # The gripper is configured as a separate action term.
        return 3

    @property
    def target_position(self):
        # Read the current controlled frame pose.  Before the first action of an
        # episode, use that pose as the initial target; afterwards return the
        # integrated target stored above.
        pos, _ = self._compute_frame_pose()
        return torch.where(self._needs_initialization[:, None], pos, self._target_position)

    def reset(self, env_ids=None):
        # Reset the parent action/controller for all environments or only the
        # subset whose episode ended.
        super().reset(env_ids)

        # The next action for the selected environments will reinitialize its
        # target from the current TCP pose.
        self._needs_initialization[env_ids if env_ids is not None else slice(None)] = True

    def process_actions(self, actions):
        # Keep the raw policy output for logging/observations.
        self._raw_actions[:] = actions

        # Policy outputs are normalized to [-1, 1].  Scaling converts them to a
        # physical TCP displacement; ACTION_STEP is 6 mm per policy step.
        self._processed_actions[:] = actions.clamp(-1.0, 1.0) * self._scale

        # Current position/orientation of the controlled TCP frame.
        pos, quat = self._compute_frame_pose()

        # Convert workspace limits to tensors on the same device as the robot.
        low = torch.tensor(WORKSPACE_LOW, device=self.device)
        high = torch.tensor(WORKSPACE_HIGH, device=self.device)

        # The workspace is expressed in environment coordinates, while the IK
        # controller command is relative to the robot root.  Remove the robot's
        # base height from the Z limits before sending the command.
        low[2] -= ROBOT_BASE_HEIGHT
        high[2] -= ROBOT_BASE_HEIGHT

        # Integrate the Cartesian delta into the persistent target and clamp it
        # to the allowed workspace.  This is stepwise delta integration, not a
        # cubic/quintic trajectory interpolator.
        desired = torch.maximum(torch.minimum(self.target_position + self._processed_actions, high), low)
        self._target_position[:] = desired
        self._needs_initialization[:] = False

        # Keep the gripper orientation fixed downward for top-down grasping.
        rotation = torch.tensor(DOWN_QUAT, device=self.device).expand(self.num_envs, -1)

        # Concatenate position and quaternion into a 7D pose command
        # [x, y, z, qx, qy, qz, qw].  The DLS differential IK controller then
        # converts this pose error into joint targets for j1...j6.
        self._ik_controller.set_command(torch.cat([desired, rotation], dim=-1), pos, quat)
