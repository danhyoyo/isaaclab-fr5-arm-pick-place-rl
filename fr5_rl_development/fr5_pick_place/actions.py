"""Three policy actions move the TCP; the IK controller maintains a downward grasp."""
import torch
from isaaclab.envs.mdp.actions.task_space_actions import DifferentialInverseKinematicsAction
from .constants import DOWN_QUAT, WORKSPACE_LOW, WORKSPACE_HIGH, ROBOT_BASE_HEIGHT


class CartesianDeltaAction(DifferentialInverseKinematicsAction):
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self._target_position = torch.zeros(self.num_envs, 3, device=self.device)
        self._needs_initialization = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)

    @property
    def action_dim(self):
        return 3

    @property
    def target_position(self):
        pos, _ = self._compute_frame_pose()
        return torch.where(self._needs_initialization[:, None], pos, self._target_position)

    def reset(self, env_ids=None):
        super().reset(env_ids)
        self._needs_initialization[env_ids if env_ids is not None else slice(None)] = True

    def process_actions(self, actions):
        self._raw_actions[:] = actions
        self._processed_actions[:] = actions.clamp(-1.0, 1.0) * self._scale
        pos, quat = self._compute_frame_pose()
        low = torch.tensor(WORKSPACE_LOW, device=self.device)
        high = torch.tensor(WORKSPACE_HIGH, device=self.device)
        # The controller uses robot-root coordinates; workspace constants use environment coordinates.
        low[2] -= ROBOT_BASE_HEIGHT
        high[2] -= ROBOT_BASE_HEIGHT
        desired = torch.maximum(torch.minimum(self.target_position + self._processed_actions, high), low)
        self._target_position[:] = desired
        self._needs_initialization[:] = False
        rotation = torch.tensor(DOWN_QUAT, device=self.device).expand(self.num_envs, -1)
        self._ik_controller.set_command(torch.cat([desired, rotation], dim=-1), pos, quat)
