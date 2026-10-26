"""Manager-based FR5 pick, lift, transfer, place and release environment."""
import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.controllers import DifferentialIKControllerCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import (ObservationGroupCfg as ObsGroup, ObservationTermCfg as ObsTerm,
                               RewardTermCfg as Reward, EventTermCfg as Event,
                               TerminationTermCfg as Done)
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass, replace
from isaaclab_physx.physics import PhysxCfg
from isaaclab_physx.sim.schemas import PhysxRigidBodyCfg
from fr5_cfg import FR5_CFG
from . import mdp
from .actions_cfg import CartesianDeltaActionCfg
from .constants import *


@configclass
class SceneCfg(InteractiveSceneCfg):
    robot = replace(FR5_CFG, prim_path="{ENV_REGEX_NS}/Robot",
                    init_state=replace(FR5_CFG.init_state, pos=(0.0, 0.0, ROBOT_BASE_HEIGHT),
                                       joint_pos={**HOME_JOINTS, "fj[12]": 0.0}),
                    actuators={name: replace(cfg, stiffness=8000.0 if name == "arm" else 2000.0,
                                             damping=200.0 if name == "arm" else (80.0 if name == "wrist" else 20.0))
                               for name, cfg in FR5_CFG.actuators.items()})
    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        init_state=AssetBaseCfg.InitialStateCfg(pos=(-0.45, 0, TABLE_TOP - 0.02)),
        spawn=sim_utils.CuboidCfg(size=(0.50, 0.70, 0.04),
            collision_props=sim_utils.UsdPhysicsCollisionCfg(collision_enabled=True),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.25, 0.29, 0.33)),
            physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=0.7, dynamic_friction=0.5, restitution=0.0)),
    )
    object = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Object",
        init_state=RigidObjectCfg.InitialStateCfg(pos=PICK_CENTER),
        spawn=sim_utils.CuboidCfg(size=(CUBE_SIZE,) * 3,
            rigid_props=PhysxRigidBodyCfg(solver_position_iteration_count=16, solver_velocity_iteration_count=4,
                                         max_depenetration_velocity=1.0),
            mass_props=sim_utils.MassPropertiesCfg(mass=CUBE_MASS),
            collision_props=sim_utils.UsdPhysicsCollisionCfg(collision_enabled=True),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.95, 0.57, 0.08)),
            physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=0.9, dynamic_friction=0.7, restitution=0.0)),
    )
    target = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Target",
        init_state=RigidObjectCfg.InitialStateCfg(pos=GOAL_CENTER),
        spawn=sim_utils.CuboidCfg(size=(0.07, 0.07, 0.001),
            rigid_props=[sim_utils.UsdPhysicsRigidBodyCfg(kinematic_enabled=True), PhysxRigidBodyCfg(disable_gravity=True)],
            collision_props=sim_utils.UsdPhysicsCollisionCfg(collision_enabled=False),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.1, 0.75, 0.30))),
    )
    plane = AssetBaseCfg(prim_path="/World/Ground", spawn=sim_utils.GroundPlaneCfg())
    light = AssetBaseCfg(prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=2500))


@configclass
class ActionsCfg:
    arm = CartesianDeltaActionCfg(
        asset_name="robot", joint_names=["j[1-6]"], body_name="hand_base_link",
        body_offset=CartesianDeltaActionCfg.OffsetCfg(pos=TCP_OFFSET), scale=ACTION_STEP,
        controller=DifferentialIKControllerCfg(command_type="pose", use_relative_mode=False, ik_method="dls",
                                               ik_params={"lambda_val": 0.03}),
    )
    gripper = mdp.BinaryJointPositionActionCfg(asset_name="robot", joint_names=["fj[12]"],
                 open_command_expr={"fj[12]": 0.0}, close_command_expr={"fj[12]": 0.04})


@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, scale=0.1)
        task = ObsTerm(func=mdp.task_observation)
        previous_action = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True
    policy: PolicyCfg = PolicyCfg()


@configclass
class EventsCfg:
    defaults = Event(func=mdp.reset_scene_to_default, mode="reset", params={"reset_joint_targets": True})
    task = Event(func=mdp.reset_task, mode="reset", params={"randomize": True})


@configclass
class RewardsCfg:
    reach = Reward(func=mdp.task_reward, params={"stage": "reach"}, weight=1.0)
    grasp = Reward(func=mdp.task_reward, params={"stage": "grasp"}, weight=1.5)
    lift = Reward(func=mdp.task_reward, params={"stage": "lift"}, weight=5.0)
    transport = Reward(func=mdp.task_reward, params={"stage": "transport"}, weight=8.0)
    lower = Reward(func=mdp.task_reward, params={"stage": "lower"}, weight=8.0)
    release = Reward(func=mdp.task_reward, params={"stage": "release"}, weight=10.0)
    withdraw = Reward(func=mdp.task_reward, params={"stage": "withdraw"}, weight=10.0)
    success = Reward(func=mdp.task_reward, params={"stage": "success"}, weight=5000.0)
    time_penalty = Reward(func=mdp.task_reward, params={"stage": "time"}, weight=-0.25)
    action_rate = Reward(func=mdp.action_rate_l2, weight=-0.005)
    joint_velocity = Reward(func=mdp.joint_vel_l2, weight=-0.0001)


@configclass
class TerminationsCfg:
    time_out = Done(func=mdp.time_out, time_out=True)
    success = Done(func=mdp.success)
    dropped = Done(func=mdp.object_out_of_workspace)


@configclass
class FR5PickPlaceEnvCfg(ManagerBasedRLEnvCfg):
    scene: SceneCfg = SceneCfg(num_envs=128, env_spacing=2.5, replicate_physics=True)
    actions: ActionsCfg = ActionsCfg()
    observations: ObservationsCfg = ObservationsCfg()
    events: EventsCfg = EventsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    def __post_init__(self):
        self.decimation = 4
        self.episode_length_s = 12.0
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.sim.physics = PhysxCfg(bounce_threshold_velocity=0.01,
                                    gpu_found_lost_aggregate_pairs_capacity=4 * 1024 * 1024)
        self.viewer.eye = (-1.3, -1.2, 1.0)
        self.viewer.lookat = (-0.45, 0, 0.25)


@configclass
class FR5PickPlacePlayEnvCfg(FR5PickPlaceEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 4
