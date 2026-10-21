# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.assets import RigidObjectCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.sensors import FrameTransformerCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from isaaclab_physx.sim.schemas import PhysxRigidBodyCfg, PhysxArticulationCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from isaaclab.utils import configclass, clone, replace
import isaaclab.sim as sim_utils

from .. import mdp
from ..lift_place_env_cfg import LiftEnvCfg

from pathlib import Path
import math

PROJECT_ROOT = Path(__file__).resolve().parents[1]
GRIPPER_OPEN_ANGLE = 0.0
GRIPPER_CLOSED_ANGLE = math.radians(-60.0)

from isaaclab.markers.config import FRAME_MARKER_CFG  # isort: skip

##
# Configuration of the arm
##

CUSTOM_ARM_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=(PROJECT_ROOT / "custom_arm/cumotion_arm/cumotion_arm_gripper_fixed.usda").as_posix(),
        activate_contact_sensors=False,
        rigid_props=PhysxRigidBodyCfg(
            disable_gravity=False,
            max_depenetration_velocity=5.0,
        ),
        articulation_props=PhysxArticulationCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=0
        ),
        # collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.005, rest_offset=0.0),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        joint_pos={
            "cumotion_arm_joint1": 0.0,
            "cumotion_arm_joint2": 0.0,
            "cumotion_arm_joint3": 0.0,
            "cumotion_arm_joint4": 0.0,
            "cumotion_arm_joint5": 0.0,
            "cumotion_arm_joint6": 0.0,
            "cumotion_arm_joint_hand": 0.0,
        },
    ),
    actuators={
        "arm_actuator": ImplicitActuatorCfg(
            joint_names_expr=["cumotion_arm_joint[1-6]"],
            joint_effort_limit=87.0,
            stiffness=80.0,
            damping=4.0,
        ),
        "hand_actuator": ImplicitActuatorCfg(
            joint_names_expr=["cumotion_arm_joint_hand"],
            joint_effort_limit=80.0,
            stiffness=80,
            damping=4.0,
        ),
    },
    soft_joint_pos_limit_factor=1.0,
)
"""Configuration of the custom arm."""

@configclass
class CustomArmCubeLiftPlaceEnvCfg(LiftEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # Set the custom arm as robot.
        self.scene.robot = replace(CUSTOM_ARM_CFG, prim_path="{ENV_REGEX_NS}/Robot")

        # Configure joint position control and the binary gripper action.
        self.actions.arm_action = mdp.JointPositionActionCfg(
            asset_name="robot",
            joint_names=["cumotion_arm_joint[1-6]"],
            scale=0.5,
            use_default_offset=True
        )
        self.actions.gripper_action = mdp.BinaryJointPositionActionCfg(
            asset_name="robot",
            joint_names=["cumotion_arm_joint_hand"],
            open_command_expr={"cumotion_arm_joint_hand": GRIPPER_OPEN_ANGLE},
            # Geometry verified: +60 widens the jaw; -60 moves toward the fixed
            # finger. The USD overlay corrects the original limits to [-60, 0].
            close_command_expr={"cumotion_arm_joint_hand": GRIPPER_CLOSED_ANGLE},
        )
        # Set the body name for the end effector
        #self.commands.object_pose.body_name = "panda_hand"

        # Set Cube as object
        self.scene.object = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Object",
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=[0.1, 0, 0.018], # [0.1, 0, 0.055]
                rot=[0, 0, 0, 1]
            ),
            spawn=UsdFileCfg(
                usd_path=(PROJECT_ROOT / "cube_5cm/cube_5cm.usd").as_posix(),
                scale=(0.72, 0.72, 0.72),
                rigid_props=PhysxRigidBodyCfg(
                    solver_position_iteration_count=16,
                    solver_velocity_iteration_count=1,
                    max_angular_velocity=1000.0,
                    max_linear_velocity=1000.0,
                    max_depenetration_velocity=5.0,
                    disable_gravity=False,
                ),
            ),
        )
        
        # Listens to the required transforms
        marker_cfg = clone(FRAME_MARKER_CFG)
        marker_cfg.markers["frame"].scale = (0.05, 0.05, 0.05)
        marker_cfg.prim_path = "/Visuals/FrameTransformer"
        self.scene.ee_frame = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/Robot/cumotion_arm_base",
            debug_vis=False,
            visualizer_cfg=marker_cfg,
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/cumotion_arm_link_hand",
                    name="end_effector",
                    offset=OffsetCfg(
                        pos=[0.0, 0.0, 0.037],
                    ),
                ),
            ],
        )

        self.scene.target_region = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/TargetRegion",
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=[0.1, 0.0, 0.001],
                rot=[0, 0, 0, 1],
            ),
            spawn=sim_utils.CuboidCfg(
                size=(0.036, 0.036, 0.01),
                rigid_props=[
                    sim_utils.UsdPhysicsRigidBodyCfg(kinematic_enabled=True),
                    PhysxRigidBodyCfg(disable_gravity=True),
                ],
                visual_material=sim_utils.PreviewSurfaceCfg(
                    diffuse_color=(1.0, 0.0, 0.0),
                ),
            ),
        )


@configclass
class CustomArmCubeLiftPlaceEnvCfg_PLAY(CustomArmCubeLiftPlaceEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # disable randomization for play
        self.observations.policy.enable_corruption = False
