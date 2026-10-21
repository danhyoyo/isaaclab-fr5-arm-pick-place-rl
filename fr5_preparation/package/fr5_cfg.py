"""FR5 simulation starting configuration for this machine's Isaac Lab checkout."""
from pathlib import Path
from isaaclab.assets import ArticulationCfg
from isaaclab.actuators import ImplicitActuatorCfg
import isaaclab.sim as sim_utils
from isaaclab_physx.sim.schemas import PhysxRigidBodyCfg, PhysxArticulationCfg

ROOT = Path(__file__).resolve().parent
ROBOT_USD = ROOT / "fr5_description/urdf/fr5_withgripper/fr5_arm.usda"
GRIPPER_JOINTS = ["fj1", "fj2"]
GRIPPER_OPEN = 0.0
GRIPPER_CLOSED = 0.04
FR5_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=ROBOT_USD.as_posix(),
        rigid_props=PhysxRigidBodyCfg(disable_gravity=False, max_depenetration_velocity=2.0),
        articulation_props=PhysxArticulationCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=16,
            solver_velocity_iteration_count=4,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(joint_pos={"j[1-6]": 0.0, "fj[12]": 0.0}),
    actuators={
        "arm": ImplicitActuatorCfg(joint_names_expr=["j[1-3]"], joint_effort_limit=150.0,
                                   joint_velocity_limit=3.15, stiffness=2000.0, damping=100.0),
        "wrist": ImplicitActuatorCfg(joint_names_expr=["j[4-6]"], joint_effort_limit=28.0,
                                     joint_velocity_limit=3.2, stiffness=500.0, damping=25.0),
        "gripper": ImplicitActuatorCfg(joint_names_expr=["fj[12]"], joint_effort_limit=40.0,
                                       joint_velocity_limit=0.1, stiffness=2000.0, damping=20.0),
    },
    soft_joint_pos_limit_factor=1.0,
)
