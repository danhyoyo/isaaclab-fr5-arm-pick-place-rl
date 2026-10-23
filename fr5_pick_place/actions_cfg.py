"""Keep action implementation lazy so USD/Kit modules load after simulator startup."""
from isaaclab.envs.mdp.actions.actions_cfg import DifferentialInverseKinematicsActionCfg
from isaaclab.utils import configclass


@configclass
class CartesianDeltaActionCfg(DifferentialInverseKinematicsActionCfg):
    class_type: str = "fr5_pick_place.actions:CartesianDeltaAction"
