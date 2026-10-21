from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch,warp as wp
from run import register_tasks
wp.config.enable_backward=False
tasks=register_tasks()
from isaaclab.app import launch_simulation,add_launcher_args
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry
from fr5_pick_place import mdp
import argparse,gymnasium as gym
p=argparse.ArgumentParser();add_launcher_args(p);a=p.parse_args()
cfg=load_cfg_from_registry(tasks.TASK,'env_cfg_entry_point');cfg.scene.num_envs=1
cfg.scene.table=None;cfg.scene.plane=None
with launch_simulation(cfg,a):
 e=gym.make(tasks.TASK,cfg=cfg).unwrapped
 try:
  e.reset();r=e.scene['robot'];term=e.action_manager.get_term('arm')
  print('BODIES',r.body_names,'ROOT',r.data.root_pos_w.torch,'JOINTS',r.data.joint_pos.torch,flush=True)
  for n in range(3):
   tcp=mdp.geometry(e)[0];pos,quat=term._compute_frame_pose()
   print('DEBUG',n,'TCP',tcp,'TERM',pos,quat,'Q',r.data.joint_pos.torch,flush=True)
   actions=torch.tensor([[0.,0.,-0.2,1.]],device=e.device)
   term.process_actions(actions[:,:3]);pos0,quat0=term._compute_frame_pose();jp=r.data.joint_pos.torch[:,term._joint_ids]
   print('PLANNED',term._ik_controller.compute(pos0,quat0,term._compute_frame_jacobian(),jp),'RAWJ',r.data.body_link_jacobian_w.torch[:,term._jacobi_body_idx],flush=True)
   e.step(actions)
   print('JAC',term._compute_frame_jacobian(),'TARGET',term._ik_controller.ee_pos_des,flush=True)
 finally:e.close()
