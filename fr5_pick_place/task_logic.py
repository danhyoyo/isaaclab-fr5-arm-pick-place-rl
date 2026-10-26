"""Simulator-independent success logic and per-environment episode memory."""
import torch


def placement_candidate(lifted, object_pos, goal_pos, object_velocity, fingers, tcp_distance):
    xy = torch.linalg.vector_norm(object_pos[:, :2] - goal_pos[:, :2], dim=-1)
    return (lifted & (xy < 0.020) & ((object_pos[:, 2] - goal_pos[:, 2]).abs() < 0.008)
            & (torch.linalg.vector_norm(object_velocity, dim=-1) < 0.06)
            & (fingers.abs().amax(dim=-1) < 0.004) & (tcp_distance > 0.055))


def controlled_placement_candidate(lifted, object_pos, goal_pos, object_velocity, fingers,
                                   tcp_distance, xy_tol=0.025, z_tol=0.012, max_speed=0.08):
    """Require a slow, still-held object before latching the placement milestone."""
    xy = torch.linalg.vector_norm(object_pos[:, :2] - goal_pos[:, :2], dim=-1)
    speed = torch.linalg.vector_norm(object_velocity, dim=-1)
    held = (fingers.mean(-1) > 0.012) & (tcp_distance < 0.045)
    return (lifted & held & (xy < xy_tol)
            & ((object_pos[:, 2] - goal_pos[:, 2]).abs() < z_tol)
            & (speed < max_speed))


def completed_placement_candidate(placed_once, released_once, lifted, object_pos, goal_pos,
                                  object_velocity, fingers, tcp_distance):
    """Only a controlled place-and-release sequence may become successful."""
    return (placed_once & released_once
            & placement_candidate(lifted, object_pos, goal_pos, object_velocity,
                                  fingers, tcp_distance))


def staged_rewards(object_pos, goal_pos, fingers, tcp_distance, lifted, rest_height,
                   placed_once=None, released_once=None, just_grasped=None,
                   just_lifted=None, just_placed=None, just_released=None,
                   reach_gain=None, lift_gain=None, transport_gain=None,
                   lower_xy_gain=None, lower_z_gain=None, opening_gain=None,
                   withdraw_gain=None, step_dt=0.02,
                   release_xy_tol=0.025, release_z_tol=0.012):
    """Return mutually exclusive shaping rewards for the pick-place stages.

    Late-stage shaping is based on one-time milestones and new-best progress.
    It therefore cannot be accumulated by hovering or repeating a motion.
    """
    xy_goal = torch.linalg.vector_norm(object_pos[:, :2] - goal_pos[:, :2], dim=-1)
    z_error = (object_pos[:, 2] - rest_height).abs()
    release_zone = lifted & (xy_goal < release_xy_tol) & (z_error < release_z_tol)
    zeros_bool = torch.zeros_like(lifted)
    zeros_float = torch.zeros_like(tcp_distance)
    placed_once = release_zone if placed_once is None else placed_once
    released_once = zeros_bool if released_once is None else released_once
    just_grasped = zeros_bool if just_grasped is None else just_grasped
    just_lifted = zeros_bool if just_lifted is None else just_lifted
    just_placed = zeros_bool if just_placed is None else just_placed
    just_released = zeros_bool if just_released is None else just_released
    reach_gain = zeros_float if reach_gain is None else reach_gain
    lift_gain = zeros_float if lift_gain is None else lift_gain
    transport_gain = zeros_float if transport_gain is None else transport_gain
    lower_xy_gain = zeros_float if lower_xy_gain is None else lower_xy_gain
    lower_z_gain = zeros_float if lower_z_gain is None else lower_z_gain
    opening_gain = zeros_float if opening_gain is None else opening_gain
    withdraw_gain = zeros_float if withdraw_gain is None else withdraw_gain
    inv_dt = 1.0 / max(float(step_dt), 1.0e-6)

    return {
        "reach": 2.0 * reach_gain * inv_dt,
        "grasp": 2.0 * just_grasped.float() * inv_dt,
        "lift": (2.0 * lift_gain * inv_dt + 2.0 * just_lifted.float() * inv_dt),
        # These stages pay only for a new best progress value.  Dividing by dt
        # compensates for RewardManager's dt multiplication, so their total
        # episode value is independent of control frequency and standing still
        # always pays zero.
        "transport": 2.0 * transport_gain * inv_dt,
        # XY alignment and vertical descent have independent best values.  A
        # small XY regression can therefore no longer cancel real Z progress.
        # Z is weighted more strongly because it is the final precision skill.
        "lower_xy": 1.0 * lower_xy_gain * inv_dt,
        "lower_z": 3.0 * lower_z_gain * inv_dt,
        # Retained for callers that want the combined shaping value.  The
        # environment logs lower_xy and lower_z separately.
        "lower": (1.0 * lower_xy_gain + 3.0 * lower_z_gain) * inv_dt,
        # Milestones are one-shot bonuses.  Opening pays only for previously
        # unseen physical opening, never merely for repeating an open command.
        # Waiting in the release zone has a small per-second cost.
        "release": (3.0 * just_placed.float() * inv_dt
                    + 4.0 * opening_gain * inv_dt
                    + 5.0 * just_released.float() * inv_dt
                    - 0.10 * (placed_once & release_zone & ~released_once).float()),
        # Likewise, only additional physical separation after release is paid.
        "withdraw": 3.0 * withdraw_gain * inv_dt,
    }


class EpisodeProgress:
    def __init__(self, num_envs, device):
        self.lifted = torch.zeros(num_envs, dtype=torch.bool, device=device)
        self.grasped = torch.zeros(num_envs, dtype=torch.bool, device=device)
        self.placed = torch.zeros(num_envs, dtype=torch.bool, device=device)
        self.released = torch.zeros(num_envs, dtype=torch.bool, device=device)
        self.just_grasped = torch.zeros(num_envs, dtype=torch.bool, device=device)
        self.just_lifted = torch.zeros(num_envs, dtype=torch.bool, device=device)
        self.just_placed = torch.zeros(num_envs, dtype=torch.bool, device=device)
        self.just_released = torch.zeros(num_envs, dtype=torch.bool, device=device)
        self.opening = torch.zeros(num_envs, device=device)
        self.opening_delta = torch.zeros(num_envs, device=device)
        self.reach_best = torch.zeros(num_envs, device=device)
        self.reach_gain = torch.zeros(num_envs, device=device)
        self.lift_best = torch.zeros(num_envs, device=device)
        self.lift_gain = torch.zeros(num_envs, device=device)
        self.transport_best = torch.zeros(num_envs, device=device)
        self.transport_gain = torch.zeros(num_envs, device=device)
        self.lower_xy_best = torch.zeros(num_envs, device=device)
        self.lower_xy_gain = torch.zeros(num_envs, device=device)
        self.lower_z_started = torch.zeros(num_envs, dtype=torch.bool, device=device)
        self.lower_z_start_error = torch.zeros(num_envs, device=device)
        self.lower_z_best = torch.zeros(num_envs, device=device)
        self.lower_z_gain = torch.zeros(num_envs, device=device)
        self.opening_best = torch.zeros(num_envs, device=device)
        self.opening_gain = torch.zeros(num_envs, device=device)
        self.withdraw_best = torch.zeros(num_envs, device=device)
        self.withdraw_gain = torch.zeros(num_envs, device=device)
        self.hold = torch.zeros(num_envs, device=device)
        self.last_step = torch.full((num_envs,), -1, dtype=torch.long, device=device)

    def reset(self, ids, step):
        self.lifted[ids] = False
        self.grasped[ids] = False
        self.placed[ids] = False
        self.released[ids] = False
        self.just_grasped[ids] = False
        self.just_lifted[ids] = False
        self.just_placed[ids] = False
        self.just_released[ids] = False
        self.opening[ids] = 0
        self.opening_delta[ids] = 0
        self.reach_best[ids] = 0
        self.reach_gain[ids] = 0
        self.lift_best[ids] = 0
        self.lift_gain[ids] = 0
        self.transport_best[ids] = 0
        self.transport_gain[ids] = 0
        self.lower_xy_best[ids] = 0
        self.lower_xy_gain[ids] = 0
        self.lower_z_started[ids] = False
        self.lower_z_start_error[ids] = 0
        self.lower_z_best[ids] = 0
        self.lower_z_gain[ids] = 0
        self.opening_best[ids] = 0
        self.opening_gain[ids] = 0
        self.withdraw_best[ids] = 0
        self.withdraw_gain[ids] = 0
        self.hold[ids] = 0
        self.last_step[ids] = step

    def update(self, step, dt, lifting_now, candidate_fn, grasping_now=None,
               placement_now=None, release_now=None, opening=None, reach_score=None,
               reach_active=None, lift_score=None, lift_active=None, transport_score=None,
               transport_active=None, lower_xy_score=None, lower_xy_active=None,
               lower_z_error=None, lower_z_active=None, lower_z_target_error=0.0,
               opening_active=None, withdraw_score=None, withdraw_active=None):
        fresh = self.last_step != step
        was_lifted = self.lifted.clone()
        was_released = self.released.clone()
        self.lifted |= lifting_now & fresh
        self.just_lifted[fresh] = lifting_now[fresh] & ~was_lifted[fresh]
        if grasping_now is not None:
            self.just_grasped[fresh] = grasping_now[fresh] & ~self.grasped[fresh]
            self.grasped[fresh] |= grasping_now[fresh]
        else:
            self.just_grasped[fresh] = False
        if placement_now is not None:
            self.just_placed[fresh] = placement_now[fresh] & ~self.placed[fresh]
            self.placed[fresh] |= placement_now[fresh]
        else:
            self.just_placed[fresh] = False
        if opening is not None:
            self.opening_delta[fresh] = opening[fresh] - self.opening[fresh]
            self.opening[fresh] = opening[fresh]
        else:
            self.opening_delta[fresh] = 0
        if release_now is not None:
            self.just_released[fresh] = release_now[fresh] & self.placed[fresh] & ~self.released[fresh]
            self.released[fresh] |= release_now[fresh] & self.placed[fresh]
        else:
            self.just_released[fresh] = False

        def update_best(score, active, best, gain):
            gain[fresh] = 0
            if score is None or active is None:
                return
            eligible = fresh & active
            gain[eligible] = (score[eligible] - best[eligible]).clamp(min=0)
            best[eligible] = torch.maximum(best[eligible], score[eligible])

        update_best(reach_score, reach_active, self.reach_best, self.reach_gain)
        update_best(lift_score, lift_active, self.lift_best, self.lift_gain)
        update_best(transport_score, transport_active, self.transport_best, self.transport_gain)
        update_best(lower_xy_score, lower_xy_active, self.lower_xy_best, self.lower_xy_gain)
        # Start vertical progress at the height where the held object first
        # enters the precise XY zone.  This removes the fixed-height dead zone:
        # descent from any approach height now has a useful gradient.  The
        # entry step only records a baseline and deliberately pays zero.
        self.lower_z_gain[fresh] = 0
        if lower_z_error is not None and lower_z_active is not None:
            eligible = fresh & lower_z_active
            starting = eligible & ~self.lower_z_started
            self.lower_z_start_error[starting] = lower_z_error[starting]
            self.lower_z_started[starting] = True
            continuing = eligible & ~starting
            target = torch.as_tensor(lower_z_target_error, device=lower_z_error.device,
                                     dtype=lower_z_error.dtype)
            # If the first eligible sample is already inside the placement
            # height tolerance, descent is complete; do not create a tiny
            # denominator that could turn a millimetre into a full bonus.
            self.lower_z_best[starting & (lower_z_error <= target)] = 1.0
            span = (self.lower_z_start_error - target).clamp(min=1.0e-6)
            lower_z_score = ((self.lower_z_start_error - lower_z_error) / span).clamp(0, 1)
            self.lower_z_gain[continuing] = (
                lower_z_score[continuing] - self.lower_z_best[continuing]).clamp(min=0)
            self.lower_z_best[continuing] = torch.maximum(
                self.lower_z_best[continuing], lower_z_score[continuing])
        update_best(opening, opening_active, self.opening_best, self.opening_gain)
        update_best(withdraw_score, withdraw_active, self.withdraw_best, self.withdraw_gain)
        # The release pose is the baseline for withdrawal.  It must not count as
        # progress before the TCP has actually moved away on a later step.
        newly_released = fresh & ~was_released & self.released
        if withdraw_score is not None:
            self.withdraw_best[newly_released] = withdraw_score[newly_released]
            self.withdraw_gain[newly_released] = 0
        candidate = candidate_fn(self.lifted)
        self.hold[fresh] = torch.where(candidate[fresh], self.hold[fresh] + dt, 0.0)
        self.last_step[fresh] = step
