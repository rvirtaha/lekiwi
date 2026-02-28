#!/usr/bin/env python

import os
import time

import numpy as np
import torch
from dotenv import load_dotenv

from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.policies.factory import make_pre_post_processors
from lerobot.robots.lekiwi import LeKiwiClient, LeKiwiClientConfig
from lerobot.teleoperators.keyboard.teleop_keyboard import KeyboardTeleop, KeyboardTeleopConfig
from lerobot.utils.robot_utils import precise_sleep
from lerobot.utils.visualization_utils import init_rerun, log_rerun_data

load_dotenv()

FPS = 30
POLICY_PATH = os.environ.get("POLICY_PATH", "models/cigarette_pickup")
DEVICE = os.environ.get("DEVICE", "cuda")
TASK = os.environ.get("TASK", "move to the white candy and pick it up")

# Trained model uses camera1/camera2 names, action dim=9 (6 arm + 3 base)
CAMERA_MAP = {"front": "camera1", "wrist": "camera2"}
ACTION_KEYS = [
    "arm_shoulder_pan.pos", "arm_shoulder_lift.pos", "arm_elbow_flex.pos",
    "arm_wrist_flex.pos", "arm_wrist_roll.pos", "arm_gripper.pos",
    "x.vel", "y.vel", "theta.vel",
]


def fix_camera_orientations(observation):
    for key, val in observation.items():
        if isinstance(val, np.ndarray) and val.ndim == 3:
            if "wrist" in key:
                observation[key] = np.rot90(val, k=1)
            elif "birdseye" not in key:
                observation[key] = val[::-1, ::-1]
    return observation


def robot_obs_to_policy_obs(observation, task):
    """Convert robot observation to SmolVLA input format."""
    policy_obs = {}

    # Map camera images to SmolVLA's expected keys
    for robot_key, policy_key in CAMERA_MAP.items():
        if robot_key in observation:
            img = observation[robot_key]
            img = torch.from_numpy(img.copy()).permute(2, 0, 1).unsqueeze(0).float() / 255.0
            policy_obs[f"observation.images.{policy_key}"] = img

    # State: all 9 dims (6 arm + 3 base)
    if "observation.state" in observation:
        state = observation["observation.state"]
        policy_obs["observation.state"] = torch.from_numpy(state).unsqueeze(0).float()

    policy_obs["task"] = task
    return policy_obs


def policy_action_to_robot_action(action):
    """Convert SmolVLA action tensor (dim=9) to robot action dict."""
    if action.ndim == 2:
        action = action[0]
    action = action.cpu().numpy()

    robot_action = {}
    for i, key in enumerate(ACTION_KEYS):
        if i < len(action):
            robot_action[key] = float(action[i])

    return robot_action


def main():
    robot_config = LeKiwiClientConfig(
        remote_ip=os.environ["JETSON_IP"],
        id=os.environ["ROBOT_ID"],
        teleop_keys={
            "forward": "w",
            "backward": "s",
            "left": "a",
            "right": "d",
            "rotate_left": "q",
            "rotate_right": "e",
            "speed_up": "r",
            "speed_down": "f",
            "quit": "z",
        },
    )
    keyboard_config = KeyboardTeleopConfig(id="my_laptop_keyboard")

    robot = LeKiwiClient(robot_config)
    keyboard = KeyboardTeleop(keyboard_config)

    # Load policy directly (bypass make_policy which requires ds_meta)
    print(f"Loading policy from {POLICY_PATH} on {DEVICE}...")
    policy = SmolVLAPolicy.from_pretrained(POLICY_PATH)
    policy.eval()
    policy.to(DEVICE)

    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy.config,
        pretrained_path=POLICY_PATH,
    )
    print("Policy loaded.")

    robot.connect()
    keyboard.connect()

    init_rerun(session_name="lekiwi_eval")

    if not robot.is_connected or not keyboard.is_connected:
        raise ValueError("Robot or keyboard is not connected!")

    print(f"Running eval with task: '{TASK}'")
    print("Use keyboard to override base movement or Ctrl+C to stop.")

    step = 0
    while True:
        t0 = time.perf_counter()

        observation = robot.get_observation()
        fix_camera_orientations(observation)

        # Policy inference
        policy_obs = robot_obs_to_policy_obs(observation, TASK)

        t_inf = time.perf_counter()
        with torch.inference_mode():
            processed_obs = preprocessor(policy_obs)
            raw_action = policy.select_action(processed_obs)
            action = postprocessor(raw_action)
        inf_time = time.perf_counter() - t_inf

        robot_action = policy_action_to_robot_action(action)

        # Debug: print raw (normalized) and unnormalized actions every 30 steps
        if step % 30 == 0:
            raw = raw_action[0].cpu().numpy() if raw_action.ndim == 2 else raw_action.cpu().numpy()
            arm = [f"{robot_action[k]:+.3f}" for k in ACTION_KEYS[:6]]
            base = [f"{robot_action[k]:+.3f}" for k in ACTION_KEYS[6:]]
            raw_base = [f"{raw[i]:+.3f}" for i in range(6, min(9, len(raw)))]
            print(f"[step {step}] inf={inf_time:.3f}s base_raw={raw_base} base_unnorm={base}")

        # Keyboard override for base (and e-stop via Ctrl+C)
        keyboard_keys = keyboard.get_action()
        base_override = robot._from_keyboard_to_base_action(keyboard_keys)
        if len(base_override) > 0:
            robot_action.update(base_override)

        robot.send_action(robot_action)

        log_rerun_data(observation=observation, action=robot_action)

        dt = time.perf_counter() - t0
        precise_sleep(max(1.0 / FPS - dt, 0.0))
        step += 1


if __name__ == "__main__":
    main()
