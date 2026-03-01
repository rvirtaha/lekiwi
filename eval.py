#!/usr/bin/env python

import os
import time

import numpy as np
from dotenv import load_dotenv
import torch

from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.utils import prepare_observation_for_inference, make_robot_action
from lerobot.robots.lekiwi import LeKiwiClient, LeKiwiClientConfig
from lerobot.utils.robot_utils import precise_sleep
from lerobot.utils.visualization_utils import init_rerun, log_rerun_data

load_dotenv()

FPS = 30
POLICY_PATH = os.environ.get("POLICY_PATH", "models/cigarette_pickup")
DEVICE = os.environ.get("DEVICE", "cuda")
TASK = os.environ.get("TASK", "move to the white candy and pick it up")

# SmolVLA was trained with camera1/camera2 names
CAMERA_MAP = {"front": "camera1", "wrist": "camera2"}
ACTION_KEYS = [
    "arm_shoulder_pan.pos", "arm_shoulder_lift.pos", "arm_elbow_flex.pos",
    "arm_wrist_flex.pos", "arm_wrist_roll.pos", "arm_gripper.pos",
    "x.vel", "y.vel", "theta.vel",
]
ACTION_FEATURES = {"action": {"names": ACTION_KEYS}}


def fix_camera_orientations(observation):
    for key, val in observation.items():
        if isinstance(val, np.ndarray) and val.ndim == 3:
            if "wrist" in key:
                observation[key] = np.rot90(val, k=1)
            elif "birdseye" not in key:
                observation[key] = val[::-1, ::-1]
    return observation


def extract_policy_obs(observation):
    """Extract and rename keys for SmolVLA (front→camera1, wrist→camera2)."""
    return {
        f"observation.images.{CAMERA_MAP[k]}": np.ascontiguousarray(observation[k])
        for k in CAMERA_MAP if k in observation
    } | {
        "observation.state": observation["observation.state"],
    }


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

    robot = LeKiwiClient(robot_config)

    print(f"Loading policy from {POLICY_PATH} on {DEVICE}...")
    policy = SmolVLAPolicy.from_pretrained(POLICY_PATH,device=DEVICE,dtype=torch.bfloat16)
    policy.eval()
    policy.to(DEVICE)

    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy.config,
        pretrained_path=POLICY_PATH,
    )
    print("Policy loaded.")

    robot.connect()

    init_rerun(session_name="lekiwi_eval")

    if not robot.is_connected:
        raise ValueError("Robot is not connected!")

    print(f"Running eval with task: '{TASK}'")

    step = 0
    while True:
        t0 = time.perf_counter()

        observation = robot.get_observation()
        fix_camera_orientations(observation)

        policy_obs = extract_policy_obs(observation)
        policy_obs = prepare_observation_for_inference(policy_obs, DEVICE, task=TASK)

        t_inf = time.perf_counter()
        with torch.inference_mode():
            processed_obs = preprocessor(policy_obs)
            raw_action = policy.select_action(processed_obs)
            action = postprocessor(raw_action)
        inf_time = time.perf_counter() - t_inf

        robot_action = make_robot_action(action, ACTION_FEATURES)

        if step % 30 == 0:
            raw = raw_action[0].cpu().numpy() if raw_action.ndim == 2 else raw_action.cpu().numpy()
            base = [f"{robot_action[k]:+.3f}" for k in ACTION_KEYS[6:]]
            raw_base = [f"{raw[i]:+.3f}" for i in range(6, min(9, len(raw)))]
            print(f"[step {step}] inf={inf_time:.3f}s base_raw={raw_base} base_unnorm={base}")

        # Keyboard override for base
        robot.send_action(robot_action)

        #log_rerun_data(observation=observation, action=robot_action)

        dt = time.perf_counter() - t0
        precise_sleep(max(1.0 / FPS - dt, 0.0))
        step += 1


if __name__ == "__main__":
    main()
