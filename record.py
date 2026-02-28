# !/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import os

import numpy as np
from dotenv import load_dotenv

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.utils import hw_to_dataset_features
from lerobot.processor import make_default_processors
from lerobot.robots.lekiwi.config_lekiwi import LeKiwiClientConfig
from lerobot.robots.lekiwi.lekiwi_client import LeKiwiClient
from lerobot.scripts.lerobot_record import record_loop
from lerobot.teleoperators.keyboard import KeyboardTeleop, KeyboardTeleopConfig
from lerobot.teleoperators.so_leader import SO100Leader, SO100LeaderConfig
from lerobot.utils.constants import ACTION, OBS_STR
from lerobot.utils.control_utils import init_keyboard_listener
from lerobot.utils.utils import log_say
from lerobot.utils.visualization_utils import init_rerun

load_dotenv()

FPS = 30
RESET_TIME_SEC = 10
HF_REPO_ID = "rvirtaha/candy-pickup"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, required=True, help="Task description for this recording session")
    parser.add_argument("--num-episodes", type=int, default=20)
    parser.add_argument("--episode-time", type=int, default=30)
    args = parser.parse_args()

    # Create the robot and teleoperator configurations
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
    leader_arm_config = SO100LeaderConfig(port=os.environ["LEADER_PORT"], id=os.environ["LEADER_ID"])
    keyboard_config = KeyboardTeleopConfig(id="my_laptop_keyboard")

    # Initialize the robot and teleoperator
    robot = LeKiwiClient(robot_config)
    leader_arm = SO100Leader(leader_arm_config)
    keyboard = KeyboardTeleop(keyboard_config)

    # Fix camera orientations (wrist: 90° CCW, others: 180°)
    _original_get_observation = robot.get_observation

    def get_observation_fixed():
        obs = _original_get_observation()
        for key, val in obs.items():
            if isinstance(val, np.ndarray) and val.ndim == 3:
                if "wrist" in key:
                    obs[key] = np.rot90(val, k=1)
                else:
                    obs[key] = val[::-1, ::-1]
        return obs

    robot.get_observation = get_observation_fixed

    # Patch observation_features to reflect rotated wrist shape
    _original_obs_features = robot.observation_features
    for key, shape in _original_obs_features.items():
        if "wrist" in key and isinstance(shape, tuple) and len(shape) == 3:
            h, w, c = shape
            _original_obs_features[key] = (w, h, c)
    robot.observation_features = _original_obs_features

    # TODO(Steven): Update this example to use pipelines
    teleop_action_processor, robot_action_processor, robot_observation_processor = make_default_processors()

    # Configure the dataset features
    action_features = hw_to_dataset_features(robot.action_features, ACTION)
    obs_features = hw_to_dataset_features(robot.observation_features, OBS_STR)
    dataset_features = {**action_features, **obs_features}

    # Load existing dataset or create a new one
    try:
        dataset = LeRobotDataset(
            repo_id=HF_REPO_ID,
        )
        dataset.start_image_writer(num_processes=0, num_threads=4)
        print(f"Resuming dataset with {dataset.num_episodes} existing episodes")
    except FileNotFoundError:
        dataset = LeRobotDataset.create(
            repo_id=HF_REPO_ID,
            fps=FPS,
            features=dataset_features,
            robot_type=robot.name,
            use_videos=True,
            image_writer_threads=4,
        )
        print("Created new dataset")

    # Connect the robot and teleoperator
    # To connect you already should have this script running on LeKiwi: `python -m lerobot.robots.lekiwi.lekiwi_host --robot.id=my_awesome_kiwi`
    robot.connect()
    leader_arm.connect()
    keyboard.connect()

    # Initialize the keyboard listener and rerun visualization
    listener, events = init_keyboard_listener()
    init_rerun(session_name="lekiwi_record")

    try:
        if not robot.is_connected or not leader_arm.is_connected or not keyboard.is_connected:
            raise ValueError("Robot or teleop is not connected!")

        print("Starting record loop...")
        recorded_episodes = 0
        while recorded_episodes < args.num_episodes and not events["stop_recording"]:
            log_say(f"Recording episode {recorded_episodes}")

            # Main record loop
            record_loop(
                robot=robot,
                events=events,
                fps=FPS,
                dataset=dataset,
                teleop=[leader_arm, keyboard],
                control_time_s=args.episode_time,
                single_task=args.task,
                display_data=True,
                teleop_action_processor=teleop_action_processor,
                robot_action_processor=robot_action_processor,
                robot_observation_processor=robot_observation_processor,
            )

            # Reset the environment if not stopping or re-recording
            if not events["stop_recording"] and (
                (recorded_episodes < args.num_episodes - 1) or events["rerecord_episode"]
            ):
                log_say("Reset the environment")
                record_loop(
                    robot=robot,
                    events=events,
                    fps=FPS,
                    teleop=[leader_arm, keyboard],
                    control_time_s=RESET_TIME_SEC,
                    single_task=args.task,
                    display_data=True,
                    teleop_action_processor=teleop_action_processor,
                    robot_action_processor=robot_action_processor,
                    robot_observation_processor=robot_observation_processor,
                )

            if events["rerecord_episode"]:
                log_say("Re-record episode")
                events["rerecord_episode"] = False
                events["exit_early"] = False
                dataset.clear_episode_buffer()
                continue

            # Save episode
            dataset.save_episode()
            recorded_episodes += 1
            events["exit_early"] = False
    finally:
        # Clean up
        log_say("Stop recording")
        robot.disconnect()
        leader_arm.disconnect()
        keyboard.disconnect()
        listener.stop()

        dataset.finalize()
        dataset.push_to_hub()


if __name__ == "__main__":
    main()