1. Collect Demonstrations (Teleoperation)

First, run lekiwi_host on the Jetson, then record episodes from the laptop:

# Jetson: start the robot server

python -m lerobot.robots.lekiwi.lekiwi_host

# Laptop: record teleoperation episodes into a dataset

python -m lerobot record \
 --robot.type=lekiwi \
 --dataset.repo_id=<your-hf-username>/<dataset-name> \
 --dataset.num_episodes=50

You physically teleoperate the leader arm while the system records observations
(camera frames, joint positions) and actions.

2. Train a Policy

lerobot supports several imitation learning policies. The two main ones:

# ACT (Action Chunking with Transformers) — good default

python -m lerobot train \
 --policy.type=act \
 --dataset.repo_id=<your-hf-username>/<dataset-name>

# Or Diffusion Policy

python -m lerobot train \
 --policy.type=diffusion \
 --dataset.repo_id=<your-hf-username>/<dataset-name>

You already have wandb in your dependencies for tracking training runs. Your
pyproject.toml also has the smolvla optional — that's a Vision-Language-Action model
if you want to try that route later.

3. Evaluate

Run the trained policy on the real robot:

python -m lerobot eval \
 --policy.path=outputs/train/<run-name>/checkpoints/last/pretrained_model \
 --robot.type=lekiwi
