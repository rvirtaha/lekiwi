# Episode Recording Plan

Train a SmolVLA policy to pick up a white candy from the floor and place it on the box mounted on the robot's back.

## Task Breakdown

Record 20 episodes each, ordered from simple to complex:

| #   | Task description                                  | What to demonstrate                                                                                          |
| --- | ------------------------------------------------- | ------------------------------------------------------------------------------------------------------------ |
| 1   | `"move to the white candy"`                       | Drive base toward candy, stop near it. No arm movement.                                                      |
| 2   | `"pick up the white candy"`                       | Start near candy. Reach down, grasp, lift. No driving.                                                       |
| 3   | `"drop candy into box"`                           | Candy already in gripper. Move arm back and place candy on the box on the robot's back, release. No driving. |
| 4   | `"move to the white candy and pick it up"`        | Drive to candy + pick it up. Combines 1 & 2.                                                                 |
| 5   | `"pick up the white candy and put it on the box"` | Full task: navigate to candy, pick up, place on the box on the robot's back.                                 |

**Total: ~100 episodes** (5 tasks x 20 episodes)

## Tips

- Vary starting position/orientation between episodes for generalization.
- Vary candy placement on the floor.
- Default episode time is 30s. For task 5, consider `--episode-time 45`.

## Commands

```bash
# On Jetson first:
python -m lerobot.robots.lekiwi.lekiwi_host

# Then on laptop:
python record.py --task "pick up the white candy"
python record.py --task "move to the white candy"
python record.py --task "drop candy into box"
python record.py --task "move to the white candy and pick it up"
python record.py --task "pick up the white candy and drop candy into box"
python record.py --task "move to the white candy, pick it up and put into box" --episode-time 45
```
