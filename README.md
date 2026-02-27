# LeKiwi

Host

```bash
ssh jetsonl7
cd ~/lerobot
conda lerobot activate
python -m lerobot.robots.lekiwi.lekiwi_host --robot.id=follower --host.connection_time_s 3600
```

Master

```bash
conda lerobot activate
python teleoperate.py
```

Add to .env

```text
LEADER_PORT=/dev/ttyACM0
LEADER_ID=leader1
JETSON_IP=10.76.184.62
ROBOT_ID=follower
```
