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
