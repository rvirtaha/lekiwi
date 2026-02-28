# LeKiwi

Host

```bash
ssh jetsonl7
cd ~/lerobot
conda lerobot activate
python -m lerobot.robots.lekiwi.lekiwi_host --robot.id=follower --host.connection_time_s 3600

# with birdseye camera
python -m lerobot.robots.lekiwi.lekiwi_host --robot.id=follower --host.connection_time_s 3600 --robot.cameras='{"front": {"type": "opencv", "index_or_path": 0, "width": 640, "height": 480, "fps": 30}, "wrist": {"type": "opencv", "index_or_path": 2, "width": 640, "height": 480, "fps": 30}, "birdseye": {"type": "opencv", "index_or_path": "http://10.76.184.104:5050/video_feed", "width": 800, "height": 600, "fps": 25}}'
```

```bash
# fix port (lerobot-find-port) after reboot
sudo chmod 666 /dev/ttyACM0

# fix camera resolution after reboot (OpenCV fails to set 640x480, defaults to 1920x1080)
v4l2-ctl -d /dev/video0 --set-fmt-video=width=640,height=480,pixelformat=MJPG
v4l2-ctl -d /dev/video2 --set-fmt-video=width=640,height=480,pixelformat=MJPG
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

huggingface-cli download rvirtaha/cigarette_pickup_lora --local-dir models/cigarette_pickup_lora
