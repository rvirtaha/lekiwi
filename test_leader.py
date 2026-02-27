import time
from lerobot.motors import Motor
from lerobot.motors.motors_bus import MotorNormMode
from lerobot.motors.feetech import FeetechMotorsBus

NM = MotorNormMode.DEGREES
bus = FeetechMotorsBus(port="/dev/ttyACM0", motors={
    "shoulder_pan": Motor(id=1, model="sts3215", norm_mode=NM),
    "shoulder_lift": Motor(id=2, model="sts3215", norm_mode=NM),
    "elbow_flex": Motor(id=3, model="sts3215", norm_mode=NM),
    "wrist_flex": Motor(id=4, model="sts3215", norm_mode=NM),
    "wrist_roll": Motor(id=5, model="sts3215", norm_mode=NM),
    "gripper": Motor(id=6, model="sts3215", norm_mode=NM),
})
bus.connect()
print("Connected. Move the arm...")
motors = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
for _ in range(20):
    positions = {m: bus.read("Present_Position", m, normalize=False) for m in motors}
    print(positions)
    time.sleep(0.25)
bus.disconnect()
