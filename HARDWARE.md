# Hardware Setup

## Network

| Machine          | Hostname | IP           | User     | Password |
| ---------------- | -------- | ------------ | -------- | -------- |
| Laptop (mylly)   | mylly    | -            | rvirt    | -        |
| Jetson Orin Nano | jetsonl7 | 10.76.184.62 | jetsonl7 | jetsonl7 |

SSH config:

```
Host jetsonl7
    Hostname 10.76.184.62
    User jetsonl7
    Port 22
```

USB gadget mode available if necessary at 192.168.55.1 (direct USB-ethernet) but prefer wifi.

## Physical Connections

**Laptop (mylly)**

- Leader arm (SO101) via USB → /dev/ttyACM0

**Jetson Orin Nano (jetsonl7)**

- Follower arm (SO101) via USB
- LeKiwi omnidirectional base (3 wheel motors)
- Cameras (front + wrist)
- Mounted on the LeKiwi platform
- Connected to WiFi "foobar"

## Architecture

```
┌──────────────┐    ZeroMQ (5555/5556)    ┌──────────────────────┐
│    Laptop    │◄────────WiFi─────────────►│   Jetson on LeKiwi   │
│              │                           │                      │
│ - Leader arm │                           │ - Follower arm       │
│ - Teleop UI  │                           │ - 3 wheel motors     │
│ - Training   │                           │ - Cameras            │
│ - Eval       │                           │ - lekiwi_host        │
└──────────────┘                           └──────────────────────┘
```

Jetson runs `lekiwi_host` (motor control + camera streaming).
Laptop runs everything else (teleop client, recording, training, evaluation).

## Motor IDs

Leader/Follower arm: 1-6 (shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper)
LeKiwi wheels: 7 (left front), 8 (rear), 9 (right front)
All motors: Feetech STS3215, 12V for chassis.
