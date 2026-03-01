import pygame

class PS3Teleop:
    def __init__(self):
        self.is_connected = False
        self.joystick = None
        # Deadband threshold to prevent the robot from creeping when sticks are centered
        self.deadband = 0.05

    def connect(self):
        pygame.init()
        pygame.joystick.init()
        
        if pygame.joystick.get_count() == 0:
            raise ValueError("No gamepad detected. Ensure it is connected and visible under `/dev/input/js*`.")
        
        self.joystick = pygame.joystick.Joystick(0)
        self.joystick.init()
        self.is_connected = True
        print(f"Connected to gamepad: {self.joystick.get_name()}")

    def disconnect(self):
        if self.is_connected:
            pygame.quit()
            self.is_connected = False
            print("Gamepad disconnected.")

    def get_action(self):
        if not self.is_connected:
            return {}

        # Process internal Pygame events to update the hardware state
        pygame.event.pump()
        
        # --- LEFT JOYSTICK: Translation ---
        # Axis 0: Left Stick X (Pygame: Right is positive. Robotics: Left is positive y)
        # Axis 1: Left Stick Y (Pygame: Down is positive. Robotics: Forward is positive x)
        y_translation = -self.joystick.get_axis(0) 
        x_forward = -self.joystick.get_axis(1)
        
        # --- RIGHT JOYSTICK: Rotation ---
        # Axis 3: Right Stick X on Linux (Axis 2 is the L2 trigger)
        # Pygame: Right is positive. Robotics: CCW/Left turn is positive theta
        theta_turn = -self.joystick.get_axis(3) 
        
        # Apply deadbands
        x_forward = 0.0 if abs(x_forward) < self.deadband else x_forward
        y_translation = 0.0 if abs(y_translation) < self.deadband else y_translation
        theta_turn = 0.0 if abs(theta_turn) < self.deadband else theta_turn
        
        return {
            "x.vel": x_forward,
            "y.vel": y_translation,
            "theta.vel": theta_turn
        }
