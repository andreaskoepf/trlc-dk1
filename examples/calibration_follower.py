from lerobot_robot_trlc_dk1.motors.DM_Control_Python.DM_CAN import *

import argparse
import serial
import time

parser = argparse.ArgumentParser(description="Set follower motor encoder zeros")
parser.add_argument("--port", default="/dev/ttyACM1", help="Follower serial port")
args = parser.parse_args()

port = args.port

ser = serial.Serial(port, 921600, timeout=0.5)
time.sleep(0.5)
control = MotorControl(ser)

motors = {
    "joint_1": Motor(DM_Motor_Type.DM4340, 0x01, 0x11),
    "joint_2": Motor(DM_Motor_Type.DM4340, 0x02, 0x12),
    "joint_3": Motor(DM_Motor_Type.DM4340, 0x03, 0x13),
    "joint_4": Motor(DM_Motor_Type.DM4310, 0x04, 0x14),
    "joint_5": Motor(DM_Motor_Type.DM4310, 0x05, 0x15),
    "joint_6": Motor(DM_Motor_Type.DM4310, 0x06, 0x16),
    "gripper": Motor(DM_Motor_Type.DM4310, 0x07, 0x17),
}

for key, motor in motors.items():
    control.addMotor(motor)
    for _ in range(3):
        control.refresh_motor_status(motor)
        time.sleep(0.01)

    if control.read_motor_param(motor, DM_variable.CTRL_MODE) is not None:
        print(f"{key} ({motor.MotorType.name}) is connected.")
    else:
        raise Exception(f"Unable to read from {key} ({motor.MotorType.name}).")

for key, motor in motors.items():
    control.set_zero_position(motor)
    print(f"{key} ({motor.MotorType.name}) set to zero position.")

control.serial_.close()
