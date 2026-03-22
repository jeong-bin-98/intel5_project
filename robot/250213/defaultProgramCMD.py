from indy_utils import indydcp_client as client

from time import sleep  
import threading
import numpy as np

robot_ip = "192.168.3.7"  # Robot (Indy) IP
robot_name = "NRMK-Indy7"  # Robot name (Indy7)

indy = client.IndyDCPClient(robot_ip, robot_name)

indy.connect()

indy.set_default_program(5)
cmd_name = "x1"
indy.execute_move(cmd_name)

sleep(3)

cmd_name = "x2"
indy.execute_move(cmd_name)
sleep(3)

indy.disconnect()