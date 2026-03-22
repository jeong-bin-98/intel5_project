from indy_utils import indydcp_client as client

robot_ip = "192.168.3.7"
robot_name = "NRMK-Indy7"

indy = client.IndyDCPClient(robot_ip, robot_name)
indy.connect()
indy.disconnect()