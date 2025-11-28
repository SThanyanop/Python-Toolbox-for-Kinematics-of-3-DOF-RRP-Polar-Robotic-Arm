"""
Example usage of RRPToolbox and RRPVisualization classes.
Demonstrates the separation of concerns between kinematics and visualization/workspace analysis.
"""

from RRPToolbox import RRPToolbox
from RRPVisualization import RRPVisualization


link_params = [
    [(5, 0, 0), (0, 0, 5)],  # Link 1
    [(3, 0, 0)],              # Link 2
    [(0, 0, 0.5)]               # End Effector
]
joint_limits = [
    (0, 90),     # theta1 limits (degrees)
    (0, 180),    # theta2 limits (degrees)
    (0, 5)       # d3 limits (units)
]

robot = RRPToolbox(link_params,joint_limits)

viz = RRPVisualization(robot)

viz.plot_workspace_3d()

v_q = [1,1,1]

DFK = robot.differential_forward_kinematics(v_q)
DIK = robot.differential_inverse_kinematics(DFK)

q = [45,45,3]

FK = robot.forward_kinematics(q)
IK = robot.inverse_kinematics(FK)
RFK = robot.forward_kinematics(IK)

print(FK)
print(RFK)