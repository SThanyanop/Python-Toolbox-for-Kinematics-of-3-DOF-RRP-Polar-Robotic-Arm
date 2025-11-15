import test_Toolbox as toolbox
import math

link_1 = [(0, 0, 5)]          # Second Joint parameters
link_2 = [(5, 0, 0)]          # Third Joint parameters
end_effector = [(0, 0, 0)]    # End Effector parameters

link_params = [link_1, link_2, end_effector]
joint_limits = [(-180, 180),(-90, 90), (-10, 10)]       # Second Joint and Third Joint limits

rrp_toolbox = toolbox.RRPToolbox(link_params, joint_limits)

joint_parameters = (0, 0, 0)  # theta1, theta2, d3
target_position = (3, 0, 8)
T = rrp_toolbox.Inverse_Kinematics(target_position)

print("Calculated Joint Parameters (theta1, theta2, d3):", T)

FK = rrp_toolbox.Forward_Kinematics(T)
print("Forward Kinematics Resulting Position:", FK)