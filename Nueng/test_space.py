import test_Toolbox as toolbox
import math

link_1 = [(0, 0, 1)]          # Second Joint parameters
link_2 = [(1, 0, 0)]          # Third Joint parameters
end_effector = [(0, 0, 1)]    # End Effector parameters

link_params = [link_1, link_2, end_effector]
joint_limits = [(-180, 180),(-90, 90), (-10, 10)]       # Second Joint and Third Joint limits

rrp_toolbox = toolbox.RRPToolbox(link_params, joint_limits)

joint_parameters = (0, 45, 0)  # theta1, theta2, d3
target_position = (1, 0, 2)
# T = rrp_toolbox.Inverse_Kinematics(target_position)
T = rrp_toolbox.get_RRP_Tramsform_Matrix(joint_parameters)

# print("Calculated Joint Parameters (theta1, theta2, d3):", T)

FK = rrp_toolbox.Forward_Kinematics(joint_parameters)
for row in T:
    print(row)
print("Forward Kinematics Resulting Position:", FK)

# DIK = rrp_toolbox.Differential_Inverse_Kinematics(target_position, 5)
# print("Differential Inverse Kinematics Resulting Joint Velocities:", DIK)

