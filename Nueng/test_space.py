import test_Toolbox as toolbox
import math

link_1 = [(0, 0, 0), (0, 0, 5)]          # Second Joint parameters
link_2 = [(0, 0, 0), (5, 0, 0)]          # Third Joint parameters
end_effector = [(0, 0, 0), (2, 0, 0)]    # End Effector parameters

link_params = [link_1, link_2, end_effector]
joint_limits = [(-180, 180),(-90, 90), (0, 10)]       # Second Joint and Third Joint limits

rrp_toolbox = toolbox.RRPToolbox(link_params, joint_limits)

print(math.cos(math.radians(180)))