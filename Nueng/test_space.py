import test_Toolbox as tb

tbx = tb.TestToolbox([[(1,1,1),(2,0,1)],[(1,1,1)],[(2,2,2)],[(3,3,3)]], joint_limits=[(-3.14,3.14), (-1.57,1.57), (0,1)])
print(tbx.all_joint_position)
