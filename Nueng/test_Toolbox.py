# Import Necessary Libraries
import math

#Define parameters
PI = math.pi

class RRPToolbox:
    
    def __init__(self, link_params, joint_limits):
        self.link_params  = link_params  # This will contain second Joint, third Joint and End Effector parameters
        self.joint_limits = joint_limits # This will contain second Joint and third Joint limits
        
        self.joint_local_positions  = [(0, 0, 0)] # First Revolute Joiny always at origin
        self.joint_global_positions = [(0, 0, 0)] # First Revolute Joiny always at origin
        self.end_effector_position  = (0, 0, 0)
        
        self.get_position() # Calculate positions upon initialization
        
        print("RRP Toolbox Initialized, (Parameters order (x, y, z))")
        print("-----------------------")
        print("Link Parameters       :", self.link_params)
        print("Joint Limits          :", self.joint_limits)
        print("Joint Local Positions :", self.joint_local_positions)
        print("Joint Global Positions:", self.joint_global_positions)
        print("End Effector Position :", self.end_effector_position)
        
    def get_position(self):
        for sub_vec in self.link_params:
            x, y, z = 0, 0, 0
            for pos in sub_vec:
                x += pos[0]
                y += pos[1]
                z += pos[2]
                
            self.joint_local_positions.append((x, y, z))
            
            x += self.joint_global_positions[-1][0]
            y += self.joint_global_positions[-1][1]
            z += self.joint_global_positions[-1][2]
            
            self.joint_global_positions.append((x, y, z))
        
        self.end_effector_position = self.joint_global_positions[-1]