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
        
    def deg_to_rad(self, degrees):
        return degrees * (PI / 180)
    
    def rad_to_deg(self, radians):
        return radians * (180 / PI)
    
    def matrix_multiply(self, A, B):
        result = [[0 for _ in range(len(B[0]))] for _ in range(len(A))]
        for i in range(len(A)):
            for j in range(len(B[0])):
                for k in range(len(B)):
                    result[i][j] += A[i][k] * B[k][j]
        return result
    
    def DH_Martrix_Transform(self, theta, d, a, alpha):
        theta = self.deg_to_rad(theta)
        alpha = self.deg_to_rad(alpha)
        
        DH_matrix = [[math.cos(theta), -math.sin(theta)*math.cos(alpha),  math.sin(theta)*math.sin(alpha), a*math.cos(theta)],
                    [math.sin(theta),  math.cos(theta)*math.cos(alpha), -math.cos(theta)*math.sin(alpha), a*math.sin(theta)],
                    [0,                math.sin(alpha),                   math.cos(alpha),                  d],
                    [0,                0,                                 0,                                1]]
        return DH_matrix
    
    def get_RRP_Tramsform_Matrix(self, joint_parameters):
        theta1  = joint_parameters[0]
        theta2  = joint_parameters[1]
        d3      = joint_parameters[2]
        
        x1 = self.joint_local_positions[1][0]
        x2 = self.joint_local_positions[2][0]
        x3 = self.joint_local_positions[2][2]
        x4 = self.joint_local_positions[3][0] + d3
        
        z1 = self.joint_local_positions[1][2]
        z2 = self.joint_local_positions[1][1]
        z3 = self.joint_local_positions[2][1]
        z4 = self.joint_local_positions[3][1]
        z5 = self.joint_local_positions[3][2]
        
        T01 = self.DH_Martrix_Transform(theta1,   0,    0,   0)
        T12 = self.DH_Martrix_Transform(0     ,  z1,   x1,   0)
        T23 = self.DH_Martrix_Transform(theta2,  z2,    0,  90)
        T34 = self.DH_Martrix_Transform(90    ,  z3,   x2,   0)
        T45 = self.DH_Martrix_Transform(-90   ,  z4,   x3,   0)
        T5E = self.DH_Martrix_Transform(0     ,  z5,   x4, -90)
        
        to_multiply = [T01, T12, T23, T34, T45, T5E]
        
        T = [[1, 0, 0, 0],
             [0, 1, 0, 0],
             [0, 0, 1, 0],
             [0, 0, 0, 1]]
        
        for matrix in to_multiply:
            T = self.matrix_multiply(T, matrix)
            
        return T
    
    def Forward_Kinematics(self, joint_parameters):
        if len(joint_parameters) != 3:
            raise ValueError("Joint parameters must contain exactly 2 values: [theta1 ,theta2, d3]")
        if not (self.joint_limits[0][0] <= joint_parameters[0] <= self.joint_limits[0][1]):
            raise ValueError(f"Theta1 must be within limits: {self.joint_limits[0]}")
        if not (self.joint_limits[1][0] <= joint_parameters[1] <= self.joint_limits[1][1]):
            raise ValueError(f"Theta2 must be within limits: {self.joint_limits[1]}")
        if not (self.joint_limits[2][0] <= joint_parameters[2] <= self.joint_limits[2][1]):
            raise ValueError(f"d3 must be within limits: {self.joint_limits[2]}")
        
        result_matrix = self.get_RRP_Tramsform_Matrix(joint_parameters)
        position = (result_matrix[0][3], result_matrix[1][3], result_matrix[2][3])
        
        return position
        
        