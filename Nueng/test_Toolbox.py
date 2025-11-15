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
        return degrees * (PI / 180.0)
    
    def rad_to_deg(self, radians):
        return radians * (180.0 / PI)
    
    def matrix_multiply(self, A, B):
        result = [[0 for _ in range(len(B[0]))] for _ in range(len(A))]
        for i in range(len(A)):
            for j in range(len(B[0])):
                for k in range(len(B)):
                    result[i][j] += A[i][k] * B[k][j]
        return result
    
    def DH_Martrix_Transform(self, a, alpha, d, theta):
        theta = self.deg_to_rad(theta)
        alpha = self.deg_to_rad(alpha)
        
        TX = [[1, 0, 0, a],
              [0, math.cos(alpha), -math.sin(alpha), 0],
                [0, math.sin(alpha),  math.cos(alpha), 0],
                [0, 0, 0, 1]]
        TZ = [[math.cos(theta), -math.sin(theta), 0, 0],
              [math.sin(theta),  math.cos(theta), 0, 0],
                [0, 0, 1, d],
                [0, 0, 0, 1]]
        
        DH_matrix = self.matrix_multiply(TX, TZ)
        
        return DH_matrix
    
    def get_RRP_Tramsform_Matrix(self, joint_parameters):
        theta1  = joint_parameters[0]
        theta2  = joint_parameters[1]
        d3      = joint_parameters[2]
        
        x1 = self.joint_local_positions[1][0]  # Link 1 x
        y1 = self.joint_local_positions[1][1]  # Link 1 y
        z1 = self.joint_local_positions[1][2]  # Link 1 z
        
        x2 = self.joint_local_positions[2][0]  # Link 2 x
        y2 = self.joint_local_positions[2][1]  # Link 2 y
        z2 = self.joint_local_positions[2][2]  # Link 2 z
        
        x3 = self.joint_local_positions[3][0]  # End Effector x
        y3 = self.joint_local_positions[3][1]  # End Effector y
        z3 = self.joint_local_positions[3][2]  # End Effector z
        
        T01 = self.DH_Martrix_Transform(0,      0,      0,   theta1)
        T12 = self.DH_Martrix_Transform(x1,     0,     z1,        0)
        T23 = self.DH_Martrix_Transform(0,     90,     y1,   theta2)
        T34 = self.DH_Martrix_Transform(x2,     0,     y2,       90)
        T45 = self.DH_Martrix_Transform(z2,    90,      0,        0)
        T56 = self.DH_Martrix_Transform(z3,     0,  x3+d3,       90)
        T6E = self.DH_Martrix_Transform(y3,    90,      0,       90)
        
        to_multiply = [T01, T12, T23, T34, T45, T56, T6E]
        
        for i in range(len(to_multiply)):
            for j in range(len(to_multiply[i])):
                for k in range(len(to_multiply[i][j])):
                    if abs(to_multiply[i][j][k]) < 1e-10:
                        to_multiply[i][j][k] = 0.0
        
        T = [[1, 0, 0, 0],
             [0, 1, 0, 0],
             [0, 0, 1, 0],
             [0, 0, 0, 1]]
        
        for matrix in to_multiply:
            T = self.matrix_multiply(T, matrix)
            
        return T
    
    def get_RRP_Jacobian_Matrix(self, joint_parameters):
        theta1  = joint_parameters[0]
        theta2  = joint_parameters[1]
        d3      = joint_parameters[2]
        
        
    
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
        
        