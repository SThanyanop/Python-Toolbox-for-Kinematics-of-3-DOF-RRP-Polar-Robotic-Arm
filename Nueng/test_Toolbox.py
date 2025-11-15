#import necessary libraries
import math

#Define necessary constants
PI = math.pi

class TestToolbox:
    
    #initialize function
    def __init__(self, link, joint_limits):
        self.link               = link
        self.joint_limits       = joint_limits
        self.all_joint_position = []             # Initialize with base position note : All position is a local positions
        
        self.register_joint_positions(self.link) # local joint position make transformation easier
        
        print("Initialization complete. Joint positions registered.")
    
    def DH_transform(self, theta, d, a, alpha):
        ct = math.cos(theta)
        st = math.sin(theta)
        ca = math.cos(alpha)
        sa = math.sin(alpha)
        
        return [
            [ct, -st * ca,  st * sa, a * ct],
            [st,  ct * ca, -ct * sa, a * st],
            [0,      sa,      ca,      d],
            [0,      0,       0,      1]
        ]
        
    def matrix_multiply(self, A, B):
        result = [[0 for _ in range(len(B[0]))] for _ in range(len(A))]
        
        for i in range(len(A)):
            for j in range(len(B[0])):
                for k in range(len(B)):
                    result[i][j] += A[i][k] * B[k][j]
                    
        return result
        
    def link_to_position(self, link):
        x, y, z = 0, 0, 0
        for pos in link:
            x += pos[0]
            y += pos[1]
            z += pos[2]
            
        return (x, y, z)
    
    def register_joint_positions(self, link):
        pos_1 = self.link_to_position(link[0]) # Joint 1
        pos_2 = self.link_to_position(link[1]) # Joint 2
        pos_3 = self.link_to_position(link[2]) # Joint 3
        pos_4 = self.link_to_position(link[3]) # End Effector
    
        self.all_joint_position.extend([pos_1, pos_2, pos_3, pos_4]) #Joint 1, Joint 2, Joint 3, End Effector
        
    def ForwardKinematics(self, q1, q2, d3):
        if not (self.joint_limits[0][0] <= q1 <= self.joint_limits[0][1]):
            raise ValueError("q1 out of limits")
        if not (self.joint_limits[1][0] <= q2 <= self.joint_limits[1][1]):
            raise ValueError("q2 out of limits")
        if not (self.joint_limits[2][0] <= d3 <= self.joint_limits[2][1]):
            raise ValueError("d3 out of limits")
        
        local_transforms = [self.DH_transform(0, 0, 0, -PI/2)] # initial transformation
        
        for i,pos in enumerate(self.all_joint_position):
            
            x, y, z = pos
            
            theta = 0 # Z axis rotation
            d     = 0 # Z axis translation
            alpha = 0 # X axis rotation
            a     = 0 # X axis translation

            # Move X axis and Z axis first
            
            if i == 2: # prismatic joint
                x += d3
            
            a = x
            alpha = 0
            d = y
            theta = 0
            local_transforms.append(self.DH_transform(theta, d, a, alpha))
            
            # Rotate X, Move Z and Rotate X back
            a = 0
            d = z
            # Rotate and move according to link order
            if i == 0:
                theta = q1 # First joint rotation
            else:
                theta = 0  # Need to rotate back to 0 for other joints
            
            # Determine alpha based on y position
            # This Rotate X and move Z if i = 0 then rotate Z for q1
            if y > 0:
                alpha = -PI/2
            else:
                alpha = PI/2    
            local_transforms.append(self.DH_transform(theta, d, a, alpha))
            
            # If it's the second joint, apply q2 rotation
            if i == 1:
                theta = q2
            else:
                theta = 0
            alpha = -alpha
            local_transforms.append(self.DH_transform(theta, d, a, alpha))
            
            # For the end effector, apply fixed transformation and rotate axis to normal axis
            if i == 3:
                a = 0
                d = 0
                theta = 0
                alpha = -PI/2
                local_transforms.append(self.DH_transform(theta, d, a, alpha))
                
        # Combine all local transformations
        T = [[1 if i == j else 0 for j in range(4)] for i in range(4)] # Identity matrix
        for lt in local_transforms:
            T = self.matrix_multiply(T, lt)
            
        return (T[0][3], T[1][3], T[2][3])  # Return end effector position (x, y, z)
                
                    
                