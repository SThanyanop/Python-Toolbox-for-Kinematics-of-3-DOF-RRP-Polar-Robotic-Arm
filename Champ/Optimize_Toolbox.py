import math

PI = math.pi
DEG_TO_RAD = PI / 180.0
RAD_TO_DEG = 180.0 / PI
EPSILON = 1e-10


class RRPToolbox:
    def __init__(self, link_params, joint_limits):
        self.link_params = link_params
        self.joint_limits = joint_limits
        
        self.joint_local_positions = [(0, 0, 0)]
        self.joint_global_positions = [(0, 0, 0)]
        self.end_effector_position = (0, 0, 0)
        
        self.current_joint_config = [0.0, 0.0, 0.0]
        
        self._compute_link_geometry()
        
        print("RRP Toolbox Initialized (Parameters order: x, y, z)")
        print("-" * 50)
        print(f"Link Parameters       : {self.link_params}")
        print(f"Joint Limits          : {self.joint_limits}")
        print(f"Joint Local Positions : {self.joint_local_positions}")
        print(f"Joint Global Positions: {self.joint_global_positions}")
        print(f"End Effector Position : {self.end_effector_position}")
        print(f"Current Configuration : {self.current_joint_config}")
    
    def _compute_link_geometry(self):
        for sub_vec in self.link_params:
            x = sum(pos[0] for pos in sub_vec)
            y = sum(pos[1] for pos in sub_vec)
            z = sum(pos[2] for pos in sub_vec)
                
            self.joint_local_positions.append((x, y, z))
            
            prev = self.joint_global_positions[-1]
            self.joint_global_positions.append((x + prev[0], y + prev[1], z + prev[2]))
        
        self.end_effector_position = self.joint_global_positions[-1]
        
    def update_current_config(self, joint_parameters):

        self._validate_joint_parameters(joint_parameters)
        self.current_joint_config = list(joint_parameters)
        
    def get_current_config(self):

        return tuple(self.current_joint_config)
    
    def get_current_pose(self):

        return self.forward_kinematics(self.current_joint_config, update_state=False)
    
    def set_pose_from_joints(self, joint_parameters):

        position = self.forward_kinematics(joint_parameters, update_state=True)
        print(f"Pose set via joints: {joint_parameters}")
        print(f"End-effector at: ({position[0]:.3f}, {position[1]:.3f}, {position[2]:.3f})")
        return position
    
    def set_pose_from_cartesian(self, target_position, validate=True):
        joint_params = self.inverse_kinematics(target_position, validate=validate)
        self.update_current_config(joint_params)
        print(f"Pose set via Cartesian: {target_position}")
        print(f"Joint configuration: [{joint_params[0]:.3f}, {joint_params[1]:.3f}, {joint_params[2]:.3f}]")
        return joint_params
    
    @staticmethod
    def deg_to_rad(degrees):
        return degrees * DEG_TO_RAD
    
    @staticmethod
    def rad_to_deg(radians):
        return radians * RAD_TO_DEG
    
    @staticmethod
    def matrix_multiply(A, B):

        rows_A, cols_A = len(A), len(A[0])
        cols_B = len(B[0])
        
        result = [[0.0] * cols_B for _ in range(rows_A)]
        
        for i in range(rows_A):
            A_row = A[i]
            result_row = result[i]
            for k in range(cols_A):
                A_ik = A_row[k]
                if abs(A_ik) > EPSILON:
                    B_row = B[k]
                    for j in range(cols_B):
                        result_row[j] += A_ik * B_row[j]
        
        return result
    
    @staticmethod
    def inverse_matrix(matrix):

        size = len(matrix)
        
        augmented = [matrix[i][:] + [float(i == j) for j in range(size)] for i in range(size)]
        
        for i in range(size):
            max_row = i
            max_val = abs(augmented[i][i])
            for j in range(i + 1, size):
                if abs(augmented[j][i]) > max_val:
                    max_val = abs(augmented[j][i])
                    max_row = j
            
            if max_row != i:
                augmented[i], augmented[max_row] = augmented[max_row], augmented[i]
            
            pivot = augmented[i][i]
            if abs(pivot) < EPSILON:
                raise ValueError("Matrix is singular and cannot be inverted")
            
            inv_pivot = 1.0 / pivot
            row_i = augmented[i]
            for k in range(2 * size):
                row_i[k] *= inv_pivot
            
            for j in range(size):
                if j != i:
                    factor = augmented[j][i]
                    if abs(factor) > EPSILON:
                        row_j = augmented[j]
                        for k in range(2 * size):
                            row_j[k] -= factor * row_i[k]
        
        return [row[size:] for row in augmented]
    
    @staticmethod
    def compute_determinant(matrix):

        if len(matrix) != 3 or len(matrix[0]) != 3:
            raise ValueError("Only 3x3 matrices supported")
        
        return (matrix[0][0] * (matrix[1][1] * matrix[2][2] - matrix[1][2] * matrix[2][1]) -
                matrix[0][1] * (matrix[1][0] * matrix[2][2] - matrix[1][2] * matrix[2][0]) +
                matrix[0][2] * (matrix[1][0] * matrix[2][1] - matrix[1][1] * matrix[2][0]))
    
    def dh_matrix_transform(self, a, alpha, d, theta):

        theta_rad = theta * DEG_TO_RAD
        alpha_rad = alpha * DEG_TO_RAD
 
        cos_theta = math.cos(theta_rad)
        sin_theta = math.sin(theta_rad)
        cos_alpha = math.cos(alpha_rad)
        sin_alpha = math.sin(alpha_rad)
        
        # Clean up near-zero values
        if abs(cos_theta) < EPSILON: cos_theta = 0.0
        if abs(sin_theta) < EPSILON: sin_theta = 0.0
        if abs(cos_alpha) < EPSILON: cos_alpha = 0.0
        if abs(sin_alpha) < EPSILON: sin_alpha = 0.0
        
        return [
            [cos_theta, -sin_theta, 0.0, a],
            [sin_theta * cos_alpha, cos_theta * cos_alpha, -sin_alpha, -sin_alpha * d],
            [sin_theta * sin_alpha, cos_theta * sin_alpha, cos_alpha, cos_alpha * d],
            [0.0, 0.0, 0.0, 1.0]
        ]
    
    def _validate_joint_parameters(self, joint_parameters):

        if len(joint_parameters) != 3:
            raise ValueError("Joint parameters must contain exactly 3 values: [theta1, theta2, d3]")
        
        theta1, theta2, d3 = joint_parameters
        
        if not (self.joint_limits[0][0] <= theta1 <= self.joint_limits[0][1]):
            raise ValueError(f"Theta1 ({theta1}) must be within limits: {self.joint_limits[0]}")
        
        if not (self.joint_limits[1][0] <= theta2 <= self.joint_limits[1][1]):
            raise ValueError(f"Theta2 ({theta2}) must be within limits: {self.joint_limits[1]}")
        
        if not (self.joint_limits[2][0] <= d3 <= self.joint_limits[2][1]):
            raise ValueError(f"d3 ({d3}) must be within limits: {self.joint_limits[2]}")
    
    def get_rrp_transform_matrix(self, joint_parameters):

        theta1, theta2, d3 = joint_parameters
        
        x1, y1, z1 = self.joint_local_positions[1]
        x2, y2, z2 = self.joint_local_positions[2]
        x3, y3, z3 = self.joint_local_positions[3]
        
        matrices = [
            self.dh_matrix_transform(0, 0, 0, theta1),
            self.dh_matrix_transform(x1, 0, z1, 0),
            self.dh_matrix_transform(0, 90, y1, theta2),
            self.dh_matrix_transform(x2, 0, y2, 90),
            self.dh_matrix_transform(z2, 90, 0, 0),
            self.dh_matrix_transform(z3, 0, x3 + d3, 90),
            self.dh_matrix_transform(y3, 90, 0, 90)
        ]
        
        T = matrices[0]
        for matrix in matrices[1:]:
            T = self.matrix_multiply(T, matrix)
            
        return T
    
    def get_rrp_jacobian_matrix(self, joint_parameters):

        theta1, theta2, d3 = joint_parameters
        
        theta1_rad = theta1 * DEG_TO_RAD
        theta2_rad = theta2 * DEG_TO_RAD
        
        c1 = math.cos(theta1_rad)
        s1 = math.sin(theta1_rad)
        c2 = math.cos(theta2_rad)
        s2 = math.sin(theta2_rad)
        
        x1, y1, z1 = self.joint_local_positions[1]
        x2, y2, z2 = self.joint_local_positions[2]
        x3, y3, z3 = self.joint_local_positions[3]
        
        y_sum = y1 + y2 + y3
        z_sum = z2 + z3
        d3_x_sum = d3 + x3 + x2
        
        s1_c2 = s1 * c2
        s1_s2 = s1 * s2
        c1_s2 = c1 * s2
        c1_c2 = c1 * c2
        s1_c2_d3 = s1_c2 * d3_x_sum
        c1_s2_d3 = c1_s2 * d3_x_sum
        c1_c2_z = c1_c2 * z_sum
        s1_s2_d3 = s1_s2 * d3_x_sum
        s1_c2_z = s1_c2 * z_sum
        
        J = [
            [-s1*x1 + c1*y_sum - s1_c2_d3 + s1_s2*z_sum,
             -c1_s2_d3 - c1_c2_z,
             c1_c2],
            [c1*x1 + s1*y_sum + c1_c2*d3_x_sum - c1_s2*z_sum,
             -s1_s2_d3 - s1_c2_z,
             s1_c2],
            [0.0,
             c2*d3_x_sum - s2*z_sum,
             s2],
            [0.0, 0.0, 0.0],  
            [0.0, 0.0, 0.0],
            [1.0, 1.0, 0.0]
        ]
        
        reduced_J = [J[0][:], J[1][:], J[2][:]]
        
        try:
            inv_reduced_J = self.inverse_matrix(reduced_J)
        except ValueError as e:
            raise ValueError(f"Singular configuration detected: {e}")
        
        return J, reduced_J, inv_reduced_J
    
    def check_singularity(self, joint_parameters, threshold=1e-3):
        try:
            _, reduced_J, _ = self.get_rrp_jacobian_matrix(joint_parameters)
            det = self.compute_determinant(reduced_J)
            return abs(det) < threshold
        except ValueError:
            return True
    
    def is_reachable(self, target_position):
        x, y, z = target_position

        distance = math.sqrt(x**2 + y**2 + z**2)

        x1, y1, z1 = self.joint_global_positions[1]
        x2, y2, z2 = self.joint_global_positions[2]
        x3, y3, z3 = self.joint_global_positions[3]

        d3_max = self.joint_limits[2][1]
        max_reach = math.sqrt((x1 + x2 + x3 + d3_max)**2 + (y1 + y2 + y3)**2 + (z1 + z2 + z3)**2)

        d3_min = self.joint_limits[2][0]
        min_reach = abs(d3_min) 
        
        if distance > max_reach:
            return False, f"Target too far (distance: {distance:.3f}, max: {max_reach:.3f})"
        
        if distance < min_reach:
            return False, f"Target too close (distance: {distance:.3f}, min: {min_reach:.3f})"
        
        return True, "Reachable"
    
    def forward_kinematics(self, joint_parameters, update_state=True):

        self._validate_joint_parameters(joint_parameters)
        
        T = self.get_rrp_transform_matrix(joint_parameters)
        
        if update_state:
            self.update_current_config(joint_parameters)
        
        return (T[0][3], T[1][3], T[2][3])
    
    def inverse_kinematics(self, target_position, validate=True):

        if validate:
            is_reachable, reason = self.is_reachable(target_position)
            if not is_reachable:
                raise ValueError(f"Target unreachable: {reason}")
        
        x, y, z = target_position

        x1, y1, z1 = self.joint_global_positions[1]
        x2, y2, z2 = self.joint_global_positions[2]
        x3, y3, z3 = self.joint_global_positions[3]

        theta1_offset = math.atan2(y3, x3)
        theta1 = math.atan2(y, x) - theta1_offset

        cos_neg_theta1 = math.cos(-theta1)
        sin_neg_theta1 = math.sin(-theta1)
        
        x_org = x * cos_neg_theta1 - y * sin_neg_theta1 - x1
        y_org = x * sin_neg_theta1 + y * cos_neg_theta1 - y1
        z_org = z - z1

        r_target_sq = x_org**2 + y_org**2 + z_org**2
        
        r_target = math.sqrt(r_target_sq)

        dx = x3 - x1
        dy = y3 - y1
        dz = z3 - z1
        
        dy_sq = dy * dy
        dz_sq = dz * dz
        
        discriminant = r_target_sq - dy_sq - dz_sq
        if discriminant < 0:
            raise ValueError(f"Target unreachable: invalid geometry (discriminant: {discriminant:.3f})")
        
        sqrt_term = math.sqrt(discriminant)
        
        d_minus = -dx - sqrt_term
        d_plus = -dx + sqrt_term
        
        r_ref_sq = dx**2 + dy_sq + dz_sq
        d3 = d_plus if r_target_sq > r_ref_sq else d_minus
        
        dx += d3
        
        theta2_offset = math.atan2(dz, math.sqrt(dx**2 + dy_sq))
        theta2 = math.atan2(z_org, math.sqrt(x_org**2 + y_org**2)) - theta2_offset
        
        result = (theta1 * RAD_TO_DEG, theta2 * RAD_TO_DEG, d3)
        
        try:
            self._validate_joint_parameters(result)
        except ValueError as e:
            raise ValueError(f"IK solution outside joint limits: {e}")
        
        return result
        
    def differential_forward_kinematics(self, joint_velocities, joint_config=None):
        if joint_config is None:
            joint_config = self.current_joint_config
        
        _, reduced_J, _ = self.get_rrp_jacobian_matrix(joint_config)
        
        theta1_dot = joint_velocities[0] * DEG_TO_RAD
        theta2_dot = joint_velocities[1] * DEG_TO_RAD
        d3_dot = joint_velocities[2]

        vx = reduced_J[0][0] * theta1_dot + reduced_J[0][1] * theta2_dot + reduced_J[0][2] * d3_dot
        vy = reduced_J[1][0] * theta1_dot + reduced_J[1][1] * theta2_dot + reduced_J[1][2] * d3_dot
        vz = reduced_J[2][0] * theta1_dot + reduced_J[2][1] * theta2_dot + reduced_J[2][2] * d3_dot
        
        return (vx, vy, vz)
    
    def differential_inverse_kinematics(self, target_velocities, joint_config=None):

        if joint_config is None:
            joint_config = self.current_joint_config
        
        _, _, inv_reduced_J = self.get_rrp_jacobian_matrix(joint_config)
        
        vx = target_velocities[0]
        vy = target_velocities[1]
        vz = target_velocities[2]
        
        theta1_dot = inv_reduced_J[0][0] * vx + inv_reduced_J[0][1] * vy + inv_reduced_J[0][2] * vz
        theta2_dot = inv_reduced_J[1][0] * vx + inv_reduced_J[1][1] * vy + inv_reduced_J[1][2] * vz
        d3_dot = inv_reduced_J[2][0] * vx + inv_reduced_J[2][1] * vy + inv_reduced_J[2][2] * vz
        
        return (theta1_dot, theta2_dot, d3_dot)
    
    def print_status(self):
        print("\n" + "=" * 50)
        print("RRP Robot Status")
        print("=" * 50)
        print(f"Current Configuration: [{self.current_joint_config[0]:.3f}, "
              f"{self.current_joint_config[1]:.3f}, {self.current_joint_config[2]:.3f}]")
        
        try:
            pos = self.forward_kinematics(self.current_joint_config, update_state=False)
            print(f"End-Effector Position: ({pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f})")
        except Exception as e:
            print(f"End-Effector Position: Error - {e}")
        
        is_singular = self.check_singularity(self.current_joint_config)
        print(f"Singularity Status   : {'SINGULAR' if is_singular else 'OK'}")
        print("=" * 50)
    
    def get_workspace(self, theta1_samples=12, theta2_samples=12, d3_samples=6):
        """
        Generate workspace points by sampling joint parameters within their limits.
        Only sample d3 at its boundaries (d3_min and d3_max) to show only outer surface.
        
        Args:
            theta1_samples: Number of samples for theta1
            theta2_samples: Number of samples for theta2
            d3_samples: Not used - always samples only d3_min and d3_max
            
        Returns:
            workspace_points: List of (x, y, z) positions reachable by the robot
        """
        import numpy as np
        workspace_points = []
        
        theta1_min, theta1_max = self.joint_limits[0]
        theta2_min, theta2_max = self.joint_limits[1]
        d3_min, d3_max = self.joint_limits[2]
        
        # Generate samples
        theta1_range = [theta1_min + (theta1_max - theta1_min) * i / (theta1_samples - 1) for i in range(theta1_samples)]
        theta2_range = [theta2_min + (theta2_max - theta2_min) * i / (theta2_samples - 1) for i in range(theta2_samples)]
        d3_range = [d3_min, d3_max]  # Only sample boundary values
        
        # For each theta1, sample ALL theta2 with only d3_min and d3_max
        for theta1 in theta1_range:
            for theta2 in theta2_range:
                for d3 in d3_range:
                    # Convert to radians
                    th1_rad = np.radians(theta1)
                    th2_rad = np.radians(theta2)
                    
                    c1 = math.cos(th1_rad)
                    s1 = math.sin(th1_rad)
                    c2 = math.cos(th2_rad)
                    s2 = math.sin(th2_rad)
                    
                    # Start from base at origin
                    current_x, current_y, current_z = 0, 0, 0
                    
                    # Apply Link 1 segments (rotated by theta1 around Z)
                    for segment in self.link_params[0]:
                        segment_x, segment_y, segment_z = segment
                        # Rotate by theta1 around Z axis
                        rotated_x = c1 * segment_x - s1 * segment_y
                        rotated_y = s1 * segment_x + c1 * segment_y
                        rotated_z = segment_z
                        
                        current_x += rotated_x
                        current_y += rotated_y
                        current_z += rotated_z
                    
                    # Apply Link 2 segments (rotated by theta1 around Z)
                    for segment in self.link_params[1]:
                        segment_x, segment_y, segment_z = segment
                        # Rotate by theta1 around Z axis
                        rotated_x = c1 * segment_x - s1 * segment_y
                        rotated_y = s1 * segment_x + c1 * segment_y
                        rotated_z = segment_z
                        
                        current_x += rotated_x
                        current_y += rotated_y
                        current_z += rotated_z
                    
                    # Apply Prismatic joint (d3) with direction controlled by theta2
                    # Direction: [sin(theta2)*cos(theta1), sin(theta2)*sin(theta1), cos(theta2)]
                    prismatic_x = d3 * s2 * c1
                    prismatic_y = d3 * s2 * s1
                    prismatic_z = d3 * c2
                    
                    current_x += prismatic_x
                    current_y += prismatic_y
                    current_z += prismatic_z
                    
                    # Apply End effector segments
                    for segment in self.link_params[2]:
                        segment_x, segment_y, segment_z = segment
                        # Rotate by theta1 around Z axis
                        rotated_x = c1 * segment_x - s1 * segment_y
                        rotated_y = s1 * segment_x + c1 * segment_y
                        rotated_z = segment_z
                        
                        current_x += rotated_x
                        current_y += rotated_y
                        current_z += rotated_z
                    
                    workspace_points.append((current_x, current_y, current_z))
        
        return workspace_points
    
    def find_singularities(self, theta1_samples=20, theta2_samples=20, d3_samples=10):
        """
        Find singularity positions by computing Jacobian determinant.
        Singularities occur where the Jacobian matrix is singular (determinant ≈ 0).
        
        Args:
            theta1_samples: Number of samples for theta1
            theta2_samples: Number of samples for theta2
            d3_samples: Number of samples for d3
            
        Returns:
            singularity_positions: List of (x, y, z) positions where singularities occur
            singularity_configs: List of ([theta1, theta2, d3], position) tuples
        """
        singularity_positions = []
        singularity_configs = []
        singularity_threshold = 1e-3  # Threshold for determinant to be considered singular
        
        theta1_min, theta1_max = self.joint_limits[0]
        theta2_min, theta2_max = self.joint_limits[1]
        d3_min, d3_max = self.joint_limits[2]
        
        # Generate samples
        def linspace(start, end, samples):
            if samples == 1:
                return [start]
            return [start + (end - start) * i / (samples - 1) for i in range(samples)]

        theta1_range = linspace(theta1_min, theta1_max, theta1_samples)
        theta2_range = linspace(theta2_min, theta2_max, theta2_samples)
        d3_range     = linspace(d3_min,     d3_max,     d3_samples)
        
        for theta1 in theta1_range:
            for theta2 in theta2_range:
                for d3 in d3_range:
                    try:
                        is_singular = self.check_singularity([theta1, theta2, d3], threshold=singularity_threshold)
                        if is_singular:
                            position = self.forward_kinematics([theta1, theta2, d3], update_state=False)
                            singularity_positions.append(position)
                            singularity_configs.append(([theta1, theta2, d3], position))
                    except:
                        # Skip invalid configurations
                        pass
        
        return singularity_positions, singularity_configs
    
    def plot_workspace_3d(self, theta1_samples=25, theta2_samples=25, d3_samples=15):
        """
        Plot the robot workspace in 3D using matplotlib with convex hull.
        Shows the edge/boundary, robot links, joints, end effector, and singularities.
        
        Args:
            theta1_samples: Number of samples for theta1
            theta2_samples: Number of samples for theta2
            d3_samples: Number of samples for d3
        """
        try:
            import matplotlib.pyplot as plt
            from mpl_toolkits.mplot3d import Axes3D
            from mpl_toolkits.mplot3d.art3d import Poly3DCollection
            from matplotlib.widgets import Slider, Button
            import numpy as np
        except ImportError as e:
            print(f"Required libraries missing: {e}")
            print("Install them using: pip install matplotlib scipy numpy")
            return
        
        workspace_points = self.get_workspace(theta1_samples, theta2_samples, d3_samples)
        
        if not workspace_points:
            print("No valid workspace points generated.")
            return
        
        # Find singularities and add them to workspace points
        print("Finding singularities...")
        singularity_positions, singularity_configs = self.find_singularities(
            theta1_samples=max(8, theta1_samples//2), 
            theta2_samples=max(8, theta2_samples//2), 
            d3_samples=max(5, d3_samples//2)
        )
        print(f"Found {len(singularity_positions)} singularity positions")
        
        # Combine workspace points with singularities for complete hull
        all_points = workspace_points + singularity_positions
        
        if len(all_points) < 4:
            print("Not enough points to compute convex hull (need at least 4).")
            return
        
        # Convert to numpy array
        points = np.array(all_points)
        ws_array = np.array(workspace_points)
        
        # Create 3D plot
        fig = plt.figure(figsize=(16, 11))
        ax = fig.add_subplot(111, projection='3d')
        
        # Slider axes
        ax_elev = fig.add_axes([0.2, 0.05, 0.3, 0.03])
        ax_azim = fig.add_axes([0.2, 0.01, 0.3, 0.03])
        ax_toggle_vertices = fig.add_axes([0.65, 0.05, 0.15, 0.03])
        ax_toggle_faces = fig.add_axes([0.65, 0.01, 0.15, 0.03])
        
        # Create sliders
        slider_elev = Slider(ax_elev, 'Elevation', -90, 90, valinit=0, color='orange')
        slider_azim = Slider(ax_azim, 'Azimuth', -180, 180, valinit=0, color='cyan')
        
        # Create toggle buttons for vertices and faces
        btn_toggle_vertices = Button(ax_toggle_vertices, 'Hide Vertices', color='lightgray', hovercolor='0.975')
        btn_toggle_faces = Button(ax_toggle_faces, 'Hide Faces', color='lightgray', hovercolor='0.975')
        
        # State for toggles
        toggle_state = {'vertices_visible': True, 'faces_visible': True}
        vertex_scatter = None
        face_collections = []
        
        def update_view(val):
            """Update the 3D view based on slider values"""
            ax.view_init(elev=slider_elev.val, azim=slider_azim.val)
            fig.canvas.draw_idle()
        
        def toggle_vertices(event):
            """Toggle vertex visibility"""
            toggle_state['vertices_visible'] = not toggle_state['vertices_visible']
            if vertex_scatter is not None:
                vertex_scatter.set_visible(toggle_state['vertices_visible'])
            btn_toggle_vertices.label.set_text('Hide Vertices' if toggle_state['vertices_visible'] else 'Show Vertices')
            fig.canvas.draw_idle()
        
        def toggle_faces(event):
            """Toggle face visibility"""
            toggle_state['faces_visible'] = not toggle_state['faces_visible']
            for face_coll in face_collections:
                face_coll.set_visible(toggle_state['faces_visible'])
            btn_toggle_faces.label.set_text('Hide Faces' if toggle_state['faces_visible'] else 'Show Faces')
            fig.canvas.draw_idle()
        
        slider_elev.on_changed(update_view)
        slider_azim.on_changed(update_view)
        btn_toggle_vertices.on_clicked(toggle_vertices)
        btn_toggle_faces.on_clicked(toggle_faces)
        
        # Set initial camera view
        slider_elev.set_val(0)
        slider_azim.set_val(0)
        
        # Connect workspace grid points with edges
        theta1_min, theta1_max = self.joint_limits[0]
        theta2_min, theta2_max = self.joint_limits[1]
        d3_min, d3_max = self.joint_limits[2]
        
        theta1_range = [theta1_min + (theta1_max - theta1_min) * i / (theta1_samples - 1) for i in range(theta1_samples)]
        theta2_range = [theta2_min + (theta2_max - theta2_min) * i / (theta2_samples - 1) for i in range(theta2_samples)]
        
        # Build a dictionary to map (theta1_idx, theta2_idx, d3_idx) to point index
        point_idx = 0
        point_map = {}
        for i in range(theta1_samples):
            for j in range(len(theta2_range)):
                for d3_idx in range(2):  # 0 for d3_min, 1 for d3_max
                    point_map[(i, j, d3_idx)] = point_idx
                    point_idx += 1
        
        # Connect edges along d3 direction at theta2 boundaries
        for i in range(theta1_samples):
            for j in range(len(theta2_range)):
                is_theta2_boundary = (j == 0 or j == len(theta2_range) - 1)
                
                idx_min = point_map.get((i, j, 0))
                idx_max = point_map.get((i, j, 1))
                
                if idx_min is not None and idx_max is not None and idx_min < len(ws_array) and idx_max < len(ws_array):
                    if is_theta2_boundary:
                        p_min = ws_array[idx_min]
                        p_max = ws_array[idx_max]
                        ax.plot3D([p_min[0], p_max[0]], [p_min[1], p_max[1]], [p_min[2], p_max[2]], 
                                 'b-', linewidth=1, alpha=0.5)
        
        # Connect edges along d3 direction at theta1 boundaries
        for i in [0, theta1_samples - 1]:
            for j in range(len(theta2_range)):
                idx_min = point_map.get((i, j, 0))
                idx_max = point_map.get((i, j, 1))
                
                if idx_min is not None and idx_max is not None and idx_min < len(ws_array) and idx_max < len(ws_array):
                    p_min = ws_array[idx_min]
                    p_max = ws_array[idx_max]
                    ax.plot3D([p_min[0], p_max[0]], [p_min[1], p_max[1]], [p_min[2], p_max[2]], 
                             'b-', linewidth=1, alpha=0.5)
        
        # Connect edges along theta2 direction
        for i in range(theta1_samples):
            for d3_idx in range(2):
                for j in range(len(theta2_range) - 1):
                    idx_curr = point_map.get((i, j, d3_idx))
                    idx_next = point_map.get((i, j + 1, d3_idx))
                    
                    if idx_curr is not None and idx_next is not None and idx_curr < len(ws_array) and idx_next < len(ws_array):
                        p_curr = ws_array[idx_curr]
                        p_next = ws_array[idx_next]
                        ax.plot3D([p_curr[0], p_next[0]], [p_curr[1], p_next[1]], [p_curr[2], p_next[2]], 
                                 'b-', linewidth=1, alpha=0.5)
        
        # Connect edges along theta1 direction
        for j in range(len(theta2_range)):
            for d3_idx in range(2):
                for i in range(theta1_samples - 1):
                    idx_curr = point_map.get((i, j, d3_idx))
                    idx_next = point_map.get((i + 1, j, d3_idx))
                    
                    if idx_curr is not None and idx_next is not None and idx_curr < len(ws_array) and idx_next < len(ws_array):
                        p_curr = ws_array[idx_curr]
                        p_next = ws_array[idx_next]
                        ax.plot3D([p_curr[0], p_next[0]], [p_curr[1], p_next[1]], [p_curr[2], p_next[2]], 
                                 'b-', linewidth=1, alpha=0.5)
        
        # Create faces from connected edges
        faces = []
        face_set = set()
        
        # Create faces in theta2 direction
        for i in range(theta1_samples - 1):
            for d3_idx in range(2):
                for j in range(len(theta2_range) - 1):
                    idx1 = point_map.get((i, j, d3_idx))
                    idx2 = point_map.get((i + 1, j, d3_idx))
                    idx3 = point_map.get((i + 1, j + 1, d3_idx))
                    idx4 = point_map.get((i, j + 1, d3_idx))
                    
                    if all(idx is not None and idx < len(ws_array) for idx in [idx1, idx2, idx3, idx4]):
                        face_key = tuple(sorted([idx1, idx2, idx3, idx4]))
                        if face_key not in face_set:
                            face_set.add(face_key)
                            p1 = ws_array[idx1]
                            p2 = ws_array[idx2]
                            p3 = ws_array[idx3]
                            p4 = ws_array[idx4]
                            faces.append([p1, p2, p3, p4])
        
        # Create faces in theta1 direction at theta2 boundaries
        for j in [0, len(theta2_range) - 1]:
            for i in range(theta1_samples - 1):
                idx1 = point_map.get((i, j, 0))
                idx2 = point_map.get((i + 1, j, 0))
                idx3 = point_map.get((i + 1, j, 1))
                idx4 = point_map.get((i, j, 1))
                
                if all(idx is not None and idx < len(ws_array) for idx in [idx1, idx2, idx3, idx4]):
                    face_key = tuple(sorted([idx1, idx2, idx3, idx4]))
                    if face_key not in face_set:
                        face_set.add(face_key)
                        p1 = ws_array[idx1]
                        p2 = ws_array[idx2]
                        p3 = ws_array[idx3]
                        p4 = ws_array[idx4]
                        faces.append([p1, p2, p3, p4])
        
        # Create faces in d3 direction at theta1 boundaries
        for i in [0, theta1_samples - 1]:
            for j in range(len(theta2_range) - 1):
                idx1 = point_map.get((i, j, 0))
                idx2 = point_map.get((i, j + 1, 0))
                idx3 = point_map.get((i, j + 1, 1))
                idx4 = point_map.get((i, j, 1))
                
                if all(idx is not None and idx < len(ws_array) for idx in [idx1, idx2, idx3, idx4]):
                    face_key = tuple(sorted([idx1, idx2, idx3, idx4]))
                    if face_key not in face_set:
                        face_set.add(face_key)
                        p1 = ws_array[idx1]
                        p2 = ws_array[idx2]
                        p3 = ws_array[idx3]
                        p4 = ws_array[idx4]
                        faces.append([p1, p2, p3, p4])
        
        # Plot faces with transparency
        if faces:
            face_collection = Poly3DCollection(faces, alpha=0.25, facecolor='cyan', edgecolor='none')
            face_collection.set_visible(toggle_state['faces_visible'])
            ax.add_collection3d(face_collection)
            face_collections.append(face_collection)
        
        # Plot workspace vertices
        z_values_ws = ws_array[:, 2]
        vertex_scatter = ax.scatter(ws_array[:, 0], 
                            ws_array[:, 1], 
                            ws_array[:, 2],
                            c=z_values_ws, cmap='viridis', marker='o', s=40, alpha=0.8, 
                            label='Workspace Grid Vertices', edgecolors='black', linewidth=0.5)
        
        # Add colorbar
        cbar = plt.colorbar(vertex_scatter, ax=ax, pad=0.1, shrink=0.8)
        cbar.set_label('Height (Z-axis)', rotation=270, labelpad=15)
        
        # Plot singularity positions
        if singularity_positions:
            sing_array = np.array(singularity_positions)
            ax.scatter(sing_array[:, 0], 
                      sing_array[:, 1], 
                      sing_array[:, 2],
                      c='purple', marker='X', s=200, alpha=0.9, 
                      label=f'Singularities ({len(singularity_positions)})', 
                      edgecolors='black', linewidth=1.5)
        
        # Plot robot configuration
        joint_params = [self.joint_limits[0][0], self.joint_limits[1][0], self.joint_limits[2][1]]
        
        theta1 = joint_params[0]
        theta2 = joint_params[1]
        d3 = joint_params[2]
        
        th1 = np.radians(theta1)
        th2 = np.radians(theta2)
        
        # Build robot arm path
        robot_positions = [np.array([0, 0, 0])]
        current_pos = np.array([0, 0, 0])
        
        def Rz(angle):
            c, s = np.cos(angle), np.sin(angle)
            return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
        
        # Link 1
        for segment in self.link_params[0]:
            segment_vec = np.array(segment)
            rotated_segment = Rz(th1) @ segment_vec
            current_pos = current_pos + rotated_segment
            robot_positions.append(current_pos.copy())
        
        link1_end_idx = len(robot_positions) - 1
        
        # Link 2
        for segment in self.link_params[1]:
            segment_vec = np.array(segment)
            rotated_segment = Rz(th1) @ segment_vec
            current_pos = current_pos + rotated_segment
            robot_positions.append(current_pos.copy())
        
        link2_end_idx = len(robot_positions) - 1
        
        # Prismatic joint (d3)
        direction = np.array([
            np.sin(th2) * np.cos(th1),
            np.sin(th2) * np.sin(th1),
            np.cos(th2)
        ])
        current_pos = current_pos + d3 * direction
        robot_positions.append(current_pos.copy())
        
        d3_end_idx = len(robot_positions) - 1
        
        # End effector
        for segment in self.link_params[2]:
            segment_length = np.linalg.norm(np.array(segment))
            current_pos = current_pos + segment_length * direction
            robot_positions.append(current_pos.copy())
        
        robot_array = np.array(robot_positions)
        
        # Plot Link 1
        link1_positions = robot_array[:link1_end_idx + 1]
        ax.plot3D(link1_positions[:, 0], link1_positions[:, 1], link1_positions[:, 2], 
                 'o-', linewidth=3, markersize=8, color='steelblue', label='Link 1')
        
        # Plot Link 2
        link2_positions = robot_array[link1_end_idx:link2_end_idx + 1]
        ax.plot3D(link2_positions[:, 0], link2_positions[:, 1], link2_positions[:, 2], 
                 'o-', linewidth=3, markersize=8, color='coral', label='Link 2')
        
        # Plot Prismatic Joint (d3)
        d3_positions = robot_array[link2_end_idx:d3_end_idx + 1]
        ax.plot3D(d3_positions[:, 0], d3_positions[:, 1], d3_positions[:, 2], 
                 'o-', linewidth=4, markersize=8, color='green', label='Prismatic (d3)')
        
        # Plot End Effector
        if len(self.link_params[2]) > 0:
            ee_positions = robot_array[d3_end_idx:]
            ax.plot3D(ee_positions[:, 0], ee_positions[:, 1], ee_positions[:, 2], 
                     's-', linewidth=2, markersize=10, color='red', label='End Effector')
        
        # Plot base
        ax.scatter(robot_array[0, 0], robot_array[0, 1], robot_array[0, 2], 
                  c='green', s=150, marker='s', label='Base', zorder=5)
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        
        # Set axis limits
        x_max = max(abs(points[:, 0].min()), abs(points[:, 0].max()))
        y_max = max(abs(points[:, 1].min()), abs(points[:, 1].max()))
        z_max = max(abs(points[:, 2].min()), abs(points[:, 2].max()))
        
        max_distance = max(x_max, y_max, z_max)
        margin = max_distance * 0.1
        axis_limit = max_distance + margin
        
        ax.set_xlim(-axis_limit, axis_limit)
        ax.set_ylim(-axis_limit, axis_limit)
        ax.set_zlim(-axis_limit, axis_limit)
        
        # Title
        q1_min, q1_max = self.joint_limits[0]
        q2_min, q2_max = self.joint_limits[1]
        d3_min_val, d3_max_val = self.joint_limits[2]
        
        title_text = f'RRP Robot Workspace with Singularities (Interactive)\n'
        title_text += f'Configuration: θ1={joint_params[0]:.1f}°, θ2={joint_params[1]:.1f}°, d3={joint_params[2]:.1f}\n'
        title_text += f'Limits - θ1: [{q1_min:.1f}°, {q1_max:.1f}°], θ2: [{q2_min:.1f}°, {q2_max:.1f}°], d3: [{d3_min_val:.1f}, {d3_max_val:.1f}]'
        
        ax.set_title(title_text, fontsize=10)
        ax.legend(loc='upper left', fontsize=9)
        
        # Set initial view
        ax.view_init(elev=20, azim=45)
        
        plt.tight_layout(rect=[0, 0.08, 1, 1])
        plt.show()
        
# Example usage of RRPToolbox
if __name__ == "__main__":
    # Define link parameters and joint limits
    link_params = [
        [ (5, 0, 0), (0, 0, 5) ],  # Link 1
        [(3, 0, 0)],  # Link 2
        [(0, 0, 0)]   # End Effector
    ]
    joint_limits = [
        (0, 90),  # theta1 limits
        (0, 180),    # theta2 limits
        (0, 5)       # d3 limits
    ]
    
    # Create an instance
    toolbox = RRPToolbox(link_params, joint_limits)

    # Example: Get workspace points
    # print("\nGenerating workspace...")
    # workspace = toolbox.get_workspace(theta1_samples=10, theta2_samples=10, d3_samples=5)
    # print(f"Workspace contains {len(workspace)} reachable points")
    
    # Example: Visualize workspace (requires matplotlib)
    print("Plotting workspace...")
    toolbox.plot_workspace_3d(10,10,5)