"""
RRP Robotic Arm Visualization Toolbox
A comprehensive toolbox for simulating and visualizing an RRP (Revolute-Revolute-Prismatic) robotic arm.
"""
import math
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
        theta1  = self.deg_to_rad(joint_parameters[0])
        theta2  = self.deg_to_rad(joint_parameters[1])
        d3      = self.deg_to_rad(joint_parameters[2])
        
        c1      = math.cos(theta1)
        s1      = math.sin(theta1)
        c2      = math.cos(theta2)
        s2      = math.sin(theta2)
        
        x1 = self.joint_local_positions[1][0]  # Link 1 x
        y1 = self.joint_local_positions[1][1]  # Link 1 y
        z1 = self.joint_local_positions[1][2]  # Link 1 z
        
        x2 = self.joint_local_positions[2][0]  # Link 2 x
        y2 = self.joint_local_positions[2][1]  # Link 2 y
        z2 = self.joint_local_positions[2][2]  # Link 2 z
        
        x3 = self.joint_local_positions[3][0]  # End Effector x
        y3 = self.joint_local_positions[3][1]  # End Effector y
        z3 = self.joint_local_positions[3][2]  # End Effector z
        
        J = [[ -s1*x1 + c1*y1 + c1*y2 + c1*y3 - s1*c2*(d3+x3) - s1*c2*x2 + s1*s2*z2 + s1*s2*z3, -c1*s2*(d3+x3) - c1*s2*x2 - c1*c2*z2 - c1*c2*z3, c1*c2],
             [  c1*x1 + s1*y1 + s1*y2 + s1*y3 - c1*c2*(d3+x3) + c1*c2*x2 - c1*s2*z2 - c1*s2*z3, -s1*s2*(d3+x3) - s1*s2*x2 - s1*c2*z2 - s1*c2*z3, s1*c2],
             [                                                                               0,              c2*(d3+x3) - s2*z2 - s2*z3 + c2*x2,    s2],
             [                                                                               0,                                               0,     0],
             [                                                                               0,                                               0,     0],
             [                                                                               1,                                               1,     0]]
        
        reduced_J = [J[0][:], J[1][:], J[2][:]]
        
        return J,reduced_J
    
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
    
    def Inverse_Kinematics(self, target_position):
        x, y, z = target_position
        x1 = self.joint_global_positions[1][0]  # Link 1 x
        y1 = self.joint_global_positions[1][1]  # Link 1 y
        z1 = self.joint_global_positions[1][2]  # Link 1 z
        
        x2 = self.joint_global_positions[2][0]  # Link 2 x
        y2 = self.joint_global_positions[2][1]  # Link 2 y
        z2 = self.joint_global_positions[2][2]  # Link 2 z
        
        x3 = self.joint_global_positions[3][0]  # End Effector x
        y3 = self.joint_global_positions[3][1]  # End Effector y
        z3 = self.joint_global_positions[3][2]  # End Effector z
        
        
        offest_theta1 = math.atan2(y3, x3)
        theta1 = math.atan2(y, x) - offest_theta1
        
        offset_theta2 = math.atan2(z3-z1, math.sqrt((x3-x1)**2 + (y3-y1)**2))
        theta2 = math.atan2(z - z1, math.sqrt((x - x1)**2 + (y - y1)**2)) - offset_theta2
        
        z1 = self.joint_local_positions[1][2]  # Link 1 z
        z2 = self.joint_local_positions[2][2]  # Link 2 z
        z3 = self.joint_local_positions[3][2]  # End Effector z
        x2 = self.joint_local_positions[2][0]  # Link 2
        x3 = self.joint_local_positions[3][0]  # End Effector
        
        d3 = ((z - z1 - math.cos(theta2)*z2 - math.cos(theta2)*z3 - math.sin(theta2)*x2) / math.sin(theta2)) - x3
        
        theta1 = self.rad_to_deg(theta1)
        theta2 = self.rad_to_deg(theta2)
        
        if not (self.joint_limits[0][0] <= theta1 <= self.joint_limits[0][1]):
            raise ValueError(f"Calculated theta1 {theta1} is out of limits: {self.joint_limits[0]}")
        if not (self.joint_limits[1][0] <= theta2 <= self.joint_limits[1][1]):
            raise ValueError(f"Calculated theta2 {theta2} is out of limits: {self.joint_limits[1]}")
        if not (self.joint_limits[2][0] <= d3 <= self.joint_limits[2][1]):
            raise ValueError(f"Calculated d3 {d3} is out of limits: {self.joint_limits[2]}")
        
        return (theta1, theta2, d3)
        
    def Differential_Forward_Kinematics(self, joint_parameters, time):
        J, reduced_J = self.get_RRP_Jacobian_Matrix(joint_parameters)
        
        end_effector_velocities = [0, 0, 0]
        
        joint_velocities = (joint_parameters[0]/time, joint_parameters[1]/time, joint_parameters[2]/time)
        
        for i in range(3):
            for j in range(3):
                end_effector_velocities[i] += reduced_J[i][j] * joint_velocities[j]
        
        return tuple(end_effector_velocities)
    
    def Differential_Inverse_Kinematics(self, target_position, time):
        J, reduced_J = self.get_RRP_Jacobian_Matrix((0,0,0))
        
        joint_velocities = [0, 0, 0]
        
        target_velocities = (target_position[0]/time, target_position[1]/time, target_position[2]/time)
        
        for i in range(3):
            for j in range(3):
                if reduced_J[i][j] == 0:
                    continue
                joint_velocities[j] += target_velocities[i] / reduced_J[i][j]
        
        for i in range(3):
            joint_velocities[i] = joint_velocities[i] * time
        
        return tuple(joint_velocities)
    
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
                    th1_rad = self.deg_to_rad(theta1)
                    th2_rad = self.deg_to_rad(theta2)
                    
                    c1 = math.cos(th1_rad)
                    s1 = math.sin(th1_rad)
                    
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
                    c2 = math.cos(th2_rad)
                    s2 = math.sin(th2_rad)
                    
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
        import numpy as np
        
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
                        # Get the reduced Jacobian (3x3)
                        J, reduced_J = self.get_RRP_Jacobian_Matrix([theta1, theta2, d3])
                        
                        # Compute determinant
                        J_array = np.array(reduced_J)
                        det = np.linalg.det(J_array)
                        
                        # Check if singular
                        if abs(det) < singularity_threshold:
                            position = self.Forward_Kinematics([theta1, theta2, d3])
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
        Includes singularities in the convex hull calculation for complete boundary coverage.
        
        Args:
            theta1_samples: Number of samples for theta1
            theta2_samples: Number of samples for theta2
            d3_samples: Number of samples for d3
        """
        try:
            import matplotlib.pyplot as plt
            from mpl_toolkits.mplot3d import Axes3D
            from scipy.spatial import ConvexHull
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
        # COMMENTED OUT: Testing workspace without singularities
        all_points = workspace_points + singularity_positions
        # all_points = workspace_points
        
        if len(all_points) < 4:
            print("Not enough points to compute convex hull (need at least 4).")
            return
        
        # Convert to numpy array
        points = np.array(all_points)
        
        # Compute convex hull (or skip if degenerate)
        hull = None
        try:
            hull = ConvexHull(points)
        except Exception as e:
            print(f"Could not compute 3D convex hull: {e}")
            print(f"Points shape: {points.shape} - this may be a 2D cross-section")
            # For 2D cases, we'll still plot but without convex hull faces
        
        # Create 3D plot
        fig = plt.figure(figsize=(16, 11))
        
        # Main plot area
        ax = fig.add_subplot(111, projection='3d')
        
        # Slider axes
        ax_elev = fig.add_axes([0.2, 0.05, 0.3, 0.03])
        ax_azim = fig.add_axes([0.2, 0.01, 0.3, 0.03])
        ax_toggle_vertices = fig.add_axes([0.65, 0.05, 0.15, 0.03])
        ax_toggle_faces = fig.add_axes([0.65, 0.01, 0.15, 0.03])
        
        from matplotlib.widgets import Slider, Button
        
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
        
        # Set initial camera view to elevation=0, azimuth=0
        slider_elev.set_val(0)
        slider_azim.set_val(0)
        
        # Plot the convex hull surface with transparent face colors (hollow - no top/bottom caps)
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection
        
        if hull is not None:
            # Skip plotting convex hull faces - we'll use custom boundary faces instead
            # (commented out to avoid overlapping with boundary face toggle)
            pass
        else:
            # For 2D cases, we'll still plot but without convex hull faces
            print("Using 2D cross-section visualization (no 3D hull)")
        
        # Convert workspace points to array first
        ws_array = np.array(workspace_points)
        
        # Connect workspace grid points with edges
        # Group points by (theta1, theta2) and connect d3 values
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
        
        # Connect edges along d3 direction (within same theta1, theta2)
        # Only connect d3_min to d3_max at theta2 boundaries (theta2_min and theta2_max)
        for i in range(theta1_samples):
            for j in range(len(theta2_range)):
                # Check if this is a boundary theta2
                is_theta2_boundary = (j == 0 or j == len(theta2_range) - 1)
                
                idx_min = point_map.get((i, j, 0))
                idx_max = point_map.get((i, j, 1))
                
                if idx_min is not None and idx_max is not None and idx_min < len(ws_array) and idx_max < len(ws_array):
                    # Only draw d3 connection if at theta2 boundary
                    if is_theta2_boundary:
                        p_min = ws_array[idx_min]
                        p_max = ws_array[idx_max]
                        ax.plot3D([p_min[0], p_max[0]], [p_min[1], p_max[1]], [p_min[2], p_max[2]], 
                                 'b-', linewidth=1, alpha=0.5)
        
        # Also connect edges along d3 direction at theta1 boundaries (theta1_min and theta1_max)
        for i in [0, theta1_samples - 1]:
            for j in range(len(theta2_range)):
                idx_min = point_map.get((i, j, 0))
                idx_max = point_map.get((i, j, 1))
                
                if idx_min is not None and idx_max is not None and idx_min < len(ws_array) and idx_max < len(ws_array):
                    p_min = ws_array[idx_min]
                    p_max = ws_array[idx_max]
                    ax.plot3D([p_min[0], p_max[0]], [p_min[1], p_max[1]], [p_min[2], p_max[2]], 
                             'b-', linewidth=1, alpha=0.5)
        
        # Connect edges along theta2 direction (within same theta1, d3)
        # Draw at all theta1 values to fully connect the structure
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
        
        # Connect edges along theta1 direction (within same theta2, d3)
        # Draw at all theta2 values to fully connect the structure
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
        
        # Create faces from connected edges (avoid duplicates by tracking processed faces)
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection
        
        faces = []
        face_set = set()  # Track processed faces to avoid duplicates
        
        # Create faces in theta2 direction (rectangular faces in theta1-theta2 plane)
        for i in range(theta1_samples - 1):
            for d3_idx in range(2):
                for j in range(len(theta2_range) - 1):
                    idx1 = point_map.get((i, j, d3_idx))
                    idx2 = point_map.get((i + 1, j, d3_idx))
                    idx3 = point_map.get((i + 1, j + 1, d3_idx))
                    idx4 = point_map.get((i, j + 1, d3_idx))
                    
                    if all(idx is not None and idx < len(ws_array) for idx in [idx1, idx2, idx3, idx4]):
                        # Create face as sorted tuple to avoid duplicates
                        face_key = tuple(sorted([idx1, idx2, idx3, idx4]))
                        if face_key not in face_set:
                            face_set.add(face_key)
                            p1 = ws_array[idx1]
                            p2 = ws_array[idx2]
                            p3 = ws_array[idx3]
                            p4 = ws_array[idx4]
                            faces.append([p1, p2, p3, p4])
        
        # Create faces in theta1 direction at theta2 boundaries (rectangular faces)
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
        
        # Create faces in d3 direction at theta1 boundaries (rectangular faces)
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
            face_collection.set_visible(toggle_state['faces_visible'])  # Initially hidden
            ax.add_collection3d(face_collection)
            face_collections.append(face_collection)  # Store for toggle button
        
        # Plot all workspace points - these are the actual grid vertices
        z_values_ws = ws_array[:, 2]  # Get Z coordinates of workspace points
        vertex_scatter = ax.scatter(ws_array[:, 0], 
                            ws_array[:, 1], 
                            ws_array[:, 2],
                            c=z_values_ws, cmap='viridis', marker='o', s=40, alpha=0.8, 
                            label='Workspace Grid Vertices', edgecolors='black', linewidth=0.5)
        
        # Add colorbar to show Z-axis scale
        cbar = plt.colorbar(vertex_scatter, ax=ax, pad=0.1, shrink=0.8)
        cbar.set_label('Height (Z-axis)', rotation=270, labelpad=15)
        
        # Plot singularity positions (highlighted)
        if singularity_positions:
            sing_array = np.array(singularity_positions)
            ax.scatter(sing_array[:, 0], 
                      sing_array[:, 1], 
                      sing_array[:, 2],
                      c='purple', marker='X', s=200, alpha=0.9, 
                      label=f'Singularities ({len(singularity_positions)})', 
                      edgecolors='black', linewidth=1.5)
        
        # Plot robot configuration at a specific joint angle
        # joint_params = [(self.joint_limits[0][0]+self.joint_limits[0][1])/2, (self.joint_limits[1][0]+self.joint_limits[1][1])/2, (self.joint_limits[2][0]+self.joint_limits[2][1])/2]  # Sample configuration
        # joint_params = [self.joint_limits[0][0], self.joint_limits[1][0], self.joint_limits[2][0]]  # Min configuration
        joint_params = [0,30,0.3]
        
        # Get all joint positions for this configuration
        result_matrix = self.get_RRP_Tramsform_Matrix(joint_params)
        
        # Create line from origin through all joint positions
        joint_positions = [
            (0, 0, 0),  # Base (origin)
            self.joint_global_positions[1],  # Joint 1
            self.joint_global_positions[2],  # Joint 2
            (result_matrix[0][3], result_matrix[1][3], result_matrix[2][3])  # End Effector
        ]
        
        # Plot robot links (as lines connecting joints)
        for i in range(len(joint_positions) - 1):
            j1 = joint_positions[i]
            j2 = joint_positions[i + 1]
            ax.plot3D([j1[0], j2[0]], [j1[1], j2[1]], [j1[2], j2[2]], 
                     'r-', linewidth=3, label='Robot Link' if i == 0 else '')
        
        # Plot joint positions
        for i, pos in enumerate(joint_positions[:-1]):
            if i == 0:
                ax.scatter(*pos, c='green', marker='s', s=150, label='Base/Origin')
            else:
                ax.scatter(*pos, c='orange', marker='o', s=100, label='Joint' if i == 1 else '')
        
        # Plot end effector
        end_effector_pos = joint_positions[-1]
        ax.scatter(*end_effector_pos, c='red', marker='^', s=200, 
                  label='End Effector', edgecolors='darkred', linewidth=2)
        
        # Add text labels
        ax.text(0, 0, 0, 'Base', fontsize=10, color='green')
        for i, pos in enumerate(joint_positions[1:-1], 1):
            ax.text(pos[0], pos[1], pos[2], f'Joint {i}', fontsize=9, color='orange')
        ax.text(end_effector_pos[0], end_effector_pos[1], end_effector_pos[2], 
               'EE', fontsize=10, color='red', weight='bold')
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        
        # Set equal axis lengths to fit from base (0,0,0) to farthest end effector
        x_max = max(abs(points[:, 0].min()), abs(points[:, 0].max()))
        y_max = max(abs(points[:, 1].min()), abs(points[:, 1].max()))
        z_max = max(abs(points[:, 2].min()), abs(points[:, 2].max()))
        
        # Use the maximum distance from origin
        max_distance = max(x_max, y_max, z_max)
        
        # Add small margin
        margin = max_distance * 0.1
        axis_limit = max_distance + margin
        
        # Center on origin (base position at 0,0,0)
        ax.set_xlim(-axis_limit, axis_limit)
        ax.set_ylim(-axis_limit, axis_limit)
        ax.set_zlim(-axis_limit, axis_limit)
        
        # Create title with joint limits information
        q1_min, q1_max = self.joint_limits[0]
        q2_min, q2_max = self.joint_limits[1]
        d3_min, d3_max = self.joint_limits[2]
        
        title_text = f'RRP Robot Workspace with Singularities (Interactive)\n'
        title_text += f'Configuration: θ1={joint_params[0]:.1f}°, θ2={joint_params[1]:.1f}°, d3={joint_params[2]:.1f}\n'
        title_text += f'Limits - θ1: [{q1_min:.1f}°, {q1_max:.1f}°], θ2: [{q2_min:.1f}°, {q2_max:.1f}°], d3: [{d3_min:.1f}, {d3_max:.1f}]'
        
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
        (0, 180),  # theta1 limits
        (0, 180),    # theta2 limits
        (3, 5)       # d3 limits
    ]
    
    # Create an instance
    toolbox = RRPToolbox(link_params, joint_limits)
    
    # Example: Forward Kinematics (use valid joint parameters within limits)
    # joint_parameters = [10, 0, 2.5]  # theta1=45°, theta2=0°, d3=2.5
    # position = toolbox.Forward_Kinematics(joint_parameters)
    # print("End Effector Position:", position)
    
    # Example: Inverse Kinematics (use position from forward kinematics to ensure it's valid)
    # try:
    #     joint_angles = toolbox.Inverse_Kinematics(position)
    #     print("Joint Parameters:", joint_angles)
    # except ValueError as e:
    #     print("Inverse Kinematics Error:", e)
    
    # Example: Get Jacobian Matrix
    J, reduced_J = toolbox.get_RRP_Jacobian_Matrix([10, 0, 2.5])
    print("Reduced Jacobian Matrix:", reduced_J)
    
    # Example: Get workspace points
    print("\nGenerating workspace...")
    workspace = toolbox.get_workspace(theta1_samples=10, theta2_samples=10, d3_samples=5)
    print(f"Workspace contains {len(workspace)} reachable points")
    
    # Example: Visualize workspace (requires matplotlib)
    print("Plotting workspace...")
    toolbox.plot_workspace_3d(5,10,5)

