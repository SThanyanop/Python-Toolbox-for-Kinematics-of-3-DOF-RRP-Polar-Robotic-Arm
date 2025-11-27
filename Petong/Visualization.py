"""
RRP Robotic Arm Visualization Toolbox
A comprehensive toolbox for simulating and visualizing an RRP (Revolute-Revolute-Prismatic) robotic arm.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.widgets import Slider
import warnings
warnings.filterwarnings('ignore')


class RRPRobot:
    """
    RRP Robotic Arm class with forward kinematics and visualization capabilities.
    
    Configuration: Revolute-Revolute-Prismatic
    - Joint 1 (θ1): Revolute joint at base (rotation about z-axis)
    - Joint 2 (θ2): Revolute joint (rotation about y-axis)
    - Joint 3 (d3): Prismatic joint (translation along z-axis)
    """
    
    def __init__(self, link_1_shape=None, link_2_shape=None, end_effector_shape=None, d3_max=1.0):
        """
        Initialize RRP robot parameters with custom link shapes.
        """
        # Set default shapes if not provided
        self.link_1_shape = link_1_shape if link_1_shape is not None else [(0, 0, 1.0)]
        self.link_2_shape = link_2_shape if link_2_shape is not None else [(1.0, 0, 0)]
        self.end_effector_shape = end_effector_shape if end_effector_shape is not None else [(0, 0, 0.2)]
        
        # Calculate link lengths from shapes
        self.L1 = self._calculate_link_length(self.link_1_shape)
        self.L2 = self._calculate_link_length(self.link_2_shape)
        self.ee_length = self._calculate_link_length(self.end_effector_shape)
        self.d3_max = d3_max
        
        # Joint limits
        self.theta1_limits = (-180, 180)  # degrees
        self.theta2_limits = (0, 180)     # degrees
        self.d3_limits = (0, d3_max)      # meters
    
    def _calculate_link_length(self, shape):
        """Calculate total length of a link from its shape segments."""
        total_length = 0
        for segment in shape:
            total_length += np.linalg.norm(segment)
        return total_length
    
    def inverse_kinematics(self, x, y, z):
        """
        Compute inverse kinematics for RRP robot to reach position [x, y, z] at the END EFFECTOR.
        
        Args:
            x, y, z: Target position coordinates at the end effector tip (meters)
        
        Returns:
            (theta1, theta2, d3) tuple or None if unreachable
        """
        # Calculate theta1 (rotation around z-axis)
        theta1 = np.degrees(np.arctan2(y, x))
        th1_rad = np.radians(theta1)
        
        # Get the position at the start of link2 (end of link1)
        # Link 1 goes up by L1
        # Joint 2 is at the start of link 2
        joint2_x = 0
        joint2_y = 0
        joint2_z = self.L1
        
        # Vector from joint2 (start of link2) to target position
        dx = x - joint2_x
        dy = y - joint2_y
        dz = z - joint2_z
        
        # Distance in XY plane and total 3D distance
        r_xy = np.sqrt(dx**2 + dy**2)
        total_distance = np.sqrt(dx**2 + dy**2 + dz**2)
        
        # Calculate theta2 (angle of the prismatic joint + end effector direction)
        if total_distance > 0.001:
            theta2 = np.degrees(np.arctan2(r_xy, dz))
        else:
            theta2 = 0
        
        # The total distance must equal L2 + d3 + end_effector contribution
        d3 = total_distance - self.L2 - self.ee_length
        
        # Check if position is reachable
        if d3 > self.d3_max or d3 < 0:
            print(f"Warning: Position [{x:.2f}, {y:.2f}, {z:.2f}] unreachable. d3={d3:.2f}m (max={self.d3_max}m)")
            d3 = np.clip(d3, 0, self.d3_max)
        
        # Ensure theta2 is within limits
        theta2 = np.clip(theta2, *self.theta2_limits)
        
        return theta1, theta2, d3
    
    def forward_kinematics(self, theta1, theta2, d3):
        """
        Compute forward kinematics for RRP robot with custom link shapes.
        """
        # Convert to radians
        th1 = np.radians(theta1)
        th2 = np.radians(theta2)
        
        # Rotation matrices
        def Rz(angle):
            """Rotation matrix around Z-axis"""
            c, s = np.cos(angle), np.sin(angle)
            return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
        
        # Base position
        positions = [np.array([0, 0, 0])]
        
        # Link 1: Apply shape segments with theta1 rotation
        current_pos = np.array([0, 0, 0])
        for segment in self.link_1_shape:
            segment_vec = np.array(segment)
            rotated_segment = Rz(th1) @ segment_vec
            current_pos = current_pos + rotated_segment
            positions.append(current_pos.copy())
        
        # Joint 2 is now at current_pos (end of Link 1, start of Link 2)
        # Calculate the direction for Link 2, prismatic joint, and end effector
        direction = np.array([
            np.sin(th2) * np.cos(th1),
            np.sin(th2) * np.sin(th1),
            np.cos(th2)
        ])
        
        # Link 2: Apply shape segments in the direction controlled by theta2
        for segment in self.link_2_shape:
            segment_length = np.linalg.norm(segment)
            current_pos = current_pos + segment_length * direction
            positions.append(current_pos.copy())
        
        # Prismatic joint (d3): Extension in the same direction
        current_pos = current_pos + d3 * direction
        positions.append(current_pos.copy())
        
        # End effector: Apply shape segments in the same direction
        for segment in self.end_effector_shape:
            segment_length = np.linalg.norm(segment)
            current_pos = current_pos + segment_length * direction
            positions.append(current_pos.copy())
        
        return np.array(positions)
    
    def plot_robot(self, theta1=0, theta2=0, d3=0, ax=None, show_frame=True):
        """Plot the robot arm in 3D."""
        if ax is None:
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection='3d')
        
        # Get joint positions
        positions = self.forward_kinematics(theta1, theta2, d3)
        
        # Calculate segment indices for different links
        link1_end_idx = len(self.link_1_shape)
        link2_end_idx = link1_end_idx + len(self.link_2_shape)
        d3_end_idx = link2_end_idx + 1
        
        # Plot Link 1
        link1_positions = positions[:link1_end_idx+1]
        ax.plot(link1_positions[:, 0], link1_positions[:, 1], link1_positions[:, 2], 
                'o-', linewidth=3, markersize=8, color='steelblue', label='Link 1')
        
        # Plot Link 2
        link2_positions = positions[link1_end_idx:link2_end_idx+1]
        ax.plot(link2_positions[:, 0], link2_positions[:, 1], link2_positions[:, 2], 
                'o-', linewidth=3, markersize=8, color='coral', label='Link 2')
        
        # Plot Prismatic Joint (d3)
        d3_positions = positions[link2_end_idx:d3_end_idx+1]
        ax.plot(d3_positions[:, 0], d3_positions[:, 1], d3_positions[:, 2], 
                'o-', linewidth=4, markersize=8, color='green', label='Prismatic (d3)')
        
        # Plot End Effector
        if len(self.end_effector_shape) > 0:
            ee_positions = positions[d3_end_idx:]
            ax.plot(ee_positions[:, 0], ee_positions[:, 1], ee_positions[:, 2], 
                    's-', linewidth=2, markersize=10, color='red', label='End Effector')
        
        # Plot base joint
        ax.scatter(positions[0, 0], positions[0, 1], positions[0, 2], 
                  c='red', s=150, marker='o', label='Base', zorder=5)
        
        # Plot coordinate frames
        if show_frame:
            self._plot_frame(ax, np.eye(4), scale=0.3, label='Base')
        
        # Set plot properties
        max_reach = self.L1 + self.L2 + self.d3_max + self.ee_length + 0.5
        ax.set_xlim([-max_reach, max_reach])
        ax.set_ylim([-max_reach, max_reach])
        ax.set_zlim([0, max_reach*1.5])
        
        # Set viewing angle to match the image (elevation=20, azimuth=45)
        ax.view_init(elev=20, azim=45)
        
        ax.set_xlabel('X (m)', fontsize=10)
        ax.set_ylabel('Y (m)', fontsize=10)
        ax.set_zlabel('Z (m)', fontsize=10)
        ax.set_title(f'RRP Robot Configuration\nθ1={theta1:.1f}°, θ2={theta2:.1f}°, d3={d3:.2f}m', 
                    fontsize=12)
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
        
        return ax
    
    def _plot_frame(self, ax, T, scale=0.2, label=''):
        """Plot coordinate frame."""
        origin = T[:3, 3]
        x_axis = T[:3, 0] * scale
        y_axis = T[:3, 1] * scale
        z_axis = T[:3, 2] * scale
        
        ax.quiver(*origin, *x_axis, color='r', arrow_length_ratio=0.2, linewidth=1.5)
        ax.quiver(*origin, *y_axis, color='g', arrow_length_ratio=0.2, linewidth=1.5)
        ax.quiver(*origin, *z_axis, color='b', arrow_length_ratio=0.2, linewidth=1.5)
    
    def interpolate_trajectory(self, waypoints, total_time, fps=30):
        """
        Generate trajectory.
        
        Args:
            waypoints: List of (theta1, theta2, d3) tuples or [x, y, z] positions
            total_time: Total time (in seconds) to complete the entire trajectory
            fps: Frames per second for animation (default: 30)
        
        Returns: List of interpolated joint configurations and corresponding times
        """
        # Generate evenly spaced time stamps for each waypoint
        num_waypoints = len(waypoints)
        times = np.linspace(0, total_time, num_waypoints)
        
        trajectory = []
        time_stamps = []
        
        for i in range(len(waypoints) - 1):
            start = waypoints[i]
            end = waypoints[i + 1]
            t_start = times[i]
            t_end = times[i + 1]
            duration = t_end - t_start
            
            if duration <= 0:
                continue
            
            # Calculate number of frames for this segment
            num_frames = int(duration * fps)
            
            # Linear interpolation between waypoints
            for j in range(num_frames):
                alpha = j / num_frames
                interpolated = tuple(
                    start[k] + alpha * (end[k] - start[k]) 
                    for k in range(len(start))
                )
                trajectory.append(interpolated)
                time_stamps.append(t_start + alpha * duration)
        
        # Add final waypoint
        trajectory.append(waypoints[-1])
        time_stamps.append(times[-1])
        
        return trajectory, time_stamps
    
    def interactive_plot(self):
        """Create interactive plot with sliders for joint control"""
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
        plt.subplots_adjust(bottom=0.25)
        
        # Initial configuration
        theta1_init, theta2_init, d3_init = 0, 30, 0.3
        
        # Create sliders
        ax_theta1 = plt.axes([0.15, 0.15, 0.65, 0.03])
        ax_theta2 = plt.axes([0.15, 0.10, 0.65, 0.03])
        ax_d3 = plt.axes([0.15, 0.05, 0.65, 0.03])
        
        slider_theta1 = Slider(ax_theta1, 'θ1 (°)', *self.theta1_limits, 
                              valinit=theta1_init, valstep=1)
        slider_theta2 = Slider(ax_theta2, 'θ2 (°)', *self.theta2_limits, 
                              valinit=theta2_init, valstep=1)
        slider_d3 = Slider(ax_d3, 'd3 (m)', *self.d3_limits, 
                          valinit=d3_init, valstep=0.01)
        
        def update(val):
            ax.cla()
            theta1 = slider_theta1.val
            theta2 = slider_theta2.val
            d3 = slider_d3.val
            self.plot_robot(theta1, theta2, d3, ax=ax)
            fig.canvas.draw_idle()
        
        slider_theta1.on_changed(update)
        slider_theta2.on_changed(update)
        slider_d3.on_changed(update)
        
        # Initial plot
        self.plot_robot(theta1_init, theta2_init, d3_init, ax=ax)
        plt.show()
    
    def animate_trajectory(self, trajectory, total_time=None, trajectory_type='joint', fps=30):
        """
        Animate robot following a trajectory with time-based control.
    
        Args:
            trajectory: List of waypoints (joint configs or positions)
            total_time: Total time (in seconds) to complete the trajectory. If None, uses default based on number of waypoints
            trajectory_type: 'joint' or 'position'
            fps: Frames per second for animation
        """
        # Generate default time if not provided
        if total_time is None:
            total_time = len(trajectory)  # 1 second per waypoint by default
        
        # Interpolate trajectory
        if trajectory_type == 'position':
            # Convert positions to joint configs first
            joint_waypoints = []
            for pos in trajectory:
                x, y, z = pos
                joint_config = self.inverse_kinematics(x, y, z)
                joint_waypoints.append(joint_config)
            
            joint_trajectory, time_stamps = self.interpolate_trajectory(joint_waypoints, total_time, fps)
        else:
            joint_trajectory, time_stamps = self.interpolate_trajectory(trajectory, total_time, fps)
        
        # Calculate actual end effector positions
        actual_ee_path = []
        for joint_config in joint_trajectory:
            positions = self.forward_kinematics(*joint_config)
            actual_ee_path.append(positions[-1])
        path_points = np.array(actual_ee_path)
        
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        def update_frame(frame):
            ax.cla()
            theta1, theta2, d3 = joint_trajectory[frame]
            self.plot_robot(theta1, theta2, d3, ax=ax, show_frame=False)
            
            # Get all robot positions
            positions = self.forward_kinematics(theta1, theta2, d3)
            ee_pos = positions[-1]
            
            # Plot the actual end effector path
            ax.plot(path_points[:frame+1, 0], path_points[:frame+1, 1], 
                   path_points[:frame+1, 2], 'r--', linewidth=2, 
                   alpha=0.5, label='End Effector Path')
            ax.scatter(path_points[frame, 0], path_points[frame, 1], 
                      path_points[frame, 2], s=100, c='yellow', marker='*')
            
            current_time = time_stamps[frame]
            total_time_display = time_stamps[-1]
            
            ax.set_title(f'Time: {current_time:.2f}s / {total_time_display:.2f}s\n' + 
                        f'θ1={theta1:.1f}°, θ2={theta2:.1f}°, d3={d3:.2f}m\n' +
                        f'End Effector: [{ee_pos[0]:.2f}, {ee_pos[1]:.2f}, {ee_pos[2]:.2f}]')
            ax.legend(loc='upper right')
        
        interval = 1000 / fps  # Convert fps to milliseconds per frame
        anim = FuncAnimation(fig, update_frame, frames=len(joint_trajectory), 
                           interval=interval, repeat=True)
        plt.show()
        return anim


# Example usage
if __name__ == "__main__":
    print("RRP Robotic Arm - Time-Based Control")
    print("=" * 50)

    # Link 1
    link_1 = [(5, 0, 0), (0, 0, 5)]
    
    # Link 2
    link_2 = [(3, 0, 0)]
    
    # End effector
    end_effector = [(0, 0, 0.5)]  
    
    # Create robot with custom shapes
    robot = RRPRobot(
        link_1_shape=link_1,
        link_2_shape=link_2,
        end_effector_shape=end_effector,
        d3_max=11.0
    )
    
    print(f"\nRobot Configuration:")
    print(f"  Link 1 length: {robot.L1:.2f}m")
    print(f"  Link 2 length: {robot.L2:.2f}m")
    print(f"  End effector length: {robot.ee_length:.2f}m")
    
    # Interactive control
    print("\n1. Launching interactive control...")
    print("   Use sliders to control joint angles and prismatic extension")
    robot.interactive_plot()

    # Time-based position trajectory
    print("\n2. Creating time-based position trajectory...")
    
    position_trajectory = [
        [3, 3, 8],
        [3, -3, 10],
        [-3, -3, 8],
        [-3, 3, 10],
        [3, 3, 8],
    ]
    
    # Define total time for entire trajectory (in seconds)
    total_time = 10  # Complete all 4 waypoints in 10 seconds
    
    print(f"Target positions (completing in {total_time}s total):")
    for i, pos in enumerate(position_trajectory):
        print(f"  Point {i+1}: {pos}")
    
    robot.animate_trajectory(position_trajectory, total_time=total_time, 
                           trajectory_type='position', fps=30)
                           