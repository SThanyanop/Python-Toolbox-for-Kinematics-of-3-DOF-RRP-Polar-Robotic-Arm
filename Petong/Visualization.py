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
        
        Args:
            link_1_shape: List of (x, y, z) tuples defining link 1 path segments
                         Default: [(0, 0, 1.0)] - straight up 1m
            link_2_shape: List of (x, y, z) tuples defining link 2 path segments
                         Default: [(1.0, 0, 0)] - straight horizontal 1m
            end_effector_shape: List of (x, y, z) tuples defining end effector path segments
                               Default: [(0, 0, 0.2)] - small vertical extension
            d3_max: Maximum extension of prismatic joint (meters)
        """
        # Set default shapes if not provided
        self.link_1_shape = link_1_shape if link_1_shape is not None else [(0, 0, 1.0)]
        self.link_2_shape = link_2_shape if link_2_shape is not None else [(1.0, 0, 0)]
        self.end_effector_shape = end_effector_shape if end_effector_shape is not None else [(0, 0, 0.2)]
        
        # Calculate link lengths from shapes
        self.L1 = self._calculate_link_length(self.link_1_shape)
        self.L2 = self._calculate_link_length(self.link_2_shape)
        self.d3_max = d3_max
        
        # Joint limits
        self.theta1_limits = (-180, 180)  # degrees
        self.theta2_limits = (0, 180)     # degrees (0° = vertical/parallel to Z, 90° = horizontal)
        self.d3_limits = (0, d3_max)      # meters
    
    def _calculate_link_length(self, shape):
        """Calculate total length of a link from its shape segments."""
        total_length = 0
        for segment in shape:
            total_length += np.linalg.norm(segment)
        return total_length
    
    def dh_transform(self, theta, d, a, alpha):
        """
        Compute homogeneous transformation matrix using DH parameters.
        
        Args:
            theta: Joint angle (radians)
            d: Link offset
            a: Link length
            alpha: Link twist (radians)
        
        Returns:
            4x4 transformation matrix
        """
        ct, st = np.cos(theta), np.sin(theta)
        ca, sa = np.cos(alpha), np.sin(alpha)
        
        return np.array([
            [ct, -st*ca,  st*sa, a*ct],
            [st,  ct*ca, -ct*sa, a*st],
            [0,   sa,     ca,    d   ],
            [0,   0,      0,     1   ]
        ])
    
    def forward_kinematics(self, theta1, theta2, d3):
        """
        Compute forward kinematics for RRP robot with custom link shapes.
        
        Args:
            theta1: Base rotation angle (degrees)
            theta2: Second joint angle (degrees) - controls prismatic joint direction
            d3: Prismatic joint extension (meters)
        
        Returns:
            List of joint positions for all link segments
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
        
        # Link 2: Apply shape segments with theta1 rotation
        link2_start = current_pos.copy()
        for segment in self.link_2_shape:
            segment_vec = np.array(segment)
            rotated_segment = Rz(th1) @ segment_vec
            current_pos = current_pos + rotated_segment
            positions.append(current_pos.copy())
        
        # Prismatic joint (d3): Extension in direction controlled by theta2
        link3_start = current_pos.copy()
        direction = np.array([
            np.sin(th2) * np.cos(th1),
            np.sin(th2) * np.sin(th1),
            np.cos(th2)
        ])
        current_pos = current_pos + d3 * direction
        positions.append(current_pos.copy())
        
        # End effector: Apply shape segments with theta1 and theta2 rotation
        for segment in self.end_effector_shape:
            segment_vec = np.array(segment)
            # First rotate by theta1, then orient along d3 direction
            rotated_segment = Rz(th1) @ segment_vec
            # Scale by direction for proper orientation
            oriented_segment = rotated_segment * np.array([np.sin(th2), np.sin(th2), np.cos(th2)])
            current_pos = current_pos + oriented_segment
            positions.append(current_pos.copy())
        
        return np.array(positions)
    
    def plot_robot(self, theta1=0, theta2=0, d3=0, ax=None, show_frame=True):
        """
        Plot the robot arm in 3D.
        
        Args:
            theta1: Base rotation (degrees)
            theta2: Second joint angle (degrees)
            d3: Prismatic extension (meters)
            ax: Matplotlib 3D axis (creates new if None)
            show_frame: Whether to show coordinate frames
        
        Returns:
            Matplotlib axis object
        """
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
                'o-', linewidth=3, markersize=6, color='steelblue', label='Link 1')
        
        # Plot Link 2
        link2_positions = positions[link1_end_idx:link2_end_idx+1]
        ax.plot(link2_positions[:, 0], link2_positions[:, 1], link2_positions[:, 2], 
                'o-', linewidth=3, markersize=6, color='coral', label='Link 2')
        
        # Plot Prismatic Joint (d3)
        d3_positions = positions[link2_end_idx:d3_end_idx+1]
        ax.plot(d3_positions[:, 0], d3_positions[:, 1], d3_positions[:, 2], 
                'o-', linewidth=4, markersize=6, color='green', label='Prismatic (d3)')
        
        # Plot End Effector
        if len(self.end_effector_shape) > 0:
            ee_positions = positions[d3_end_idx:]
            ax.plot(ee_positions[:, 0], ee_positions[:, 1], ee_positions[:, 2], 
                    's-', linewidth=2, markersize=8, color='red', label='End Effector')
        
        # Plot joints
        ax.scatter(positions[0, 0], positions[0, 1], positions[0, 2], 
                  c='red', s=150, marker='o', label='Base', zorder=5)
        # ax.scatter(positions[-1, 0], positions[-1, 1], positions[-1, 2], 
        #           c='darkred', s=150, marker='s', label='Tool Point', zorder=5)
        
        # Plot coordinate frames
        if show_frame:
            self._plot_frame(ax, np.eye(4), scale=0.3, label='Base')
        
        # Set plot properties
        max_reach = self.L1 + self.L2 + self.d3_max + 0.5
        ax.set_xlim([-max_reach, max_reach])
        ax.set_ylim([-max_reach, max_reach])
        ax.set_zlim([0, max_reach*1.5])
        
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
    
    def interactive_plot(self):
        """Create interactive plot with sliders for joint control."""
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
    
    def workspace_analysis(self, n_samples=20):
        """
        Visualize the robot's workspace by sampling joint configurations.
        
        Args:
            n_samples: Number of samples per joint
        """
        fig = plt.figure(figsize=(12, 5))
        
        # Generate samples
        theta1_samples = np.linspace(*self.theta1_limits, n_samples)
        theta2_samples = np.linspace(*self.theta2_limits, n_samples)
        d3_samples = np.linspace(*self.d3_limits, n_samples)
        
        workspace_points = []
        for th1 in theta1_samples:
            for th2 in theta2_samples:
                for d3 in d3_samples:
                    pos = self.forward_kinematics(th1, th2, d3)
                    workspace_points.append(pos[-1])
        
        workspace_points = np.array(workspace_points)
        
        # 3D workspace plot
        ax1 = fig.add_subplot(121, projection='3d')
        ax1.scatter(workspace_points[:, 0], workspace_points[:, 1], 
                   workspace_points[:, 2], c=workspace_points[:, 2], 
                   cmap='viridis', alpha=0.3, s=1)
        ax1.set_xlabel('X (m)')
        ax1.set_ylabel('Y (m)')
        ax1.set_zlabel('Z (m)')
        ax1.set_title('3D Workspace')
        
        # Top view (XY plane)
        ax2 = fig.add_subplot(122)
        ax2.scatter(workspace_points[:, 0], workspace_points[:, 1], 
                   c=workspace_points[:, 2], cmap='viridis', alpha=0.3, s=1)
        ax2.set_xlabel('X (m)')
        ax2.set_ylabel('Y (m)')
        ax2.set_title('Workspace - Top View (XY)')
        ax2.axis('equal')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def animate_trajectory(self, trajectory, interval=50):
        """
        Animate robot following a trajectory.
        
        Args:
            trajectory: List of [theta1, theta2, d3] configurations
            interval: Time between frames (ms)
        """
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        def update_frame(frame):
            ax.cla()
            theta1, theta2, d3 = trajectory[frame]
            self.plot_robot(theta1, theta2, d3, ax=ax, show_frame=False)
            ax.set_title(f'Frame {frame+1}/{len(trajectory)}\n' + 
                        f'θ1={theta1:.1f}°, θ2={theta2:.1f}°, d3={d3:.2f}m')
        
        anim = FuncAnimation(fig, update_frame, frames=len(trajectory), 
                           interval=interval, repeat=True)
        plt.show()
        return anim


# Example usage and demonstrations
if __name__ == "__main__":
    print("RRP Robotic Arm Visualization Toolbox")
    print("=" * 50)
    
    # Example: Custom link shapes
    # Link 1: Go up 5m in Z, then 5m in X
    link_1 = [ (5, 0, 0), (0, 0, 5) ]
    
    # Link 2: Go 3m in X, then 2m in Y
    link_2 = [(3, 0, 0)]
    
    # End effector: Small gripper shape
    end_effector = [(0, 0, 0)]
    
    # Create robot with custom shapes
    robot = RRPRobot(
        link_1_shape=link_1,
        link_2_shape=link_2,
        end_effector_shape=end_effector,
        d3_max=2.0
    )
    
    print(f"\nRobot Configuration:")
    print(f"  Link 1 shape: {link_1}")
    print(f"  Link 1 total length: {robot.L1:.2f}m")
    print(f"  Link 2 shape: {link_2}")
    print(f"  Link 2 total length: {robot.L2:.2f}m")
    print(f"  End effector shape: {end_effector}")
    
    # Example 1: Interactive control
    print("\n1. Launching interactive control...")
    print("   Use sliders to control joint angles and prismatic extension")
    robot.interactive_plot()
    
    # Example 2: Workspace analysis (uncomment to use)
    # print("\n2. Computing workspace...")
    # robot.workspace_analysis(n_samples=15)
    
    # Example 3: Trajectory animation (uncomment to use)
    # print("\n3. Creating trajectory animation...")
    # trajectory = []
    # for t in np.linspace(0, 2*np.pi, 50):
    #     theta1 = 45 * np.sin(t)
    #     theta2 = 30 * np.cos(t)
    #     d3 = 0.4 + 0.2 * np.sin(2*t)
    #     trajectory.append([theta1, theta2, d3])
    # robot.animate_trajectory(trajectory)
    
    print("\n" + "=" * 50)
    print("Toolbox demonstration complete!")