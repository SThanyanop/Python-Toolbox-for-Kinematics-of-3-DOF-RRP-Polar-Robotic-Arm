import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.widgets import Slider
import warnings
warnings.filterwarnings('ignore')


class RRPRobot:
    """
    - Joint 1 (θ1): Revolute joint
    - Joint 2 (θ2): Revolute joint 
    - Joint 3 (d3): Prismatic joint
    """
    
    def __init__(self, L1=1.0, L2=1.0, d3_max=1.0):
        """
        Robot parameters
            L1: Length of link 1 (m)
            L2: Length of link 2 (m)
            d3_max: Maximum extension of prismatic joint (m)
        """
        self.L1 = L1
        self.L2 = L2
        self.d3_max = d3_max
        
        # Joint limits
        self.theta1_limits = (-180, 180)  # degrees
        self.theta2_limits = (0, 180)     # degrees (0° = vertical/parallel to Z, 90° = horizontal)
        self.d3_limits = (0, d3_max)      # meters
    
    def dh_transform(self, theta, d, a, alpha):
        """
            theta: Joint angle (rad)
            d: Link offset
            a: Link length
            alpha: Link twist (rad)
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
        # Convert to radians
        th1 = np.radians(theta1)
        th2 = np.radians(theta2)
        
        # Base pos
        positions = [np.array([0, 0, 0])]
        
        # Joint 1
        x1 = 0
        y1 = 0
        z1 = self.L1
        positions.append(np.array([x1, y1, z1]))
        
        # Joint 2
        x2 = self.L2 * np.cos(th1)
        y2 = self.L2 * np.sin(th1)
        z2 = z1
        positions.append(np.array([x2, y2, z2]))
        
        # Joint 3
        direction = np.array([
            np.sin(th2) * np.cos(th1),
            np.sin(th2) * np.sin(th1),
            np.cos(th2)
        ])
        x3 = x2 + d3 * direction[0]
        y3 = y2 + d3 * direction[1]
        z3 = z2 + d3 * direction[2]
        positions.append(np.array([x3, y3, z3]))
        
        return np.array(positions)
    
    def plot_robot(self, theta1=0, theta2=0, d3=0, ax=None, show_frame=True):
        """
            theta1: Base rotation (degrees)
            theta2: Second joint angle (degrees)
            d3: Prismatic extension (m)
            ax: Matplotlib 3D axis (creates new if None)
            show_frame: Whether to show coordinate frames
        """
        if ax is None:
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection='3d')
        
        # Get joint pos
        positions = self.forward_kinematics(theta1, theta2, d3)
        
        # Plot links
        ax.plot(positions[:, 0], positions[:, 1], positions[:, 2], 
                'o-', linewidth=3, markersize=8, color='steelblue', label='Robot Arm')
        
        # Plot joints
        ax.scatter(positions[0, 0], positions[0, 1], positions[0, 2], 
                  c='red', s=100, marker='o', label='Base')
        ax.scatter(positions[-1, 0], positions[-1, 1], positions[-1, 2], 
                  c='green', s=100, marker='s', label='End Effector')
        
        # Plot coordinate frames
        if show_frame:
            self._plot_frame(ax, np.eye(4), scale=0.3, label='Base')
        
        # Set plot properties
        max_reach = self.L1 + self.L2 + self.d3_max
        ax.set_xlim([-max_reach, max_reach])
        ax.set_ylim([-max_reach, max_reach])
        ax.set_zlim([0, max_reach*1.5])
        
        ax.set_xlabel('X (m)', fontsize=10)
        ax.set_ylabel('Y (m)', fontsize=10)
        ax.set_zlabel('Z (m)', fontsize=10)
        ax.set_title(f'RRP Robot Configuration\nθ1={theta1:.1f}°, θ2={theta2:.1f}°, d3={d3:.2f}m', 
                    fontsize=12)
        ax.legend()
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
    
if __name__ == "__main__":
    print("RRP Robotic Arm Visualization Toolbox")
    print("=" * 50)
    
    # Create robot instance
    robot = RRPRobot(L1=1.0, L2=1.0, d3_max=1.0)
    
    print("\n1. Launching interactive control...")
    print("   Use sliders to control joint angles and prismatic extension")
    robot.interactive_plot()
    print("\n" + "=" * 50)
    print("Toolbox demonstration complete!")