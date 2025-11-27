import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.widgets import Slider
import math

from Optimize_Toolbox import RRPToolbox


class RRPRobot(RRPToolbox):
    
    def __init__(self, link_params, joint_limits):
        super().__init__(link_params, joint_limits)
        self.d3_max = joint_limits[2][1]
    
    def get_link_positions(self, theta1, theta2, d3):
        T = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
        positions = [[T[0][3], T[1][3], T[2][3]]]
        
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
        
        for matrix in matrices:
            T = self.matrix_multiply(T, matrix)
            positions.append([T[0][3], T[1][3], T[2][3]])
        
        return positions
    
    def plot_robot(self, theta1=0, theta2=0, d3=0, ax=None, show_frame=True):
        if ax is None:
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection='3d')
        
        positions = self.get_link_positions(theta1, theta2, d3)
        
        def plot_segment(start, end, color, label, marker='o'):
            seg = positions[start:end+1]
            ax.plot([p[0] for p in seg], [p[1] for p in seg], [p[2] for p in seg], 
                   f'{marker}-', linewidth=3, markersize=8, color=color, label=label)
        
        plot_segment(0, 2, 'steelblue', 'Link 1')
        plot_segment(2, 5, 'coral', 'Link 2')
        plot_segment(5, 6, 'green', 'Prismatic (d3)')
        plot_segment(6, 7, 'red', 'End Effector', 's')
        
        ax.scatter(0, 0, 0, c='red', s=150, marker='o', label='Base', zorder=5)
        
        if show_frame:
            scale = 0.3
            ax.quiver(0, 0, 0, scale, 0, 0, color='r', arrow_length_ratio=0.2, linewidth=1.5)
            ax.quiver(0, 0, 0, 0, scale, 0, color='g', arrow_length_ratio=0.2, linewidth=1.5)
            ax.quiver(0, 0, 0, 0, 0, scale, color='b', arrow_length_ratio=0.2, linewidth=1.5)
        
        ee_pos = self.forward_kinematics([theta1, theta2, d3], update_state=False)
        max_reach = math.sqrt(sum(x**2 for x in ee_pos)) + 1.0
        
        ax.set_xlim([-max_reach, max_reach])
        ax.set_ylim([-max_reach, max_reach])
        ax.set_zlim([0, max_reach*1.5])
        ax.set_xlabel('X (m)', fontsize=10)
        ax.set_ylabel('Y (m)', fontsize=10)
        ax.set_zlabel('Z (m)', fontsize=10)
        ax.set_title(f'RRP Robot Configuration\nθ1={theta1:.1f}°, θ2={theta2:.1f}°, d3={d3:.2f}m', fontsize=12)
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
        
        return ax
    
    def interpolate_trajectory(self, waypoints, total_time, fps=30):
        num_waypoints = len(waypoints)
        trajectory = []
        time_stamps = []
        
        for i in range(num_waypoints - 1):
            t_start = i * total_time / (num_waypoints - 1)
            t_end = (i + 1) * total_time / (num_waypoints - 1)
            num_frames = int((t_end - t_start) * fps)
            
            start = waypoints[i]
            end = waypoints[i + 1]
            
            for j in range(num_frames):
                alpha = j / num_frames
                interpolated = tuple(start[k] + alpha * (end[k] - start[k]) for k in range(len(start)))
                trajectory.append(interpolated)
                time_stamps.append(t_start + alpha * (t_end - t_start))
        
        trajectory.append(waypoints[-1])
        time_stamps.append(total_time)
        
        return trajectory, time_stamps
    
    def interactive_plot(self):
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
        plt.subplots_adjust(bottom=0.25)
        
        theta1_init, theta2_init, d3_init = 0, 30, 0.3
        
        sliders = [
            Slider(plt.axes([0.15, 0.15, 0.65, 0.03]), 'θ1 (°)', *self.joint_limits[0], valinit=theta1_init, valstep=1),
            Slider(plt.axes([0.15, 0.10, 0.65, 0.03]), 'θ2 (°)', *self.joint_limits[1], valinit=theta2_init, valstep=1),
            Slider(plt.axes([0.15, 0.05, 0.65, 0.03]), 'd3 (m)', *self.joint_limits[2], valinit=d3_init, valstep=0.01)
        ]
        
        def update(val):
            ax.cla()
            self.plot_robot(sliders[0].val, sliders[1].val, sliders[2].val, ax=ax)
            fig.canvas.draw_idle()
        
        for slider in sliders:
            slider.on_changed(update)
        
        self.plot_robot(theta1_init, theta2_init, d3_init, ax=ax)
        plt.show()
    
    def animate_trajectory(self, trajectory, total_time=None, trajectory_type='joint', fps=30):
        if total_time is None:
            total_time = len(trajectory)
        
        if trajectory_type == 'position':
            joint_waypoints = []
            for pos in trajectory:
                try:
                    joint_waypoints.append(self.inverse_kinematics(tuple(pos), validate=True))
                except ValueError as e:
                    print(f"Skipping unreachable position {pos}: {e}")
            
            if len(joint_waypoints) == 0:
                print("Error: No reachable waypoints in trajectory")
                return None
            
            joint_trajectory, time_stamps = self.interpolate_trajectory(joint_waypoints, total_time, fps)
        else:
            joint_trajectory, time_stamps = self.interpolate_trajectory(trajectory, total_time, fps)
        
        ee_path = [self.forward_kinematics(list(jc), update_state=False) for jc in joint_trajectory]
        
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        def update_frame(frame):
            ax.cla()
            theta1, theta2, d3 = joint_trajectory[frame]
            self.plot_robot(theta1, theta2, d3, ax=ax, show_frame=False)
            
            ee_pos = ee_path[frame]
            path_slice = ee_path[:frame+1]
            ax.plot([p[0] for p in path_slice], [p[1] for p in path_slice], [p[2] for p in path_slice], 
                   'r--', linewidth=2, alpha=0.5, label='End Effector Path')
            ax.scatter(*ee_pos, s=100, c='yellow', marker='*')
            
            ax.set_title(f'Time: {time_stamps[frame]:.2f}s / {time_stamps[-1]:.2f}s\n' + 
                        f'θ1={theta1:.1f}°, θ2={theta2:.1f}°, d3={d3:.2f}m\n' +
                        f'End Effector: [{ee_pos[0]:.2f}, {ee_pos[1]:.2f}, {ee_pos[2]:.2f}]')
            ax.legend(loc='upper right')
        
        anim = FuncAnimation(fig, update_frame, frames=len(joint_trajectory), interval=1000/fps, repeat=True)
        plt.show()
        return anim


if __name__ == "__main__":
    print("RRP Robotic Arm - Time-Based Control")
    print("=" * 50)

    robot = RRPRobot(
        link_params=[[(5, 0, 0), (0, 0, 5)], [(3, 0, 0)], [(0, 0, 0.5)]],
        joint_limits=[(-180, 180), (0, 180), (0, 11.0)]
    )
    
    print("\n1. Launching interactive control...")
    robot.interactive_plot()

    print("\n2. Creating time-based position trajectory...")
    robot.animate_trajectory([[8, 0, 10], [6, 3, 8], [4, 4, 7], [8, 0, 10]], 
                           total_time=5, trajectory_type='position', fps=30)