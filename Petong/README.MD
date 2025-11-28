# RRP Robotic Arm Visualization Toolbox

A comprehensive Python toolbox for simulating, controlling, and visualizing an RRP (Revolute-Revolute-Prismatic) robotic arm with advanced kinematics calculations and interactive 3D visualization.

## Features

### Core Kinematics
- **Forward Kinematics**: Calculate end-effector position from joint configurations
- **Inverse Kinematics**: Compute joint angles needed to reach target Cartesian positions
- **Differential Kinematics**: Calculate velocities using Jacobian matrices
- **Singularity Detection**: Identify and check for singular configurations
- **Workspace Analysis**: Validate reachability of target positions

### Visualization & Control
- **Interactive 3D Visualization**: Real-time 3D plotting of robot configuration
- **Interactive Sliders**: Control each joint in real-time with visual feedback
- **Trajectory Animation**: Smooth animated motion along Cartesian waypoints
- **End-Effector Path Tracking**: Visual display of the end-effector trajectory during animation
- **Time-Based Control**: Precise time-based trajectory execution with FPS control

### Mathematical Operations
- Custom matrix operations (multiplication, inversion, determinant)
- DH (Denavit-Hartenberg) transformation matrices
- Jacobian matrix computation and analysis

## Installation

### Requirements
```bash
pip install numpy matplotlib
```

### Dependencies
- `numpy` - Numerical computations and array operations
- `matplotlib` - 3D visualization and animation
- `math` - Mathematical operations

## Robot Configuration

The RRP robot consists of:
- **Joint 1 (θ1)**: Revolute joint at base (rotation about z-axis), range: -180° to 180°
- **Joint 2 (θ2)**: Revolute joint (rotation about y-axis), range: 0° to 180°
- **Joint 3 (d3)**: Prismatic joint (linear extension), default range: 0 to 11.0 meters

### Link Structure

The robot is defined by link segments with custom shapes:
- **Link 1**: Vertical revolute link (rotates around z-axis)
- **Link 2**: Articulated link (controlled by θ2)
- **Prismatic Joint**: Linear extension (d3)
- **End Effector**: Final link with marker

## Quick Start

### Basic Setup

```python
from Visualization_Optimize import RRPRobot

# Define link shapes (x, y, z) segments for visualization
link_1 = [(5, 0, 0), (0, 0, 5)]    # Link 1: moves in X then Z
link_2 = [(3, 0, 0)]                # Link 2: extends in X
end_effector = [(0, 0, 0.5)]        # End effector: small Z offset

# Create robot instance
robot = RRPRobot(
    link_1_shape=link_1,
    link_2_shape=link_2,
    end_effector_shape=end_effector,
    d3_max=11.0                      # Maximum prismatic extension
)

print(f"Link 1 length: {robot.L1:.2f}m")
print(f"Link 2 length: {robot.L2:.2f}m")
print(f"End effector length: {robot.ee_length:.2f}m")
```

## Usage Examples

### 1. Forward Kinematics

Calculate end-effector position from joint configuration:

```python
# Define joint configuration [θ1, θ2, d3]
theta1, theta2, d3 = 45, 60, 5.0  # 45°, 60°, 5.0m

# Compute forward kinematics
positions = robot.forward_kinematics(theta1, theta2, d3)
end_effector_pos = positions[-1]  # Last position is end-effector

print(f"End-effector position: ({end_effector_pos[0]:.2f}, "
      f"{end_effector_pos[1]:.2f}, {end_effector_pos[2]:.2f})")
```

### 2. Inverse Kinematics

Calculate joint configuration to reach target position:

```python
# Define target Cartesian position [x, y, z]
target_position = (3, 3, 8)

# Compute inverse kinematics
theta1, theta2, d3 = robot.inverse_kinematics(target_position[0], 
                                              target_position[1], 
                                              target_position[2])

print(f"Joint configuration: θ1={theta1:.2f}°, θ2={theta2:.2f}°, d3={d3:.2f}m")
```

### 3. Static Plot

Create a single 3D plot of the robot:

```python
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Plot robot at specific configuration
robot.plot_robot(theta1=45, theta2=60, d3=5, ax=ax, show_frame=True)

plt.show()
```

### 4. Interactive Control

Launch interactive GUI with real-time slider control:

```python
# Interactive visualization with sliders
robot.interactive_plot()
```

**Features:**
- Real-time 3D visualization
- Three sliders for θ1, θ2, and d3 control
- Automatic limit enforcement
- Live configuration display in title
- Smooth animation as you adjust sliders

### 5. Trajectory Animation - Cartesian Space

Animate robot following Cartesian waypoints:

```python
# Define waypoints in Cartesian space [x, y, z]
position_trajectory = [
    [3, 3, 8],
    [3, -3, 10],
    [-3, -3, 8],
    [-3, 3, 10],
    [3, 3, 8],
]

# Animate trajectory over 10 seconds at 30 FPS
robot.animate_trajectory(
    trajectory=position_trajectory,
    total_time=10,           # 10 seconds total
    trajectory_type='position',  # Cartesian space
    fps=30                   # 30 frames per second
)
```

**Display includes:**
- Real-time joint angles
- Current end-effector position
- Elapsed time / Total time
- End-effector path visualization (red dashed line)
- Current position marker (yellow star)

### 6. Trajectory Animation - Joint Space

Animate robot following joint configuration waypoints:

```python
# Define waypoints in joint space [θ1, θ2, d3]
joint_trajectory = [
    [0, 30, 1],
    [45, 60, 5],
    [90, 45, 8],
    [45, 90, 3],
    [0, 30, 1],
]

# Animate trajectory
robot.animate_trajectory(
    trajectory=joint_trajectory,
    total_time=15,           # 15 seconds total
    trajectory_type='joint', # Joint space
    fps=30
)
```

### 7. Trajectory Interpolation

Generate smooth interpolated trajectory without animation:

```python
# Define waypoints
waypoints = [
    (0, 30, 1),
    (45, 60, 5),
    (90, 45, 8),
]

# Generate 10-second trajectory at 30 FPS
trajectory, time_stamps = robot.interpolate_trajectory(
    waypoints=waypoints,
    total_time=10,
    fps=30
)

# Use trajectory data for further processing
print(f"Generated {len(trajectory)} frames")
for i, (config, time) in enumerate(zip(trajectory[::10], time_stamps[::10])):
    theta1, theta2, d3 = config
    print(f"Frame {i}: t={time:.2f}s, θ1={theta1:.1f}°, θ2={theta2:.1f}°, d3={d3:.2f}m")
```

## Visualization Features

### Color Scheme

- **Link 1**: Steel Blue
- **Link 2**: Coral
- **Prismatic Joint (d3)**: Green
- **End Effector**: Red (with square marker)
- **Base Joint**: Red sphere
- **End-Effector Path**: Red dashed line
- **Current Position**: Yellow star marker

### Viewing Angle

- **Elevation**: 20 degrees (viewing from slightly above)
- **Azimuth**: 45 degrees (angled view)
- Full 3D rotation support with mouse

### Coordinate Frame

- **X-axis**: Forward direction (red)
- **Y-axis**: Left direction (green)
- **Z-axis**: Upward direction (blue)
- **Origin**: Robot base at (0, 0, 0)

## API Reference

### Main Methods

#### `forward_kinematics(theta1, theta2, d3)`
Compute all joint positions from configuration.

**Parameters:**
- `theta1` (float): First joint angle in degrees
- `theta2` (float): Second joint angle in degrees
- `d3` (float): Prismatic joint extension in meters

**Returns:** numpy array of positions for all joints and end-effector

#### `inverse_kinematics(x, y, z)`
Compute joint configuration for target position.

**Parameters:**
- `x, y, z` (float): Target Cartesian position in meters

**Returns:** Tuple (theta1, theta2, d3) - Joint configuration in degrees and meters

#### `plot_robot(theta1=0, theta2=0, d3=0, ax=None, show_frame=True)`
Plot robot in current configuration.

**Parameters:**
- `theta1`, `theta2`, `d3` (float): Joint values
- `ax` (Axes3D): Matplotlib 3D axis (optional, creates new if None)
- `show_frame` (bool): Display coordinate frame (default: True)

**Returns:** Axes3D - The 3D axes object

#### `interactive_plot()`
Launch interactive GUI with sliders for joint control.

**Features:**
- Real-time 3D rendering
- Slider adjustment with valstep=1 for angles, 0.01 for d3
- Automatic limit enforcement
- Title updates with current configuration

#### `animate_trajectory(trajectory, total_time=None, trajectory_type='joint', fps=30)`
Animate robot following trajectory with visualization.

**Parameters:**
- `trajectory` (list): List of waypoints
- `total_time` (float): Total animation time in seconds (default: len(trajectory) seconds)
- `trajectory_type` (str): 'joint' or 'position'
- `fps` (int): Frames per second (default: 30)

**Returns:** FuncAnimation object

#### `interpolate_trajectory(waypoints, total_time, fps=30)`
Generate smooth interpolated trajectory.

**Parameters:**
- `waypoints` (list): List of start and end configurations
- `total_time` (float): Total interpolation time in seconds
- `fps` (int): Frames per second (default: 30)

**Returns:** Tuple (trajectory, time_stamps) - Lists of interpolated configs and times

### Properties

- `L1` (float): Length of Link 1 (calculated from shape)
- `L2` (float): Length of Link 2 (calculated from shape)
- `ee_length` (float): Length of end effector (calculated from shape)
- `d3_max` (float): Maximum prismatic extension
- `theta1_limits` (tuple): Angle limits for θ1
- `theta2_limits` (tuple): Angle limits for θ2
- `d3_limits` (tuple): Extension limits for d3

## Trajectory Types

### Position-Based Trajectory (`trajectory_type='position'`)

**Input:** Cartesian positions [x, y, z]
- Automatically converts positions to joint configurations
- Better for specifying desired end-effector locations
- May result in different joint paths for the same position

```python
robot.animate_trajectory(
    trajectory=[[3, 3, 8], [5, 5, 10]],
    total_time=5,
    trajectory_type='position'
)
```

## Complete Example

```python
from Visualization_Optimize import RRPRobot

# Create robot
link_1 = [(5, 0, 0), (0, 0, 5)]
link_2 = [(3, 0, 0)]
end_effector = [(0, 0, 0.5)]

robot = RRPRobot(
    link_1_shape=link_1,
    link_2_shape=link_2,
    end_effector_shape=end_effector,
    d3_max=11.0
)

print("RRP Robotic Arm - Visualization Demo")
print("=" * 50)
print(f"Link 1 length: {robot.L1:.2f}m")
print(f"Link 2 length: {robot.L2:.2f}m")
print(f"End effector length: {robot.ee_length:.2f}m")

# 1. Interactive control
print("\n1. Interactive Control - Use sliders to manipulate robot")
robot.interactive_plot()

# 2. Position trajectory animation
print("\n2. Position Trajectory Animation")
position_trajectory = [
    [3, 3, 8],
    [3, -3, 10],
    [-3, -3, 8],
    [-3, 3, 10],
    [3, 3, 8],
]

robot.animate_trajectory(
    trajectory=position_trajectory,
    total_time=10,
    trajectory_type='position',
    fps=30
)
```

## Troubleshooting

### Issue: Plot doesn't display

**Cause:** Matplotlib backend not set correctly
**Solution:**
```python
import matplotlib
matplotlib.use('TkAgg')  # Use TkAgg for interactive windows
import matplotlib.pyplot as plt
```

### Issue: Inverse kinematics fails with "unreachable" error

**Cause:** Target position is outside workspace
**Solution:**
```python
# Check if position is reachable first (simplified check)
# If d3 calculation results in negative or > d3_max, it's unreachable
# Try positions closer to robot base or within reachable zone
```

### Issue: Animation is too fast or too slow

**Cause:** fps parameter affects animation speed
**Solution:**
```python
# Slower animation (more frames per second = more frames total)
robot.animate_trajectory(trajectory, total_time=10, fps=60)

# Faster animation (fewer frames per second)
robot.animate_trajectory(trajectory, total_time=10, fps=15)
```

### Issue: Memory issues with long animations

**Cause:** Too many frames generated
**Solution:**
```python
# Use higher total_time to reduce frame count
robot.animate_trajectory(trajectory, total_time=20, fps=30)
# Instead of: total_time=5, fps=30 (which creates 150 frames vs 600)
```

### Issue: Robot configuration doesn't match expected position

**Cause:** Inverse kinematics may have multiple solutions
**Solution:**
```python
# Use forward kinematics to verify the actual position
positions = robot.forward_kinematics(theta1, theta2, d3)
end_effector = positions[-1]
print(f"Actual position: {end_effector}")
```

## Performance Tips

### Optimization Guidelines

| Task | Recommendation |
|------|-----------------|
| Quick visualization | Use interactive_plot() |
| Static plots | Use plot_robot() with low resolution |
| Smooth animation | fps=30, total_time > 3 seconds |
| Detailed trajectory | fps=60, total_time > 5 seconds |
| Fast preview | fps=15, total_time = number of waypoints |

### Best Practices

1. **Start with interactive_plot()** to understand robot behavior
2. **Use position-based trajectories** for end-effector control
3. **Set total_time >= 5 seconds** for smooth animations
4. **Use fps=30** as default for good balance


