"""
Example usage of RRPToolbox and RRPVisualization classes.
Demonstrates the separation of concerns between kinematics and visualization/workspace analysis.
"""

from RRPToolbox import RRPToolbox
from RRPVisualization import RRPVisualization


def main():
    # Define link parameters and joint limits
    link_params = [
        [(5, 0, 0), (0, 0, 5)],  # Link 1
        [(3, 0, 0)],              # Link 2
        [(0, 0, 0)]               # End Effector
    ]
    joint_limits = [
        (0, 90),     # theta1 limits (degrees)
        (0, 180),    # theta2 limits (degrees)
        (0, 5)       # d3 limits (units)
    ]
    
    # Create toolbox instance
    print("=" * 60)
    print("Creating RRP Toolbox")
    print("=" * 60)
    toolbox = RRPToolbox(link_params, joint_limits)
    
    # Create visualization instance
    print("\n" + "=" * 60)
    print("Creating Visualization Analyzer")
    print("=" * 60)
    viz = RRPVisualization(toolbox)
    print("Visualization analyzer created successfully!")
    
    # Example 1: Forward kinematics
    print("\n" + "=" * 60)
    print("Example 1: Forward Kinematics")
    print("=" * 60)
    joint_config = [45, 90, 4]
    position = toolbox.set_pose_from_joints(joint_config)
    
    # Example 2: Inverse kinematics
    print("\n" + "=" * 60)
    print("Example 2: Inverse Kinematics")
    print("=" * 60)
    target = (5, 5, 8)
    try:
        joints = toolbox.set_pose_from_cartesian(target)
    except ValueError as e:
        print(f"Error: {e}")
    
    # Example 3: Generate workspace points
    print("\n" + "=" * 60)
    print("Example 3: Workspace Generation")
    print("=" * 60)
    workspace_points = viz.get_workspace(theta1_samples=5, theta2_samples=5, d3_samples=3)
    print(f"Generated {len(workspace_points)} workspace points")
    print(f"Sample points (first 5):")
    for i, point in enumerate(workspace_points[:5]):
        print(f"  Point {i+1}: ({point[0]:.3f}, {point[1]:.3f}, {point[2]:.3f})")
    
    # Example 4: Find singularities
    print("\n" + "=" * 60)
    print("Example 4: Singularity Detection")
    print("=" * 60)
    print("Finding singularities...")
    sing_positions, sing_configs = viz.find_singularities(
        theta1_samples=10, 
        theta2_samples=10, 
        d3_samples=5
    )
    print(f"Found {len(sing_positions)} singularity positions")
    if sing_positions:
        print(f"Sample singularity positions (first 3):")
        for i, (config, pos) in enumerate(sing_configs[:3]):
            print(f"  Config: θ1={config[0]:.1f}°, θ2={config[1]:.1f}°, d3={config[2]:.2f}")
            print(f"  Position: ({pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f})")
    
    # Example 5: Check current configuration status
    print("\n" + "=" * 60)
    print("Example 5: Robot Status")
    print("=" * 60)
    toolbox.print_status()
    
    # Example 6: Visualize workspace (requires matplotlib)
    print("\n" + "=" * 60)
    print("Example 6: Workspace Visualization")
    print("=" * 60)
    print("Plotting workspace in 3D (this will open a matplotlib window)...")
    viz.plot_workspace_3d(theta1_samples=10, theta2_samples=10, d3_samples=5)


if __name__ == "__main__":
    main()