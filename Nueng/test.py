"""
Comprehensive example demonstrating all features of RRPToolbox and RRPVisualization.
This example showcases:
1. Forward and inverse kinematics
2. Workspace generation and analysis
3. Singularity detection
4. Static robot plotting
5. Interactive workspace visualization
6. Interactive robot control
7. Trajectory animation (both joint and Cartesian space)
"""

from RRPToolbox import RRPToolbox
from RRPVisualization import RRPVisualization
import matplotlib.pyplot as plt


def main():
    print("=" * 70)
    print("RRP ROBOT MANIPULATOR - COMPREHENSIVE DEMONSTRATION")
    print("=" * 70)
    
    # Define robot parameters
    link_params = [
        [(5, 0, 0), (0, 0, 5)],  # Link 1: horizontal and vertical segments
        [(3, 0, 0)],              # Link 2: horizontal segment
        [(0, 0, 0.5)]             # End Effector: small vertical offset
    ]
    
    joint_limits = [
        (-180, 180),  # theta1 limits (degrees) - full rotation
        (0, 180),     # theta2 limits (degrees) - hemisphere
        (0, 11.0)     # d3 limits (meters) - prismatic extension
    ]
    
    # Create robot instance
    print("\n" + "=" * 70)
    print("1. INITIALIZING ROBOT")
    print("=" * 70)
    toolbox = RRPToolbox(link_params, joint_limits)
    
    # Create visualization instance
    print("\n" + "=" * 70)
    print("2. INITIALIZING VISUALIZATION")
    print("=" * 70)
    viz = RRPVisualization(toolbox)
    print("✓ Visualization system ready")
    
    # Demonstrate forward kinematics
    print("\n" + "=" * 70)
    print("3. FORWARD KINEMATICS DEMO")
    print("=" * 70)
    test_configs = [
        [0, 45, 5],
        [90, 90, 7],
        [180, 135, 3],
    ]
    
    for config in test_configs:
        pos = toolbox.forward_kinematics(config, update_state=False)
        print(f"Config [{config[0]:6.1f}°, {config[1]:6.1f}°, {config[2]:5.2f}m] "
              f"→ Position ({pos[0]:6.2f}, {pos[1]:6.2f}, {pos[2]:6.2f})")
    
    # Demonstrate inverse kinematics
    print("\n" + "=" * 70)
    print("4. INVERSE KINEMATICS DEMO")
    print("=" * 70)
    test_positions = [
        (5, 5, 10),
        (-3, 3, 8),
        (0, 4, 12),
    ]
    
    for pos in test_positions:
        try:
            config = toolbox.inverse_kinematics(pos, validate=True)
            print(f"Position ({pos[0]:5.1f}, {pos[1]:5.1f}, {pos[2]:5.1f}) "
                  f"→ Config [{config[0]:6.1f}°, {config[1]:6.1f}°, {config[2]:5.2f}m]")
        except ValueError as e:
            print(f"Position ({pos[0]:5.1f}, {pos[1]:5.1f}, {pos[2]:5.1f}) → ERROR: {e}")
    
    # Workspace generation
    print("\n" + "=" * 70)
    print("5. WORKSPACE ANALYSIS")
    print("=" * 70)
    print("Generating workspace points...")
    workspace_points = viz.get_workspace(theta1_samples=8, theta2_samples=8, d3_samples=5)
    print(f"✓ Generated {len(workspace_points)} workspace boundary points")
    
    print(f"\nSample workspace points:")
    for i in range(min(5, len(workspace_points))):
        p = workspace_points[i]
        print(f"  Point {i+1}: ({p[0]:6.2f}, {p[1]:6.2f}, {p[2]:6.2f})")
    
    # Singularity detection
    print("\n" + "=" * 70)
    print("6. SINGULARITY DETECTION")
    print("=" * 70)
    print("Scanning for singularities...")
    sing_positions, sing_configs = viz.find_singularities(
        theta1_samples=15,
        theta2_samples=15,
        d3_samples=8
    )
    print(f"✓ Found {len(sing_positions)} singularity configurations")
    
    if sing_positions:
        print(f"\nSample singularities:")
        for i in range(min(3, len(sing_configs))):
            config, pos = sing_configs[i]
            print(f"  Config [{config[0]:6.1f}°, {config[1]:6.1f}°, {config[2]:5.2f}m] "
                  f"→ ({pos[0]:6.2f}, {pos[1]:6.2f}, {pos[2]:6.2f})")
    
    # Robot status
    print("\n" + "=" * 70)
    print("7. ROBOT STATUS")
    print("=" * 70)
    toolbox.update_current_config([45, 90, 5])
    toolbox.print_status()
    
    # Interactive demonstrations
    print("\n" + "=" * 70)
    print("8. INTERACTIVE DEMONSTRATIONS")
    print("=" * 70)
    print("\nAvailable demos:")
    print("  A. Static robot plot")
    print("  B. Interactive workspace visualization")
    print("  C. Interactive robot control with sliders")
    print("  D. Cartesian trajectory animation")
    print("  E. Joint trajectory animation")
    print("\nSelect a demo (or 'all' for all demos, 'skip' to skip):")
    
    # For automated demo, let's show option A (static plot)
    print("\nRunning Demo A: Static Robot Plot")
    print("-" * 70)
    
    fig = plt.figure(figsize=(12, 5))
    
    # Plot three different configurations
    configs = [
        ([0, 45, 5], 'Upright'),
        ([90, 90, 7], 'Extended'),
        ([180, 135, 3], 'Retracted'),
    ]
    
    for idx, (config, label) in enumerate(configs):
        ax = fig.add_subplot(1, 3, idx+1, projection='3d')
        viz.plot_robot(config[0], config[1], config[2], ax=ax)
        ax.set_title(f'{label} Configuration')
    
    plt.tight_layout()
    plt.show()
    
    print("\n✓ Static robot plot displayed")
    
    # Option to run other demos
    print("\n" + "=" * 70)
    print("9. ADDITIONAL DEMO OPTIONS")
    print("=" * 70)
    print("\nTo run additional demos, uncomment the desired section below:")
    print("""
    # DEMO B: Interactive Workspace Visualization
    # Shows 3D workspace with singularities, toggleable vertices/faces, and camera controls
    # viz.plot_workspace_3d(theta1_samples=10, theta2_samples=10, d3_samples=5)
    
    # DEMO C: Interactive Robot Control
    # Use sliders to control joint angles in real-time
    # viz.interactive_plot()
    
    # DEMO D: Cartesian Trajectory Animation
    # Animate robot following Cartesian waypoints
    # position_trajectory = [
    #     [3, 3, 8],
    #     [3, -3, 10],
    #     [-3, -3, 8],
    #     [-3, 3, 10],
    #     [3, 3, 8],
    # ]
    # viz.animate_trajectory(position_trajectory, total_time=10, trajectory_type='position', fps=30)
    
    # DEMO E: Joint Trajectory Animation
    # Animate robot following joint space waypoints
    # joint_trajectory = [
    #     [0, 45, 5],
    #     [90, 90, 8],
    #     [180, 135, 5],
    #     [90, 45, 3],
    #     [0, 45, 5],
    # ]
    # viz.animate_trajectory(joint_trajectory, total_time=8, trajectory_type='joint', fps=30)
    """)
    
    position_trajectory = [
        [3, 3, 8],
        [3, -3, 10],
        [-3, -3, 8],
        [-3, 3, 10],
        [3, 3, 8],
    ]
    viz.animate_trajectory(position_trajectory, total_time=10, trajectory_type='position', fps=30)
    
    print("\n" + "=" * 70)
    print("DEMONSTRATION COMPLETE")
    print("=" * 70)
    print("\nSummary:")
    print(f"  • Workspace points: {len(workspace_points)}")
    print(f"  • Singularities found: {len(sing_positions)}")
    print(f"  • Joint limits: θ1=[{joint_limits[0][0]}, {joint_limits[0][1]}]°, "
          f"θ2=[{joint_limits[1][0]}, {joint_limits[1][1]}]°, "
          f"d3=[{joint_limits[2][0]}, {joint_limits[2][1]}]m")
    
    print("\nFor full interactive experience, run individual demo sections!")
    viz.plot_workspace_3d()


if __name__ == "__main__":
    main()