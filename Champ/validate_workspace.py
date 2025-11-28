import numpy as np
import sys
from pathlib import Path
from optimize_Toolbox_Visualization import RRPRobot

"""
Workspace Validation Script for 3-DOF RRP Polar Robotic Arm

This script validates the plotted workspace by verifying that the boundary vertices
correspond to the actual forward kinematics calculations at joint limits.

It checks vertices at:
- q1_max, q1_min (revolute joint 1 limits)
- q2_max, q2_min (revolute joint 2 limits)  
- d3_max, d3_min (prismatic joint 3 limits)
"""


# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))



def forward_kinematics(q1, q2, d3, L1=1.0, L2=1.0):
    """
    Calculate forward kinematics for RRP robot
    
    Args:
        q1: Joint 1 angle (radians)
        q2: Joint 2 angle (radians)
        d3: Prismatic joint 3 extension
        L1: Length of link 1
        L2: Length of link 2
    
    Returns:
        x, y, z: End effector position
    """
    x = (L1 * np.cos(q2) + L2 + d3) * np.cos(q1)
    y = (L1 * np.cos(q2) + L2 + d3) * np.sin(q1)
    z = L1 * np.sin(q2)
    
    return x, y, z


def validate_workspace_vertices(robot):
    """
    Validate workspace by checking forward kinematics at vertex joint configurations
    
    Args:
        robot: RRPRobot instance with defined joint limits
    """
    # Get joint limits from robot
    q1_limits = [robot.q1_min, robot.q1_max]
    q2_limits = [robot.q2_min, robot.q2_max]
    d3_limits = [robot.d3_min, robot.d3_max]
    
    print("=" * 80)
    print("WORKSPACE VALIDATION - BOUNDARY VERTICES")
    print("=" * 80)
    print(f"\nRobot Parameters:")
    print(f"  L1 = {robot.L1}, L2 = {robot.L2}")
    print(f"\nJoint Limits:")
    print(f"  q1: [{np.degrees(robot.q1_min):.2f}°, {np.degrees(robot.q1_max):.2f}°]")
    print(f"  q2: [{np.degrees(robot.q2_min):.2f}°, {np.degrees(robot.q2_max):.2f}°]")
    print(f"  d3: [{robot.d3_min:.3f}, {robot.d3_max:.3f}]")
    print("\n" + "=" * 80)
    
    vertices = []
    
    # Generate all combinations of joint limit vertices (2^3 = 8 vertices)
    for q1 in q1_limits:
        for q2 in q2_limits:
            for d3 in d3_limits:
                x, y, z = forward_kinematics(q1, q2, d3, robot.L1, robot.L2)
                vertices.append({
                    'q1': q1, 'q2': q2, 'd3': d3,
                    'x': x, 'y': y, 'z': z
                })
    
    # Print results
    print("\nVertex Configurations and End Effector Positions:")
    print("-" * 80)
    print(f"{'#':<4} {'q1 (deg)':<12} {'q2 (deg)':<12} {'d3':<10} {'x':<10} {'y':<10} {'z':<10}")
    print("-" * 80)
    
    for i, v in enumerate(vertices, 1):
        print(f"{i:<4} {np.degrees(v['q1']):>10.2f}° {np.degrees(v['q2']):>10.2f}° "
              f"{v['d3']:>8.3f}  {v['x']:>9.4f} {v['y']:>9.4f} {v['z']:>9.4f}")
    
    print("-" * 80)
    
    # Calculate workspace bounds from vertices
    x_coords = [v['x'] for v in vertices]
    y_coords = [v['y'] for v in vertices]
    z_coords = [v['z'] for v in vertices]
    
    print("\nCalculated Workspace Bounds from Vertices:")
    print(f"  X: [{min(x_coords):.4f}, {max(x_coords):.4f}]")
    print(f"  Y: [{min(y_coords):.4f}, {max(y_coords):.4f}]")
    print(f"  Z: [{min(z_coords):.4f}, {max(z_coords):.4f}]")
    print("=" * 80)
    
    return vertices


def main():
    """Main function to run workspace validation"""
    
    # Initialize robot with joint limits
    # Adjust these parameters according to your robot configuration
    robot = RRPRobot(
        L1=1.0,
        L2=1.0,
        q1_min=-180,
        q1_max=180,
        q2_min=0,
        q2_max=90,
        d3_min=0.0,
        d3_max=5.0
    )
    
    # Validate workspace vertices
    vertices = validate_workspace_vertices(robot)
    
    print("\n✓ Validation complete!")
    print("Compare these vertex positions with your plotted workspace boundaries.")
    

if __name__ == "__main__":
    main()