# RRP Robot Workspace Visualization Manual

## Overview

This manual explains how to use the 3D workspace visualization tool for the 3-DOF RRP (Revolute-Revolute-Prismatic) robotic arm. The workspace shows all reachable positions of the end effector based on the robot's joint limits.

---

## Table of Contents

1. [Installation & Setup](#installation--setup)
2. [Quick Start](#quick-start)
3. [Using Optimize_Toolbox](#using-optimize_toolbox)
4. [Workspace Visualization Features](#workspace-visualization-features)
5. [Interactive Controls](#interactive-controls)
6. [Understanding the Visualization](#understanding-the-visualization)
7. [Examples](#examples)
8. [Troubleshooting](#troubleshooting)

---

## Installation & Setup

### Prerequisites

Ensure you have Python 3.7+ installed with the following packages:

```bash
pip install numpy matplotlib scipy
```

### Import the Toolbox

```python
from Optimize_Toolbox import RRPToolbox
```

---

## Quick Start

### Basic Setup

```python
from Optimize_Toolbox import RRPToolbox

# Define link parameters (x, y, z segments for each link)
link_params = [
    [(5, 0, 0), (0, 0, 5)],  # Link 1: segments
    [(3, 0, 0)],              # Link 2: segments
    [(0, 0, 0)]               # End Effector: segments
]

# Define joint limits
joint_limits = [
    (0, 90),      # theta1 limits (degrees)
    (0, 180),     # theta2 limits (degrees)
    (3, 5)        # d3 limits (meters)
]

# Create toolbox instance
toolbox = RRPToolbox(link_params, joint_limits)

# Plot workspace
toolbox.plot_workspace_3d(theta1_samples=25, theta2_samples=25, d3_samples=15)
```

### Parameters Explanation

- **link_params**: List of link segments as (x, y, z) tuples
  - Link 1: First revolute joint link
  - Link 2: Second revolute joint link  
  - End Effector: End effector segments

- **joint_limits**: Tuple of (min, max) for each joint
  - theta1: First revolute joint angle range (degrees)
  - theta2: Second revolute joint angle range (degrees)
  - d3: Prismatic joint extension range (meters)

---

## Using Optimize_Toolbox

### Method: `plot_workspace_3d(theta1_samples, theta2_samples, d3_samples)`

Generates and displays the 3D workspace visualization.

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `theta1_samples` | int | 10 | Number of samples for first revolute joint |
| `theta2_samples` | int | 10 | Number of samples for second revolute joint |
| `d3_samples` | int | 5 | Number of samples for prismatic joint |

#### Example

```python
# High-resolution workspace (slower but more detailed)
toolbox.plot_workspace_3d(theta1_samples=20, theta2_samples=20, d3_samples=10)

# Low-resolution workspace (faster but less detailed)
toolbox.plot_workspace_3d(theta1_samples=10, theta2_samples=10, d3_samples=5)

# Default resolution
toolbox.plot_workspace_3d()
```

---

## Workspace Visualization Features

### Visual Elements

The workspace visualization displays:

1. **Workspace Vertices** (colored dots)
   - Color represents height (Z-axis)
   - Viridis colormap: dark = low, bright = high
   - Markers show grid vertices at joint boundaries

2. **Workspace Edges** (blue lines)
   - Connect adjacent vertices
   - Show grid structure and connectivity
   - Three types:
     - θ₁ direction: horizontal connections
     - θ₂ direction: perpendicular connections
     - d₃ direction: vertical extensions at boundaries

3. **Workspace Faces** (cyan transparent surfaces)
   - Semi-transparent surfaces filling grid cells
   - Helps visualize workspace volume
   - Can be toggled on/off

4. **Robot Arm** (colored lines with joints)
   - **Link 1** (Steel Blue): First revolute joint link
   - **Link 2** (Coral): Second revolute joint link
   - **Prismatic (Green)**: d₃ extension
   - **End Effector (Red)**: End effector position
   - **Base (Green Square)**: Robot base at origin

5. **Singularities** (purple X markers)
   - Positions where robot loses degrees of freedom
   - Highlighted for reference

6. **Colorbar**
   - Shows Z-height scale
   - Helps interpret vertex colors

---

## Interactive Controls

### Sliders

Two interactive sliders at the bottom of the window:

#### Elevation Slider
- **Range**: -90° to 90°
- **Default**: 0°
- **Effect**: Rotates view up/down
- **Use**: See workspace from different vertical angles

#### Azimuth Slider
- **Range**: -180° to 180°
- **Default**: 0°
- **Effect**: Rotates view around vertical axis
- **Use**: See workspace from different sides

### Toggle Buttons

#### Hide/Show Vertices
- **Location**: Right side of window
- **Function**: Toggles workspace vertex dots visibility
- **Label Changes**: "Hide Vertices" ↔ "Show Vertices"
- **Use**: Reduce clutter, focus on structure

#### Hide/Show Faces
- **Location**: Right side of window
- **Default**: Hidden
- **Function**: Toggles workspace face surfaces visibility
- **Label Changes**: "Show Faces" ↔ "Hide Faces"
- **Use**: Visualize workspace volume or see edges clearly

### Mouse Controls

- **Rotate**: Click and drag with mouse
- **Zoom**: Scroll wheel
- **Pan**: Right-click and drag (depends on matplotlib backend)

---

## Understanding the Visualization

### Workspace Structure

The visualization shows a **grid-based workspace**:

```
- Two parallel layers: d3_min surface and d3_max surface
- Grid points at all (θ₁, θ₂) combinations at boundaries
- Edges connect vertices in three directions:
  - Within θ₁-θ₂ plane (horizontal grid)
  - Between d3_min and d3_max (vertical extensions)
```

### Color Interpretation

**Height-based coloring** (Viridis colormap):
- Dark colors → Low Z values (near base)
- Bright colors → High Z values (far from base)
- Helps identify reachable heights

### Boundary vs Interior

- **Boundary Edges**: Show actual reachable surface
- **Interior Edges**: Help visualize 3D grid structure
- **d3 Boundaries**: Only extend at θ₁ and θ₂ limits

---

## Examples

### Example 1: Default Configuration

```python
from Optimize_Toolbox import RRPToolbox

link_params = [
    [(5, 0, 0), (0, 0, 5)],
    [(3, 0, 0)],
    [(0, 0, 0)]
]

joint_limits = [
    (0, 180),
    (0, 180),
    (3, 5)
]

toolbox = RRPToolbox(link_params, joint_limits)
toolbox.plot_workspace_3d(25, 25, 15)
```

### Example 2: Quick Preview

```python
# Use fewer samples for faster preview
toolbox.plot_workspace_3d()
```

### Example 3: High-Resolution Analysis

```python
# Use more samples for detailed analysis
toolbox.plot_workspace_3d(50, 50, 20)
```

### Example 4: Custom Robot

```python
# Define different link parameters
custom_links = [
    [(6, 0, 0), (0, 0, 4)],  # Longer Link 1
    [(4, 0, 0)],              # Longer Link 2
    [(0, 0, 0.5)]             # Longer End Effector
]

custom_limits = [
    (-90, 90),    # Wider theta1 range
    (0, 120),     # Narrower theta2 range
    (2, 6)        # Larger d3 range
]

custom_toolbox = RRPToolbox(custom_links, custom_limits)
custom_toolbox.plot_workspace_3d()
```

---

## Advanced Features

### Workspace Points

Get workspace points without visualization:

```python
# Generate workspace points
workspace_points = toolbox.get_workspace(theta1_samples=25, theta2_samples=25)

# Use for analysis
for point in workspace_points:
    x, y, z = point
    print(f"Reachable point: ({x:.2f}, {y:.2f}, {z:.2f})")
```

### Singularity Detection

Find singular configurations:

```python
# Find singularities
singularity_positions, singularity_configs = toolbox.find_singularities()

print(f"Found {len(singularity_positions)} singular positions")
for config, pos in singularity_configs:
    theta1, theta2, d3 = config
    x, y, z = pos
    print(f"Singular config: θ₁={theta1:.1f}°, θ₂={theta2:.1f}°, d₃={d3:.2f}")
    print(f"  Position: ({x:.2f}, {y:.2f}, {z:.2f})")
```

## Troubleshooting

### Issue: Plot doesn't appear

**Solution**: Ensure matplotlib backend is set correctly
```python
import matplotlib
matplotlib.use('TkAgg')  # Use TkAgg for interactive windows
import matplotlib.pyplot as plt
```

### Issue: Workspace looks empty

**Cause**: Joint limits may be unrealistic or samples too few

**Solution**: 
```python
# Check joint limits
print(toolbox.joint_limits)

# Use more samples
toolbox.plot_workspace_3d(50, 50, 20)
```

### Issue: Faces not showing

**Solution**: Click "Show Faces" button in the window, or toggle it

### Issue: Slow performance

**Cause**: Too many samples creating too many points

**Solution**: Use fewer samples
```python
# Reduce from default 10, 10, 5
toolbox.plot_workspace_3d(5, 5, 3)
```

### Issue: Memory error with large samples

**Solution**: Use moderate sample values
```python
# Recommended: 10-20 samples
toolbox.plot_workspace_3d(20, 20, 10)
```

---

## Performance Tips

### Recommended Settings

| Use Case | theta1_samples | theta2_samples | d3_samples |
|----------|---|---|---|
| Quick preview | 10 | 10 | 5 |
| Normal analysis | 20 | 20 | 10 |
| Detailed study | 40+ | 40+ | 20+ ||

### Optimization

1. **Start with fewer samples** for testing
2. **Toggle vertices/faces** to reduce visual clutter
3. **Use elevation/azimuth sliders (optional)** for precise rotation 
4. **Hide singularities** if not needed (modify source code)

---

## Legend

### Colors and Markers

- **Blue edges**: Grid connectivity
- **Cyan transparent**: Workspace faces
- **Viridis dots**: Workspace vertices (height-colored)
- **Purple X**: Singularities
- **Steel Blue line**: Link 1
- **Coral line**: Link 2
- **Green line**: Prismatic joint (d3)
- **Red line/star**: End effector
- **Green square**: Robot base

---

*Last updated: November 28, 2025*
