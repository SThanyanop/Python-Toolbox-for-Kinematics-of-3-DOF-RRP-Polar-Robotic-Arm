import sys
import math
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.widgets import Slider

try:
    from scipy.spatial import ConvexHull
except Exception:
    ConvexHull = None

# Try to import the RRP toolbox from the sibling Nueng folder
ROOT = None
if __name__ == "__main__":
    # when run directly assume project root is two levels up
    ROOT = None

def _import_rrp_toolbox():
    # Ensure the repository root is on sys.path so we can import Nueng.test_Toolbox
    import os
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    try:
        from Nueng.test_Toolbox import RRPToolbox
        return RRPToolbox
    except Exception as e:
        raise ImportError("Could not import RRPToolbox from Nueng.test_Toolbox: " + str(e))



class WorkspacePlotter:
    """
    Samples joint space for a 3-DOF RRP (theta1, theta2, d3), computes FK
    using Nueng's RRPToolbox, finds singularities from the reduced Jacobian,
    and plots the workspace surface (convex hull) and singularity points.

    You can pass either:
    - an existing RRPToolbox instance (preferred for custom setups), or
    - link_params and joint_limits (for backward compatibility)

    Example:
        toolbox = RRPToolbox(link_params, joint_limits)
        plotter = WorkspacePlotter(toolbox=toolbox)
        plotter.sample_workspace()
        plotter.plot()
    """

    def __init__(self, link_params=None, joint_limits=None, samples=(72, 36, 15), det_threshold=1e-6, toolbox=None):
        if toolbox is not None:
            self.toolbox = toolbox
            # Try to get link_params and joint_limits from toolbox if not provided
            self.link_params = getattr(toolbox, 'link_params', link_params)
            self.joint_limits = getattr(toolbox, 'joint_limits', joint_limits)
        else:
            RRPToolbox = _import_rrp_toolbox()
            self.toolbox = RRPToolbox(link_params, joint_limits)
            self.link_params = link_params
            self.joint_limits = joint_limits
        self.samples = samples
        self.det_threshold = det_threshold
        self.points = None
        self.joint_grid = None
        self.singularity_points = None

    def sample_workspace(self):
        """
        Sample the entire joint space and find all singularity positions.
        For every valid joint configuration, compute FK and check the reduced Jacobian determinant.
        Every configuration where |det(J)| < threshold is considered a singularity and its position is added to singularity_points.
        All singularities found in the sampled space are included (duplicates possible if workspace overlaps).
        """
        # Create grids for each joint
        t1_min, t1_max = self.joint_limits[0]
        t2_min, t2_max = self.joint_limits[1]
        d3_min, d3_max = self.joint_limits[2]

        t1_vals = np.linspace(t1_min, t1_max, self.samples[0])
        t2_vals = np.linspace(t2_min, t2_max, self.samples[1])
        d3_vals = np.linspace(d3_min, d3_max, self.samples[2])

        pts = []
        joints = []
        singular_pts = []

        for t1 in t1_vals:
            for t2 in t2_vals:
                for d3 in d3_vals:
                    jp = (float(t1), float(t2), float(d3))
                    try:
                        p = self.toolbox.Forward_Kinematics(jp)
                    except Exception:
                        # Skip invalid configurations
                        continue
                    pts.append(p)
                    joints.append(jp)
                    # Jacobian check (reduced 3x3)
                    try:
                        _, reduced_J = self.toolbox.get_RRP_Jacobian_Matrix(jp)
                        Jmat = np.array(reduced_J, dtype=float)
                        if Jmat.shape == (3, 3):
                            det = np.linalg.det(Jmat)
                            # Add every singularity found (including duplicates)
                            if abs(det) < self.det_threshold:
                                singular_pts.append(p)
                    except Exception:
                        pass

        self.points = np.array(pts)
        self.joint_grid = joints
        self.singularity_points = np.array(singular_pts) if singular_pts else np.zeros((0, 3))
        return self.points

    def compute_convex_hull(self):
        if self.points is None:
            self.sample_workspace()
        if ConvexHull is None:
            raise RuntimeError("scipy.spatial.ConvexHull not available; install scipy to compute surface mesh")
        if len(self.points) < 4:
            raise RuntimeError("Not enough points to build a 3D hull")
        hull = ConvexHull(self.points)
        return hull

    def _joint_positions_for_plot(self, joint_parameters):
        # Reconstruct intermediate joint positions using the same DH sequence
        # used in RRPToolbox.get_RRP_Tramsform_Matrix so we can draw links.
        theta1, theta2, d3 = joint_parameters
        # use toolbox's joint_local_positions which were computed from link_params
        x1, y1, z1 = self.toolbox.joint_local_positions[1]
        x2, y2, z2 = self.toolbox.joint_local_positions[2]
        x3, y3, z3 = self.toolbox.joint_local_positions[3]

        # helper to convert toolbox DH array -> numpy
        def to_np(M):
            return np.array(M, dtype=float)

        T01 = to_np(self.toolbox.DH_Martrix_Transform(0, 0, 0, theta1))
        T12 = to_np(self.toolbox.DH_Martrix_Transform(x1, 0, z1, 0))
        T23 = to_np(self.toolbox.DH_Martrix_Transform(0, 90, y1, theta2))
        T34 = to_np(self.toolbox.DH_Martrix_Transform(x2, 0, y2, 90))
        T45 = to_np(self.toolbox.DH_Martrix_Transform(z2, 90, 0, 0))
        T56 = to_np(self.toolbox.DH_Martrix_Transform(z3, 0, x3 + d3, 90))
        T6E = to_np(self.toolbox.DH_Martrix_Transform(y3, 90, 0, 90))

        Ts = [T01, T12, T23, T34, T45, T56, T6E]
        origins = [np.array([0.0, 0.0, 0.0])]
        T_cum = np.eye(4)
        for T in Ts:
            T_cum = T_cum.dot(T)
            origins.append(T_cum[:3, 3].copy())

        return np.array(origins)

    def plot(self, show_robot_joint=None, ax=None, title="Workspace (convex hull) and singularities"):
        if self.points is None:
            self.sample_workspace()

        own_ax = False
        if ax is None:
            fig = plt.figure(figsize=(10, 8))
            # leave room below for sliders
            fig.subplots_adjust(bottom=0.18)
            ax = fig.add_subplot(111, projection='3d')
            own_ax = True

        # Plot surface via convex hull if available
        if ConvexHull is not None and len(self.points) >= 4:
            hull = ConvexHull(self.points)
            faces = hull.simplices
            verts = [self.points[simplex] for simplex in faces]
            poly = Poly3DCollection(verts, alpha=0.25, facecolor='cyan', edgecolor='k')
            ax.add_collection3d(poly)
        else:
            # fallback scatter if hull unavailable
            if len(self.points) > 0:
                ax.scatter(self.points[:, 0], self.points[:, 1], self.points[:, 2], s=1, alpha=0.3)

        # plot singularity points (always add label for legend clarity)
        if self.singularity_points is not None and len(self.singularity_points) > 0:
            ax.scatter(self.singularity_points[:, 0], self.singularity_points[:, 1], self.singularity_points[:, 2],
                       c='r', s=10, label='Singularities')
        else:
            # Add a dummy point for legend if no singularities found
            ax.scatter([], [], [], c='r', s=10, label='Singularities')

        # optional: plot robot links for a given joint
        if show_robot_joint is not None:
            origins = self._joint_positions_for_plot(show_robot_joint)
            ax.plot(origins[:, 0], origins[:, 1], origins[:, 2], '-o', color='orange', label='Robot')

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(title)
        ax.legend()

        # Add interactive sliders to control elevation and azimuth when we
        # created the figure here (own_ax True). Sliders update view and redraw.
        if own_ax:
            try:
                # create slider axes
                ax_elev = fig.add_axes([0.15, 0.10, 0.7, 0.03])
                ax_azim = fig.add_axes([0.15, 0.05, 0.7, 0.03])

                elev_init = getattr(ax, 'elev', 30)
                azim_init = getattr(ax, 'azim', -60)

                elev_slider = Slider(ax_elev, 'Elev', -90.0, 90.0, valinit=elev_init)
                azim_slider = Slider(ax_azim, 'Azim', -180.0, 180.0, valinit=azim_init)

                def _update(val):
                    ax.view_init(elev=float(elev_slider.val), azim=float(azim_slider.val))
                    fig.canvas.draw_idle()

                elev_slider.on_changed(_update)
                azim_slider.on_changed(_update)
            except Exception:
                # If widgets fail (very old matplotlib), fall back to static show
                pass

            plt.show()
