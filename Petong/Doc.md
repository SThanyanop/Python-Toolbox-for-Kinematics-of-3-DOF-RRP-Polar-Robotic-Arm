# Python Toolbox for Kinematics of 3-DOF (RRP) Polar Robotic Arm

### 10.x หาวิธีการทำ Visualization
Visualization ถูกใช้เพื่อจำลองแขนกล RRP และการเคลื่อนที่แบบ 3D แสดงโครงสร้างแขนกล โดย Input Joints เข้าไปแล้วใช้ Foward Kinematics คำนวณตำแหน่งแต่ละ joint ด้วยเวกเตอร์และ rotation matrix เพื่อหาตำแหน่งจุดทุกข้อต่อ 

Input : joint parameters (θ1, θ2, d3)

Output : ตำแหน่ง joint

โดยนำมาประยุกต์ใช้เป็น Code ได้ตามฟังก์ชั่นนี้ (อ้างอิงจาก file RRPVisualization)
```python
def _compute_visual_positions(self, theta1, theta2, d3):
        
        # เปลี่ยนองศาเป็นเรเดียน
        theta1_rad = theta1 * DEG_TO_RAD
        theta2_rad = theta2 * DEG_TO_RAD
        
        # นิยาม matrix หมุนรอบแกน Z สำหรับ θ1
        def Rz(angle):
            c, s = np.cos(angle), np.sin(angle)
            return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
        
        # กำหนดตำแหน่ง Base
        positions = [np.array([0, 0, 0])]
        
        # คำนวณ Link 1 (ใช้ rotation matrix Rz)
        current_pos = np.array([0, 0, 0])
        for segment in self.link_params[0]:
            segment_vec = np.array(segment)
            rotated_segment = Rz(theta1_rad) @ segment_vec
            current_pos = current_pos + rotated_segment
            positions.append(current_pos.copy())
        
        # หาทิศของ Link 2 และ End-effector ตาม θ2
        direction = np.array([
            np.sin(theta2_rad) * np.cos(theta1_rad),
            np.sin(theta2_rad) * np.sin(theta1_rad),
            np.cos(theta2_rad)
        ])
        
        # คำนวณ Link 2 ด้วยทิศทาง
        for segment in self.link_params[1]:
            segment_length = np.linalg.norm(segment)
            current_pos = current_pos + segment_length * direction
            positions.append(current_pos.copy())
        
        # คำนวณระยะยืดของ Joint 3 (Prismatic)
        current_pos = current_pos + d3 * direction
        positions.append(current_pos.copy())
        
        # คำนวณ End Effector ด้วยทิศทาง
        for segment in self.link_params[2]:
            segment_length = np.linalg.norm(segment)
            current_pos = current_pos + segment_length * direction
            positions.append(current_pos.copy())
        
        return np.array(positions)
```
---


จากนั้นทำการวาดเส้นเชื่อมจุดข้อต่อเพื่อแสดงโครงสร้างแขนกล โดยใช้ Matplotlib 3D plot ดังรูป

![alt text](Img/Robot%20arm.png)

Input : joint parameters (θ1, θ2, d3)

Output : ภาพหุ่นยนต์ RRP แบบ 3 มิติ

โดยนำมาประยุกต์ใช้เป็น Code ได้ตามฟังก์ชั่นนี้ (อ้างอิงจาก file RRPVisualization)
```python
def plot_robot(self, theta1=0, theta2=0, d3=0, ax=None, show_frame=True):
        
        #สร้างกราฟ 3D ถ้ายังไม่มี
        if ax is None:
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection='3d')
        
        # คำนวณตำแหน่งจุดของแขนกล
        positions = self._compute_visual_positions(theta1, theta2, d3)
        
        # แยกจุดสำหรับ Link 1, Link 2 และ d3
        link1_end_idx = len(self.link_params[0])
        link2_end_idx = link1_end_idx + len(self.link_params[1])
        d3_end_idx = link2_end_idx + 1
        
        # วาด Link 1
        link1_positions = positions[:link1_end_idx+1]
        ax.plot(link1_positions[:, 0], link1_positions[:, 1], link1_positions[:, 2], 
                'o-', linewidth=3, markersize=8, color='steelblue', label='Link 1')
        
        # วาด Link 2
        link2_positions = positions[link1_end_idx:link2_end_idx+1]
        ax.plot(link2_positions[:, 0], link2_positions[:, 1], link2_positions[:, 2], 
                'o-', linewidth=3, markersize=8, color='coral', label='Link 2')
        
        # วาดแกน Prismatic
        d3_positions = positions[link2_end_idx:d3_end_idx+1]
        ax.plot(d3_positions[:, 0], d3_positions[:, 1], d3_positions[:, 2], 
                'o-', linewidth=4, markersize=8, color='green', label='Prismatic (d3)')
        
        # วาด End Effector
        if len(self.link_params[2]) > 0:
            ee_positions = positions[d3_end_idx:]
            ax.plot(ee_positions[:, 0], ee_positions[:, 1], ee_positions[:, 2], 
                    's-', linewidth=2, markersize=10, color='red', label='End Effector')
        
        # วาดจุด Base ของแขนกล
        ax.scatter(positions[0, 0], positions[0, 1], positions[0, 2], 
                  c='red', s=150, marker='o', label='Base', zorder=5)
        
        # ตั้งค่าช่วงแกน X,Y,Z
        x1, y1, z1 = self.joint_global_positions[1]
        x2, y2, z2 = self.joint_global_positions[2]
        x3, y3, z3 = self.joint_global_positions[3]
        d3_max = self.joint_limits[2][1]
        
        max_reach = math.sqrt((x1 + x2 + x3 + d3_max)**2 + (y1 + y2 + y3)**2 + (z1 + z2 + z3)**2) + 0.5
        
        ax.set_xlim([-max_reach, max_reach])
        ax.set_ylim([-max_reach, max_reach])
        ax.set_zlim([0, max_reach*1.5])
        
        # ตั้งมุมมองกล้อง
        ax.view_init(elev=20, azim=45)
        
        ax.set_xlabel('X (m)', fontsize=10)
        ax.set_ylabel('Y (m)', fontsize=10)
        ax.set_zlabel('Z (m)', fontsize=10)
        ax.set_title(f'RRP Robot Configuration\nθ1={theta1:.1f}°, θ2={theta2:.1f}°, d3={d3:.2f}m', 
                    fontsize=12)
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
        
        return ax
```
---



#### ระบบ Visualization รองรับการปรับท่าแบบ Real-time และ การเคลื่อนที่ตาม time series
การจำลองแขนกล RRP ให้ปรับท่าแบบ Real-time ได้นั้นจะใช้ Slide bar ในการปรับเปลี่ยนค่าของ Joint1(θ1) Joint 2(θ2) และความยาวของ Prismatic link(d3)

โดยนำมาประยุกต์ใช้เป็น Code ได้ตามฟังก์ชั่นนี้ (อ้างอิงจาก file RRPVisualization)
```python
def interactive_plot(self):
        
        # สร้างหน้าต่างกราฟ 3D
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
        plt.subplots_adjust(bottom=0.25)
        
        # กำหนดค่าเริ่มต้นของ Joint
        theta1_init = (self.joint_limits[0][0] + self.joint_limits[0][1]) / 2
        theta2_init = (self.joint_limits[1][0] + self.joint_limits[1][1]) / 2
        d3_init = (self.joint_limits[2][0] + self.joint_limits[2][1]) / 2
        
        # สร้าง Slider bar สำหรับควบคุมแต่ละ Joint
        ax_theta1 = plt.axes([0.15, 0.15, 0.65, 0.03])
        ax_theta2 = plt.axes([0.15, 0.10, 0.65, 0.03])
        ax_d3 = plt.axes([0.15, 0.05, 0.65, 0.03])
        
        slider_theta1 = Slider(ax_theta1, 'θ1 (°)', *self.joint_limits[0], 
                              valinit=theta1_init, valstep=1)
        slider_theta2 = Slider(ax_theta2, 'θ2 (°)', *self.joint_limits[1], 
                              valinit=theta2_init, valstep=1)
        slider_d3 = Slider(ax_d3, 'd3 (m)', *self.joint_limits[2], 
                          valinit=d3_init, valstep=0.01)
        
        def update(val):
            # ล้างภาพเดิมในกราฟ อ่านค่าจาก slider แล้ววาดแขนกล
            ax.cla()
            theta1 = slider_theta1.val
            theta2 = slider_theta2.val
            d3 = slider_d3.val
            self.plot_robot(theta1, theta2, d3, ax=ax)
            
            # แสดงตำแหน่ง End effector
            try:
                ee_pos = self.forward_kinematics([theta1, theta2, d3], update_state=False)
                ax.text2D(0.05, 0.95, f'End Effector: [{ee_pos[0]:.2f}, {ee_pos[1]:.2f}, {ee_pos[2]:.2f}]',
                         transform=ax.transAxes, fontsize=10, verticalalignment='top',
                         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            except Exception as e:
                ax.text2D(0.05, 0.95, f'Error: {str(e)}',
                         transform=ax.transAxes, fontsize=10, verticalalignment='top',
                         bbox=dict(boxstyle='round', facecolor='red', alpha=0.5))
            
            fig.canvas.draw_idle()
        
        # เชื่อม Slider bar กับ ฟังก์ชั่น update()
        slider_theta1.on_changed(update)
        slider_theta2.on_changed(update)
        slider_d3.on_changed(update)
        
        # แสดงภาพเริ่มต้น
        self.plot_robot(theta1_init, theta2_init, d3_init, ax=ax)
        plt.show()
```

การจำลองการเคลื่อนที่แขนกล RRP นั้นเรารับ position มาเป็น vector xyz แล้วสร้างเส้นทาง โดยจะสร้างจุดย่อย ๆ ระหว่างแต่ละ waypoint เพื่อให้หุ่นยนต์เคลื่อนที่ นุ่มนวลต่อเนื่อง โดยไม่กระโดดข้ามทีละจุดใหญ่เกินไป

Input : waypoints, total_time, fps

Output: trajectory, time_stamps

โดยนำมาประยุกต์ใช้เป็น Code ได้ตามฟังก์ชั่นนี้ (อ้างอิงจาก file RRPVisualization)
```python
def interpolate_trajectory(self, waypoints, total_time, fps=30):
      
        # แบ่งเวลาให้แต่ละ waypoint
        num_waypoints = len(waypoints)
        times = np.linspace(0, total_time, num_waypoints)
        
        trajectory = []
        time_stamps = []
        
        # Loop ผ่าน waypoint ทีละคู่
        for i in range(len(waypoints) - 1):
            start = waypoints[i]
            end = waypoints[i + 1]
            t_start = times[i]
            t_end = times[i + 1]
            duration = t_end - t_start
            
            if duration <= 0:
                continue
            
            # หาจำนวนเฟรมของช่วงนั้น
            num_frames = int(duration * fps)
            
            # ทำ Linear Interpolation ระหว่างจุด
            for j in range(num_frames):
                alpha = j / num_frames
                interpolated = tuple(
                    start[k] + alpha * (end[k] - start[k]) 
                    for k in range(len(start))
                )
                trajectory.append(interpolated)
                time_stamps.append(t_start + alpha * duration)
        
        # เพิ่ม waypoint สุดท้าย
        trajectory.append(waypoints[-1])
        time_stamps.append(times[-1])
        
        return trajectory, time_stamps
```

จากนั้นสร้าง Animation ของหุ่นยนต์ RRP ที่เคลื่อนที่ตาม trajectory ด้วย FuncAnimation() ของ matplotlib โดยฟังก์ชั่น ใช้ Inverse Kinematics แปลงเป็น joint angles จากนั้น Interpolate trajectory เพื่อให้แขนกลเคลื่อนที่ สร้างหน้าต่าง 3D Plot และใช้ฟังก์ชั่น update_frame() ในการวาดใหม่ทุกเฟรม

Input : trajectory, total_time

Output : แสดงการเคลื่อนที่ของแขนกล

โดยนำมาประยุกต์ใช้เป็น Code ได้ตามฟังก์ชั่นนี้ (อ้างอิงจาก file RRPVisualization)
```python
def animate_trajectory(self, trajectory, total_time=None, trajectory_type='joint', fps=30):

        # กำหนดเวลาถ้าไม่ได้ระบุไว้
        if total_time is None:
            total_time = len(trajectory)  
    
        # ทำ Interpolation เพื่อให้หุ่นยนต์เคลื่อน
        if trajectory_type == 'position':
            # เปลี่ยน positions ไปเป็น joint configs
            joint_waypoints = []
            for pos in trajectory:
                x, y, z = pos
                joint_config = self.inverse_kinematics((x, y, z), validate=True)
                joint_waypoints.append(joint_config)
        
            joint_trajectory, time_stamps = self.interpolate_trajectory(joint_waypoints, total_time, fps)
        else:
            joint_trajectory, time_stamps = self.interpolate_trajectory(trajectory, total_time, fps)
    
        # คำนวณตำแหน่ง End-effector ของทุกเฟรม
        actual_ee_path = []
        for joint_config in joint_trajectory:
            theta1, theta2, d3 = joint_config
            positions = self._compute_visual_positions(theta1, theta2, d3)
            actual_ee_path.append(positions[-1])
        path_points = np.array(actual_ee_path)
    
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')

        # สำหรับวาดแต่ละเฟรม
        def update_frame(frame):
            ax.cla()
            theta1, theta2, d3 = joint_trajectory[frame]
            # วาดแขนกลตาม joint ของเฟรมนั้น
            self.plot_robot(theta1, theta2, d3, ax=ax, show_frame=False)

            # วาดเส้นทางที่ปลายแขนเดินผ่าน
            ax.plot(path_points[:frame+1, 0], path_points[:frame+1, 1], 
                path_points[:frame+1, 2], 'r--', linewidth=2, 
                alpha=0.5, label='End Effector Path')
            # วาดตำแหน่งของปลายแขนปัจจุบัน
            ax.scatter(path_points[frame, 0], path_points[frame, 1], 
                    path_points[frame, 2], s=100, c='yellow', marker='*', 
                    edgecolors='black', linewidths=2, zorder=10)

            # แสดงเวลาและค่า joint ปัจจุบันบนหัวข้อ
            current_time = time_stamps[frame]
            total_time_display = time_stamps[-1]
            ee_pos = path_points[frame]

            # กำหนดหัวข้อและข้อมูลต่างๆ
            ax.set_title(f'Time: {current_time:.2f}s / {total_time_display:.2f}s\n' + 
                        f'θ1={theta1:.1f}°, θ2={theta2:.1f}°, d3={d3:.2f}m\n' +
                        f'End Effector: [{ee_pos[0]:.2f}, {ee_pos[1]:.2f}, {ee_pos[2]:.2f}]')
            ax.legend(loc='upper right')

        # แปลง fps เป็น milliseconds per frame
        interval = 1000 / fps  

        #เรียก FuncAnimation เพื่อรัน Animation
        anim = FuncAnimation(fig, update_frame, frames=len(joint_trajectory), 
                       interval=interval, repeat=True)
        plt.show()
        return anim
```

### 11.x Visualization Validation
ในส่วนของ Visualization นั้นแบ่งได้เป็นสองส่วนคือ Interactive Visualization และ Trajectory Visualization ทำการ Validate เพื่อที่จะตรวจสอบความถูกต้อง

#### 11.x.x Validate Interactive Visualization
**Hypothesis :** ลิงค์ที่ plot ออกมานั้นมีรูปแบบและความยาวถูกต้อง

#### Test Case : Link 1 Check
**วัตถุประสงค์ :** ตรวจสอบว่าลิงค์ที่ plot ออกมานั้นมีรูปแบบและความยาวถูกต้อง เมื่อเทียบกับลิงค์ที่วาดมือ

**Input :** 
```
link_1 = [(5, 0, 0), (0, 0, 5)]
```

**ผลลัพธ์ที่คาดหวัง :** Link 1 จากการ Plot โดย code มีรูปร่างและความยาวตรงกับ Link 1 จากการวาดมือ

**ผลการทดสอบ :**   

Link 1 จากการวาดมือ  
![alt text](Img/Link1_Handdraw.jpg)

Link 1 จากการ Plot โดย Code
![alt text](Img/Link1%20(Small).png)

**สรุปผลการทดลอง :** จากผลการทดสอบ พบว่า Link 1 จากการวาดมือ และ Link 1 จากการ Plot โดย code นั้นมีรูปแบบและความยาวเหมือนกัน ดังนั้นลิงค์ที่ plot ออกมานั้นมีรูปแบบและความยาวถูกต้อง

#### 11.x.x Validate Trajectory Visualization
**Hypothesis :** การเคลื่อนที่ของแขนกลถูกต้อง

#### Test Case : Trajectory Check
**วัตถุประสงค์ :** ตรวจสอบว่าการเคลื่อนที่ของแขนกลถูกต้องตาม End Effector Path

**Input :** 
```
position_trajectory = [
        [3, 3, 8],
        [3, -3, 10],
        [-3, -3, 8],
        [-3, 3, 10],
        [3, 3, 8],
    ]
```

**ผลลัพธ์ที่คาดหวัง :** การเคลื่อนที่ของแขนกลตำแหน่ง End effector ตรงกับ End Effector Path

**ผลการทดสอบ :**

![alt text](Img/trajec.png)


Code ที่ทำการสร้าง End Effector Path
```python
ax.plot(path_points[:frame+1, 0], path_points[:frame+1, 1], 
                path_points[:frame+1, 2], 'r--', linewidth=2, 
                alpha=0.5, label='End Effector Path')
            ax.scatter(path_points[frame, 0], path_points[frame, 1], 
                    path_points[frame, 2], s=100, c='yellow', marker='*', 
                    edgecolors='black', linewidths=2, zorder=10)
```
**สรุปผลการทดลอง :** จากผลการทดสอบสังเกตุได้ว่าการเคลื่อนที่ของ End effector อยู่ในตำแหน่งที่ตรงกับ End Effector Path ดังนั้นแขนกลมีการเคลื่อนที่ที่ถูกต้อง