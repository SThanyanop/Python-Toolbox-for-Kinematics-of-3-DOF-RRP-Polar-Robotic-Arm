# Python Toolbox for Kinematics Analysis of 3-DOF (RRP) Polar Robotic Arm

## Project Team
- ชยากร ชื่นประเสริฐ (66340500012)
- ศักดิ์ไชย ธนภาสวัฒน์ (66340500052)
- ธัญนพ ศรีวานิช (66340500073)

**Institution:** Institute of Field Robotics, King Mongkut's University of Technology Thonburi  
**Program:** Senior Thesis FRAB (FIBO Robotics and Automation: Bachelor)  
**Academic Year:** 2025

---

## บทคัดย่อ (Abstract)

โครงงานนี้นำเสนอการพัฒนา Python Toolbox สำหรับการวิเคราะห์จลนศาสตร์ของแขนกลโพลาร์ 3 องศาอิสระ (3-DOF RRP Polar Robotic Arm) โดยเครื่องมือที่พัฒนาขึ้นสามารถคำนวณ Configuration Space และอัตราการเปลี่ยนแปลงจาก Task Space และในทางกลับกัน ผ่านการประยุกต์ใช้ Forward Kinematics, Inverse Kinematics และ Differential Kinematics ระบบได้รับการออกแบบให้รองรับการกำหนดพารามิเตอร์ของแขนกลตามโครงสร้างที่ผู้ใช้ต้องการ พร้อมทั้งคำนวณและแสดงผล Workspace ในปริภูมิสามมิติ รวมถึงการแสดงภาพแบบเคลื่อนไหว (Animation) เพื่อการศึกษาและวิเคราะห์พฤติกรรมของแขนกลอย่างครอบคลุม

**คำสำคัญ:** Python Toolbox, Polar Robotic Arm, Kinematics Analysis, Configuration Space, Task Space, Workspace Visualization

---

## 1. จุดประสงค์ของโครงการ (Project Objectives)

1. เพื่อศึกษาและวิเคราะห์จลนศาสตร์ของ Polar Robotic Arm ในปริภูมิสามมิติ
2. เพื่อพัฒนา Python Toolbox สำหรับการคำนวณ Forward Kinematics, Inverse Kinematics และ Differential Kinematics ของ Polar Robotic Arm
3. เพื่อพัฒนาระบบแสดงภาพแบบสามมิติ (3D Visualization) สำหรับการแสดงท่าทาง (Pose) และการเคลื่อนไหว (Animation) ของแขนกล

---

## 2. ขอบเขตของโครงการ (Project Scope)

### 2.1 ขอบเขตการพัฒนา Python Toolbox

ระบบที่พัฒนาครอบคลุมเฉพาะ Pose Kinematics ดังนี้:

- Forward Kinematics: การคำนวณตำแหน่งและการวางแนว (Position and Orientation) ของ End Effector จาก Configuration Space
- Inverse Kinematics: การคำนวณมุมและตำแหน่งของข้อต่อจากตำแหน่งที่ต้องการของ End Effector
- Forward Differential Kinematics: การคำนวณความเร็วของ End Effector จากความเร็วของข้อต่อ
- Inverse Differential Kinematics: การคำนวณความเร็วของข้อต่อจากความเร็วที่ต้องการของ End Effector

**ข้อจำกัด:** โครงงานนี้ไม่ครอบคลุมการศึกษาและการคำนวณเกี่ยวกับความเร่ง (Acceleration) ของระบบ

### 2.2 ข้อกำหนดของ Polar Robotic Arm

**คุณสมบัติของแขนกล:**
- ทำงานในปริภูมิสามมิติ (3D Workspace)
- คำนวณตำแหน่งและความเร็วใน Configuration Space (Joint Space)
- คำนวณตำแหน่งและความเร็วใน Task Space (Cartesian Space: X, Y, Z axes)
- วิเคราะห์และแสดงผล Workspace ของแขนกล

**ข้อสมมติฐาน:**
- ไม่มีสิ่งกีดขวางในพื้นที่ทำงาน
- ไม่พิจารณาโครงสร้างของ End Effector
- แขนกลเป็น Rigid Body ไร้แรงเสียดทาน
- Actuators มีความแม่นยำสมบูรณ์แบบ

### 2.3 โครงสร้างของระบบแขนกล

**องค์ประกอบของแขนกล:**
- จำนวนลิงก์: 4 links
- ชนิดของข้อต่อ: RRP Configuration (2 Revolute joints + 1 Prismatic joint)
- จำนวน Actuators: 3 ตัว
- จำนวนข้อต่อสูงสุดต่อลิงก์: 2 joints

### 2.4 พารามิเตอร์นำเข้า (Input Parameters)

**1. ขนาดของลิงก์ (Link Dimensions)**

กำหนดเป็นเซตของเวกเตอร์ที่แสดงระยะห่างระหว่างข้อต่อ:

```
Link_i = {v₁, v₂, ..., vₙ}
```

โดยที่:
- vⱼ คือเวกเตอร์ระยะทางส่วนที่ j ของลิงก์ที่ i
- v₁ เชื่อมต่อกับ joint_{i-1}
- vₙ เชื่อมต่อกับ joint_i

**2. ข้อจำกัดของข้อต่อ (Joint Constraints)**

กำหนดเป็นเมทริกซ์ขนาด 3×2 (สำหรับแกน X, Y, Z):

```
JointLimits_i = [min_x  max_x]
                [min_y  max_y]
                [min_z  max_z]
```

สำหรับข้อต่อแบบหมุน (Revolute Joint):
- กำหนดขอบเขตมุมหมุน (ในหน่วยเรเดียน) ตามกฎมือขวา
- ค่าบวกและลบแสดงทิศทางการหมุนตามแกนที่กำหนด

สำหรับข้อต่อแบบเลื่อน (Prismatic Joint):
- กำหนดระยะยืดออกสูงสุด (Extension) และระยะหดเข้าสูงสุด (Retraction)

**3. Task Space และเวลาในการเคลื่อนที่**

กำหนดตำแหน่งและการวางแนวของ End Effector พร้อมระยะเวลาในการเคลื่อนที่:

```
TaskSpace = [x, y, z, Δt]
```

**4. Configuration Space และเวลาในการเคลื่อนที่**

กำหนดค่าตำแหน่งของข้อต่อทั้งหมดพร้อมระยะเวลา:

```
ConfigSpace = [θ₁, θ₂, d₃, Δt]
```

---

## 3. ทฤษฎีพื้นฐาน (Theoretical Background)

### 3.1 คุณสมบัติของ 3-DOF (RRP) Polar Robotic Arm

แขนกลโพลาร์ 3 องศาอิสระมีโครงสร้างแบบทรงกลม (Spherical Configuration) ประกอบด้วยข้อต่อดังนี้:

1. **ข้อต่อหมุนฐาน (Base Revolute Joint):** θ₁
2. **ข้อต่อหมุนหัวไหล่ (Shoulder Revolute Joint):** θ₂
3. **ข้อต่อเลื่อน (Prismatic Joint):** d₃

โครงสร้างนี้ช่วยให้แขนกลสามารถเคลื่อนที่ครอบคลุมพื้นที่ทำงานแบบทรงกลมได้อย่างมีประสิทธิภาพ

### 3.2 พารามิเตอร์ Denavit-Hartenberg (DH Parameters)

การกำหนด Coordinate Frames ของแขนกลตามหลัก DH Convention สำหรับการคำนวณจลนศาสตร์:

**ตารางพารามิเตอร์ DH ตามการ Implementation จริง:**

| i | aᵢ | αᵢ | dᵢ | θᵢ |
|:---:|:------:|:------:|:--------:|:--------:|
| 1 | 0 | 0° | 0 | θ₁* |
| 2 | X₁ | 0° | Z₁ | 0° |
| 3 | 0 | 90° | Y₁ | θ₂* |
| 4 | X₂ | 0° | Y₂ | 90° |
| 5 | Z₂ | 90° | 0 | 0° |
| 6 | Z₃ | 0° | X₃ + d₃* | 90° |
| E | Y₃ | 90° | 0 | 90° |

*หมายเหตุ:* ตัวแปรที่มีเครื่องหมาย (*) คือตัวแปรข้อต่อ (Joint Variables)

**สัญลักษณ์ที่ใช้:**
- aᵢ: ระยะห่างระหว่างแกน Zᵢ₋₁ และ Zᵢ วัดตามแกน Xᵢ
- αᵢ: มุมระหว่างแกน Zᵢ₋₁ และ Zᵢ วัดรอบแกน Xᵢ
- dᵢ: ระยะห่างระหว่างแกน Xᵢ₋₁ และ Xᵢ วัดตามแกน Zᵢ₋₁
- θᵢ: มุมระหว่างแกน Xᵢ₋₁ และ Xᵢ วัดรอบแกน Zᵢ₋₁

### 3.3 Transformation Matrix

**เมทริกซ์การแปลงพิกัดระหว่างเฟรม:**

Transformation Matrix จากเฟรม i-1 ไปยังเฟรม i ตามหลัก Modified DH Convention:

```
ⁱ⁻¹Tᵢ = Trans(aᵢ, 0, 0) · Rot(x, αᵢ) · Rot(z, θᵢ) · Trans(0, 0, dᵢ)
```

หรือเขียนในรูปเมทริกซ์ขนาด 4×4:

```
ⁱ⁻¹Tᵢ = [cos(θᵢ)                -sin(θᵢ)                0           aᵢ           ]
        [sin(θᵢ)cos(αᵢ)          cos(θᵢ)cos(αᵢ)         -sin(αᵢ)    -sin(αᵢ)dᵢ  ]
        [sin(θᵢ)sin(αᵢ)          cos(θᵢ)sin(αᵢ)          cos(αᵢ)     cos(αᵢ)dᵢ   ]
        [0                        0                       0           1            ]
```

**สัญลักษณ์ย่อ:**
- cos(θᵢ) ≡ Cθᵢ ≡ cᵢ
- sin(θᵢ) ≡ Sθᵢ ≡ sᵢ

### 3.4 การวิเคราะห์ Workspace

**Jacobian Matrix:**

Jacobian Matrix เป็นเมทริกซ์ที่แสดงความสัมพันธ์ระหว่างความเร็วของข้อต่อกับความเร็วของ End Effector:

```
ẋ = J(q)q̇
```

โดยที่:
```
J(q) = ∂f(q)/∂q = [∂fᵢ/∂qⱼ]
```

สำหรับ i = 1, 2, 3 (X, Y, Z) และ j = 1, 2, 3 (ข้อต่อที่ 1, 2, 3)

**เงื่อนไข Singularity:**

Singularity เกิดขึ้นเมื่อ Jacobian Matrix สูญเสียอันดับ (Rank Deficiency):

```
det(J(q)) = 0
```

ณ จุด Singularity ระบบสูญเสียความสามารถในการเคลื่อนที่ในทิศทางบางทิศทาง หรือต้องใช้ความเร็วของข้อต่อที่สูงมากในการควบคุม

**เซตของจุด Singular:**

พื้นที่ทำงาน (Workspace) ถูกกำหนดโดยการรวมเซตต่อไปนี้:

- **S₁:** เซตของจุดที่ Null Space ของ J(q) มีมิติ ≥ 1 (Internal Singularities)
- **S₂:** เซตของจุดที่ขอบเขตของ Workspace (Boundary Singularities)
- **S₃:** เซตของจุดที่เกิดจากข้อจำกัดของข้อต่อ (Joint Limit Singularities)

**Total Workspace:**
```
W = S₁ ∪ S₂ ∪ S₃
```

**Reachable Workspace:** เซตของจุดทั้งหมดที่ End Effector สามารถเข้าถึงได้อย่างน้อยหนึ่งการจัดวาง (Orientation)

**Dexterous Workspace:** เซตของจุดที่ End Effector สามารถเข้าถึงได้ในทุกการจัดวาง

---

## 4. ภาพรวมของระบบ (System Overview)

### 4.1 แผนผังระบบ (System Architecture)

```
Input Parameters → Parameter Validation → Kinematic Computation → Output & Visualization
       ↓                    ↓                      ↓                        ↓
  • Link Vectors       • Range Check         • Forward Kin.          • Task Space
  • Joint Limits       • Feasibility         • Inverse Kin.          • Config Space
  • Task Space         • Error Detection     • Jacobian Matrix       • Velocities
  • Config Space                             • Workspace Calc.       • 3D Visualization
  • Time Duration                            • Singularity Check     • Animation
```

### 4.2 ฟังก์ชันหลักของระบบ

#### Function 1: Robot Specification Setup
**วัตถุประสงค์:** กำหนดพารามิเตอร์และข้อจำกัดของแขนกล

**Input:**
- เวกเตอร์ขนาดของลิงก์ทุกลิงก์
- เมทริกซ์ข้อจำกัดของข้อต่อ (X-Y-Z Matrix)

**Output:**
- Robot configuration object

**Process:**
1. รับและตรวจสอบความถูกต้องของพารามิเตอร์
2. คำนวณ DH Parameters จากข้อมูลลิงก์
3. สร้าง object เก็บข้อมูลโครงสร้างแขนกล

#### Function 2: 3D Workspace Calculation
**วัตถุประสงค์:** คำนวณและแสดงผล Workspace ของแขนกล

**Input:**
- Robot configuration object
- Sample resolution parameters (theta1_samples, theta2_samples, d3_samples)

**Output:**
- ขอบเขต Workspace ในพิกัด 3 มิติ
- ตำแหน่งของจุด Singularity
- 3D visualization พร้อม edges และ faces

**Process:**
1. คำนวณ samples สำหรับ θ₁, θ₂ และ d₃ ตามความละเอียดที่กำหนด
2. คำนวณ Forward Kinematics สำหรับทุกค่าข้อต่อที่เป็นไปได้
3. วิเคราะห์ Jacobian Matrix เพื่อหาจุด Singularity (ใช้ samples ที่ละเอียดกว่า)
4. สร้าง vertices, edges และ faces สำหรับการแสดงผล 3D
5. แสดง Workspace พร้อมทำเครื่องหมายจุด Singularity

**Sampling Strategy:**
- θ₁, θ₂: Linear spacing ตั้งแต่ min ถึง max
- d₃: ใช้เฉพาะค่า min และ max (เพื่อลดการคำนวณ)
- Singularity detection: ใช้ samples ครึ่งหนึ่งของ workspace samples

#### Function 3: Forward Kinematics
**วัตถุประสงค์:** แปลง Configuration Space เป็น Task Space

**Mathematical Foundation:**
```
T⁰ₑ(q) = T⁰₁(θ₁) · T¹₂(θ₂) · T²₃(d₃) · ... · Tⁿₑ
```

**Input:**
```
q = [θ₁, θ₂, d₃]ᵀ
```

**Output:**
```
x = [xₑ, yₑ, zₑ]ᵀ
```

**Process:**
1. คำนวณ Transformation Matrix แต่ละข้อต่อจาก DH Parameters
2. คูณ Transformation Matrices ตามลำดับ
3. สกัดตำแหน่ง End Effector จากเมทริกซ์ผลลัพธ์

#### Function 4: Inverse Kinematics
**วัตถุประสงค์:** แปลง Task Space เป็น Configuration Space

**Input:**
```
x = [xₑ, yₑ, zₑ]ᵀ
```

**Output:**
```
q = [θ₁, θ₂, d₃]ᵀ
```

**Process:**
1. ตรวจสอบว่าตำแหน่งเป้าหมายอยู่ใน Workspace
2. คำนวณ Configuration Space โดยใช้วิธีการแก้สมการเชิงวิเคราะห์
3. ตรวจสอบว่าค่าที่ได้อยู่ในขอบเขตของข้อต่อ
4. คืนค่าผลลัพธ์หรือแจ้งข้อผิดพลาด

**Error Conditions:**
- "End position out of workspace": เมื่อตำแหน่งเป้าหมายอยู่นอก Workspace
- "No valid solution": เมื่อไม่มีคำตอบที่ตอบสนองข้อจำกัดของข้อต่อ

#### Function 5: Forward Differential Kinematics
**วัตถุประสงค์:** คำนวณความเร็วของ End Effector จากความเร็วของข้อต่อ

**Mathematical Foundation:**
```
ẋ = J(q)q̇
```

**Input:**
```
q = [θ₁, θ₂, d₃]ᵀ
q̇ = [θ̇₁, θ̇₂, ḋ₃]ᵀ
```

**Output:**
```
ẋ = [ẋₑ, ẏₑ, żₑ]ᵀ
```

**Process:**
1. คำนวณ Jacobian Matrix ณ Configuration ปัจจุบัน
2. คูณ Jacobian Matrix กับเวกเตอร์ความเร็วของข้อต่อ

#### Function 6: Inverse Differential Kinematics
**วัตถุประสงค์:** คำนวณความเร็วของข้อต่อจากความเร็วที่ต้องการของ End Effector

**Mathematical Foundation:**
```
q̇ = J⁻¹(q)ẋ
```

หรือใช้ Pseudoinverse:
```
q̇ = J⁺(q)ẋ
```

โดยที่ J⁺ = (JᵀJ)⁻¹Jᵀ

**Input:**
```
q = [θ₁, θ₂, d₃]ᵀ
ẋ = [ẋₑ, ẏₑ, żₑ]ᵀ
```

**Output:**
```
q̇ = [θ̇₁, θ̇₂, ḋ₃]ᵀ
```

**Process:**
1. คำนวณ Jacobian Matrix ณ Configuration ปัจจุบัน
2. ตรวจสอบ Singularity (det(J) ≈ 0)
3. คำนวณ Inverse หรือ Pseudoinverse ของ Jacobian
4. คูณกับเวกเตอร์ความเร็วที่ต้องการ

**Singularity Handling:**
เมื่อ det(J) ใกล้ศูนย์ ใช้ Damped Least Squares:
```
q̇ = (JᵀJ + λ²I)⁻¹Jᵀẋ
```

#### Function 7: Task Space Calculation (Complete)
**วัตถุประสงค์:** คำนวณ Task Space และความเร็วจาก Configuration Space

**Input:**
```
q = [θ₁, θ₂, d₃]ᵀ
Δt (time duration)
```

**Output:**
```
x = [xₑ, yₑ, zₑ]ᵀ
ẋ = [ẋₑ, ẏₑ, żₑ]ᵀ
```

**Process:**
1. ตรวจสอบความถูกต้องของ Input
2. คำนวณ Forward Kinematics → x
3. คำนวณ q̇ = Δq/Δt
4. คำนวณ Forward Differential Kinematics → ẋ
5. คืนค่าผลลัพธ์

**Error Handling:**
- "Invalid input": พารามิเตอร์นำเข้าไม่ถูกต้อง
- "Configuration out of bounds": Configuration Space เกินขอบเขตข้อต่อ

#### Function 8: Configuration Space Calculation (Complete)
**วัตถุประสงค์:** คำนวณ Configuration Space และความเร็วจาก Task Space

**Input:**
```
x = [xₑ, yₑ, zₑ]ᵀ
Δt (time duration)
```

**Output:**
```
q = [θ₁, θ₂, d₃]ᵀ
q̇ = [θ̇₁, θ̇₂, ḋ₃]ᵀ
```

**Process:**
1. ตรวจสอบความถูกต้องของ Input
2. ตรวจสอบว่า x อยู่ใน Workspace
3. คำนวณ Inverse Kinematics → q
4. คำนวณ ẋ = Δx/Δt
5. คำนวณ Inverse Differential Kinematics → q̇
6. คืนค่าผลลัพธ์

**Error Handling:**
- "Invalid input": พารามิเตอร์นำเข้าไม่ถูกต้อง
- "End position out of workspace": Task Space อยู่นอก Workspace
- "No valid configuration": ไม่มี Configuration ที่ตอบสนองข้อจำกัด

#### Function 9: 3D Model Display
**วัตถุประสงค์:** แสดงภาพแบบจำลอง 3D ของแขนกล

**Input:**
- Robot configuration object
- Current pose (q)

**Output:**
- 3D visualization (static image)

**Process:**
1. คำนวณตำแหน่งของลิงก์และข้อต่อทั้งหมด
2. สร้างโมเดล 3D ของแขนกล
3. แสดงผล Coordinate Frames ของแต่ละข้อต่อ
4. แสดงขอบเขต Workspace (optional)

#### Function 10: 3D Animation
**วัตถุประสงค์:** แสดงการเคลื่อนไหวของแขนกลตามเส้นทางที่กำหนด

**Input:**
- Trajectory data (จาก Function 7 หรือ 8)
- Animation parameters (frame rate, duration)

**Output:**
- Animated 3D visualization

**Process:**
1. รับข้อมูลเส้นทาง (Trajectory) จากฟังก์ชันคำนวณ
2. สร้างเฟรมแอนิเมชันตามอัตราที่กำหนด
3. แสดงการเคลื่อนไหวแบบต่อเนื่อง
4. แสดง End Effector trajectory (optional)

---

## 5. เนื้อหาวิชาที่เกี่ยวข้อง (Related Course Topics)

1. **Transformation of Coordinate Frames**
   - Homogeneous Transformation
   - Rotation Matrices
   - Position Vectors

2. **Forward Kinematics**
   - DH Convention
   - Link Transformation
   - End Effector Position

3. **Inverse Kinematics**
   - Analytical Solutions
   - Geometric Approach
   - Multiple Solutions Handling

4. **Forward Differential Kinematics**
   - Jacobian Matrix
   - Velocity Transformation
   - Linear and Angular Velocity

5. **Inverse Differential Kinematics**
   - Jacobian Inverse
   - Pseudoinverse Method
   - Singularity Avoidance

---

## 6. ผลการศึกษาที่คาดหวัง (Expected Results)

1. **Python Toolbox ที่สมบูรณ์** สามารถคำนวณและแสดงผลลัพธ์ได้ครบถ้วนดังนี้:
   - ตำแหน่งเชิงมุมของข้อต่อแต่ละข้อ (Configuration Space)
   - ตำแหน่งของ End Effector ในปริภูมิงาน (Task Space)
   - ความเร็วของข้อต่อและ End Effector
   - การแสดงภาพแบบจำลอง 3D และแอนิเมชัน

2. **ความถูกต้องของการคำนวณ Workspace**
   - แสดง Reachable Workspace อย่างครบถ้วน
   - ระบุจุด Singularity ได้ถูกต้อง
   - ตรวจสอบความเป็นไปได้ของเส้นทางการเคลื่อนที่

3. **ระบบที่มีความยืดหยุ่น**
   - รองรับการกำหนดพารามิเตอร์แขนกลได้หลากหลาย
   - ตรวจสอบและแจ้งข้อผิดพลาดอย่างชัดเจน

---

## 7. แผนการดำเนินงาน (Project Timeline)

| ลำดับ | รายการกิจกรรม | สัปดาห์ที่ 1 | สัปดาห์ที่ 2 | สัปดาห์ที่ 3 | สัปดาห์ที่ 4 | สัปดาห์ที่ 5 | สัปดาห์ที่ 6 | สัปดาห์ที่ 7 |
|:-----:|---------------|:------------:|:------------:|:------------:|:------------:|:------------:|:------------:|:------------:|
| 1 | จัดทำ Proposal | ✓ | ✓ | | | | | |
| 2 | แก้ไข Proposal | | | ✓ | | | | |
| 3 | ศึกษา Pose Kinematics | ✓ | ✓ | | | | | |
| 4 | ศึกษา 3D Workspace | ✓ | ✓ | | | | | |
| 5 | พัฒนา Forward Kinematics | | | ✓ | | | | |
| 6 | พัฒนา Inverse Kinematics | | | ✓ | | | | |
| 7 | พัฒนา Forward Differential Kinematics | | | ✓ | | | | |
| 8 | พัฒนา Inverse Differential Kinematics | | | ✓ | ✓ | | | |
| 9 | พัฒนา 3D Workspace Calculation | | | ✓ | ✓ | | | |
| 10 | พัฒนา Visualization System | | | | ✓ | ✓ | | |
| 11 | ทดสอบและปรับปรุงระบบ | | | | | ✓ | ✓ | |
| 12 | จัดทำ User Manual | | | | | | ✓ | |
| 13 | จัดทำรายงานฉบับสมบูรณ์ | | | | | | ✓ | ✓ |
| 14 | นำเสนอและสาธิตโครงงาน | | | | | | | ✓ |

---

## 8. เอกสารอ้างอิง (References)

[1] Adil Khan. "Forward Kinematic Modeling of a 3-DOF (RRP) Polar Robotic Arm Using OOPS", 2009

[2] Khushdeep Goyal, Davinder Sethi "AN ANALYTICAL METHOD TO FIND WORKSPACE OF A ROBOTIC MANIPULATOR", 2010

---

## 9. รายละเอียดการดำเนินการทางคณิตศาสตร์ (Mathematical Implementation Details)

### 9.1 การคำนวณ Transformation Matrix

**สัญลักษณ์ที่ใช้:**
- cᵢ = cos(θᵢ), sᵢ = sin(θᵢ)
- Xᵢ, Yᵢ, Zᵢ = ตำแหน่งโกลบอลของข้อต่อที่ i (Joint Global Position)

**ตาราง DH Parameters ตามการ Implementation:**

| i | aᵢ | αᵢ | dᵢ | θᵢ |
|:---:|:------:|:------:|:--------:|:--------:|
| 1 | 0 | 0° | 0 | θ₁* |
| 2 | X₁ | 0° | Z₁ | 0° |
| 3 | 0 | 90° | Y₁ | θ₂* |
| 4 | X₂ | 0° | Y₂ | 90° |
| 5 | Z₂ | 90° | 0 | 0° |
| 6 | Z₃ | 0° | X₃ + d₃* | 90° |
| E | Y₃ | 90° | 0 | 90° |

*หมายเหตุ:* 
- ตัวแปรที่มีเครื่องหมาย (*) คือตัวแปรข้อต่อ
- X₁, Y₁, Z₁ = ตำแหน่งโกลบอลของข้อต่อที่ 1
- X₂, Y₂, Z₂ = ตำแหน่งโกลบอลของข้อต่อที่ 2
- X₃, Y₃, Z₃ = ตำแหน่งโกลบอลของข้อต่อที่ 3

**Transformation Matrix จากเฟรม 0 ไปเฟรม E:**

การคำนวณโดยใช้สูตร Modified DH Convention:
```
ⁱ⁻¹Tᵢ = Trans(aᵢ, 0, 0) · Rot(x, αᵢ) · Rot(z, θᵢ) · Trans(0, 0, dᵢ)
```

ผลลัพธ์สุดท้าย:

```
⁰Tₑ = [c₁c₂   -s₁    -c₁s₂   Pₓ]
      [c₂s₁    c₁    -s₁s₂   Pᵧ]
      [s₂      0      c₂     Pz]
      [0       0      0      1 ]
```

โดยที่:
```
Pₓ = c₁X₁ + s₁(Y₁ + Y₂ + Y₃) + c₁c₂(d₃ + X₂ + X₃) - c₁s₂(Z₂ + Z₃)
Pᵧ = s₁X₁ - c₁(Y₁ + Y₂ + Y₃) + c₂s₁(d₃ + X₂ + X₃) - s₁s₂(Z₂ + Z₃)
Pz = Z₁ + s₂(d₃ + X₂ + X₃) + c₂(Z₂ + Z₃)
```

### 9.2 Forward Kinematics

**สมการหลัก:**

```
FK: ℝ³ → ℝ³
FK(θ₁, θ₂, d₃) = [Pₓ, Pᵧ, Pz]ᵀ
```

โดยที่ Pₓ, Pᵧ, Pz คำนวณจาก Transformation Matrix ใน section 9.1

**Algorithm:**
1. รับ input: q = [θ₁, θ₂, d₃]ᵀ (θ₁, θ₂ ในหน่วยองศา)
2. แปลงมุมเป็นเรเดียน: θ₁_rad = θ₁ × π/180, θ₂_rad = θ₂ × π/180
3. คำนวณ c₁, s₁, c₂, s₂
4. คำนวณ Transformation Matrix แต่ละเฟรมตาม DH Parameters
5. คูณ Transformation Matrices ทั้ง 7 เฟรมตามลำดับ: T = T₁ × T₂ × ... × Tₑ
6. สกัดตำแหน่ง End Effector จาก T[0][3], T[1][3], T[2][3]
7. คืนค่า x = [Pₓ, Pᵧ, Pz]ᵀ

**Computational Complexity:** O(1)

### 9.3 Inverse Kinematics

**วิธีการแก้สมการเชิงวิเคราะห์:**

**สมการที่ 1: Base Rotation Angle**
```
θ₁ = atan2(Pᵧ, Pₓ) - atan2(Y₃, X₃)
```

**สมการที่ 2: Coordinate Transformation**

แปลงพิกัดเป้าหมายไปยังเฟรมหลังการหมุนฐาน:
```
x'ₒ = Pₓ·cos(-θ₁) - Pᵧ·sin(-θ₁) - X₁
y'ₒ = Pₓ·sin(-θ₁) + Pᵧ·cos(-θ₁) - Y₁
z'ₒ = Pz - Z₁
```

**สมการที่ 3: Prismatic Extension**

กำหนดให้:
```
dx = X₃ - X₁
dy = Y₃ - Y₁
dz = Z₃ - Z₁
r²ₑ = x'²ₒ + y'²ₒ + z'²ₒ
```

จากเงื่อนไข:
```
(dx + d₃)² + dy² + dz² = r²ₑ
```

แก้สมการกำลังสอง:
```
d₃ = -dx ± √(r²ₑ - dy² - dz²)
```

**เงื่อนไขการมีคำตอบ:**
```
r²ₑ ≥ dy² + dz²
```

**การเลือกคำตอบ:**
```
r²ᵣₑf = dx² + dy² + dz²

ถ้า r²ₑ > r²ᵣₑf : เลือก d₃ = -dx + √(r²ₑ - dy² - dz²)  (ยืดออก)
มิฉะนั้น        : เลือก d₃ = -dx - √(r²ₑ - dy² - dz²)  (หดเข้า)
```

**สมการที่ 4: Shoulder Angle**
```
dx_new = dx + d₃
θ₂ = atan2(z'ₒ, √(x'²ₒ + y'²ₒ)) - atan2(dz, √(dx²_new + dy²))
```

**Algorithm:**
1. ตรวจสอบว่า end position อยู่ใน workspace (optional)
2. คำนวณ θ₁ จากสมการที่ 1 (ผลลัพธ์เป็นเรเดียน)
3. แปลงพิกัดตามสมการที่ 2
4. คำนวณ discriminant และตรวจสอบว่า ≥ 0
5. คำนวณ d₃ จากสมการที่ 3 (เลือกคำตอบตามระยะห่าง)
6. คำนวณ θ₂ จากสมการที่ 4 (ผลลัพธ์เป็นเรเดียน)
7. แปลงมุมเป็นองศา: θ₁_deg = θ₁ × 180/π, θ₂_deg = θ₂ × 180/π
8. ตรวจสอบข้อจำกัดของข้อต่อ
9. คืนค่า q = [θ₁_deg, θ₂_deg, d₃]ᵀ

**Computational Complexity:** O(1)

**Singularity Analysis:**
- Wrist Singularity: เมื่อ d₃ ใกล้ขอบเขตต่ำสุด
- Shoulder Singularity: เมื่อ θ₂ ≈ 0° หรือ ≈ ±90°
- Workspace Boundary: เมื่อ d₃ อยู่ที่ค่าสูงสุดหรือต่ำสุด

### 9.4 Forward Differential Kinematics

**Jacobian Matrix:**

คำนวณจากอนุพันธ์ย่อยของ Forward Kinematics:

```
J(q) = ∂[Pₓ, Pᵧ, Pz]ᵀ/∂[θ₁, θ₂, d₃]
```

**ผลลัพธ์ (Reduced Jacobian 3×3):**

```
     [J₁₁  J₁₂  J₁₃]   [∂Pₓ/∂θ₁  ∂Pₓ/∂θ₂  ∂Pₓ/∂d₃]
J =  [J₂₁  J₂₂  J₂₃] = [∂Pᵧ/∂θ₁  ∂Pᵧ/∂θ₂  ∂Pᵧ/∂d₃]
     [J₃₁  J₃₂  J₃₃]   [∂Pz/∂θ₁  ∂Pz/∂θ₂  ∂Pz/∂d₃]
```

**องค์ประกอบของ Jacobian:**

กำหนดให้:
```
Y_sum = Y₁ + Y₂ + Y₃
Z_sum = Z₂ + Z₃
d₃_X_sum = d₃ + X₂ + X₃
```

คอลัมน์ที่ 1 (∂P/∂θ₁):
```
J₁₁ = -s₁X₁ + c₁Y_sum - s₁c₂(d₃_X_sum) + s₁s₂Z_sum
J₂₁ =  c₁X₁ + s₁Y_sum + c₁c₂(d₃_X_sum) - c₁s₂Z_sum
J₃₁ =  0
```

คอลัมน์ที่ 2 (∂P/∂θ₂):
```
J₁₂ = -c₁s₂(d₃_X_sum) - c₁c₂Z_sum
J₂₂ = -s₁s₂(d₃_X_sum) - s₁c₂Z_sum
J₃₂ =  c₂(d₃_X_sum) - s₂Z_sum
```

คอลัมน์ที่ 3 (∂P/∂d₃):
```
J₁₃ = c₁c₂
J₂₃ = s₁c₂
J₃₃ = s₂
```

**ความสัมพันธ์ความเร็ว:**
```
ẋ = J(q)q̇

[ẋ]   [J₁₁  J₁₂  J₁₃] [θ̇₁]
[ẏ] = [J₂₁  J₂₂  J₂₃] [θ̇₂]
[ż]   [J₃₁  J₃₂  J₃₃] [ḋ₃]
```

**หมายเหตุ:** ในการคำนวณจริง θ̇₁ และ θ̇₂ ต้องแปลงจากองศา/วินาทีเป็นเรเดียน/วินาที

**Computational Complexity:** O(1) สำหรับการคำนวณ Jacobian และการคูณเมทริกซ์

### 9.5 Inverse Differential Kinematics

**สมการหลัก:**
```
q̇ = J⁻¹(q)ẋ
```

**การคำนวณ Jacobian Inverse:**

```
J⁻¹ = adj(J)/det(J)
```

โดยที่:
```
det(J) = J₁₁(J₂₂J₃₃ - J₂₃J₃₂) - J₁₂(J₂₁J₃₃ - J₂₃J₃₁) + J₁₃(J₂₁J₃₂ - J₂₂J₃₁)
```

**Singularity Detection:**
```
Singularity exists if |det(J)| < ε
```

โดยที่ ε เป็นค่า tolerance (เช่น 10⁻⁶)

**Algorithm:**
1. คำนวณ J(q) จากสมการใน section 9.4
2. คำนวณ det(J) โดยใช้สูตร:
   ```
   det(J) = J₁₁(J₂₂J₃₃ - J₂₃J₃₂) - J₁₂(J₂₁J₃₃ - J₂₃J₃₁) + J₁₃(J₂₁J₃₂ - J₂₂J₃₁)
   ```
3. ถ้า |det(J)| < ε: แจ้งเตือน Singularity (ε = 10⁻¹⁰)
   มิฉะนั้น: คำนวณ J⁻¹ โดยใช้ Gauss-Jordan Elimination
4. คำนวณ q̇ = J⁻¹ẋ
5. แปลงความเร็วเชิงมุมจากเรเดียน/วินาทีเป็นองศา/วินาที
6. คืนค่า q̇ = [θ̇₁_deg, θ̇₂_deg, ḋ₃]ᵀ

**Computational Complexity:** 
- O(1) สำหรับการคำนวณ determinant และ inverse ของเมทริกซ์ 3×3

**Physical Interpretation:**

เมทริกซ์ Jacobian แสดงถึง:
- คอลัมน์ที่ i: ทิศทางความเร็วของ end effector เมื่อข้อต่อที่ i เคลื่อนที่ด้วยความเร็วหน่วย
- แถวที่ j: ส่วนประกอบของความเร็วในแกน j ที่เกิดจากการเคลื่อนที่ของข้อต่อทั้งหมด

### 9.6 การคำนวณและแสดงผล Workspace

#### 9.6.1 Singularity Detection Sampling

สำหรับการหา Singularity Positions ใช้การแบ่ง sample จุดในพื้นที่ workspace โดยอิงความละเอียดจากจำนวน input samples ที่กำหนด โดยมีจุดประสงค์เพื่อครอบคลุมจุดที่อยู่มุมใน workspace ที่อาจหาไม่ได้ด้วยวิธีปกติ

**จำนวน Samples สำหรับการหา Singularity:**

```
N₁ˢⁱⁿᵍ = max(8, ⌊N₁/2⌋)
N₂ˢⁱⁿᵍ = max(8, ⌊N₂/2⌋)
N₃ˢⁱⁿᵍ = max(5, ⌊N₃/2⌋)
```

โดยที่:
- N₁ˢⁱⁿᵍ: จำนวนจุดสำหรับ θ₁ (ใช้ค่าที่มากกว่าระหว่าง 8 หรือครึ่งหนึ่งของ theta1_samples)
- N₂ˢⁱⁿᵍ: จำนวนจุดสำหรับ θ₂ (ใช้ค่าที่มากกว่าระหว่าง 8 หรือครึ่งหนึ่งของ theta2_samples)
- N₃ˢⁱⁿᵍ: จำนวนจุดสำหรับ d₃ (ใช้ค่าที่มากกว่าระหว่าง 5 หรือครึ่งหนึ่งของ d3_samples)

**Algorithm:**
```python
singularity_positions = []
for θ₁ in θ₁_range:
    for θ₂ in θ₂_range:
        for d₃ in d₃_range:
            J = get_jacobian([θ₁, θ₂, d₃])
            if |det(J)| < ε:  # ε = singularity threshold
                p = forward_kinematics([θ₁, θ₂, d₃])
                singularity_positions.append(p)
```

#### 9.6.2 Workspace Point Generation

การสร้างจุดใน Workspace ใช้การแบ่ง samples อย่างสม่ำเสมอสำหรับ θ₁ และ θ₂:

**การแบ่ง Samples สำหรับ θ₁ และ θ₂:**

```
θ₁⁽ⁱ⁾ = θ₁ᵐⁱⁿ + (θ₁ᵐᵃˣ - θ₁ᵐⁱⁿ)/(N₁ - 1) · i,  i ∈ {0, 1, ..., N₁-1}

θ₂⁽ʲ⁾ = θ₂ᵐⁱⁿ + (θ₂ᵐᵃˣ - θ₂ᵐⁱⁿ)/(N₂ - 1) · j,  j ∈ {0, 1, ..., N₂-1}
```

**การแบ่ง Samples สำหรับ d₃:**

เนื่องจาก Workspace เป็นพื้นที่แบบทรงกลม จุดภายในจะอยู่ในระนาบเดียวกัน จึงใช้เฉพาะค่าขอบเขต:

```
d₃⁽ᵏ⁾ = {d₃ᵐⁱⁿ  ถ้า k = 0
        {d₃ᵐᵃˣ  ถ้า k = 1
```

**เซตของจุดทั้งหมดใน Workspace:**

```
W = {pₑₑ(θ₁⁽ⁱ⁾, θ₂⁽ʲ⁾, d₃⁽ᵏ⁾) : i ∈ [0, N₁-1], j ∈ [0, N₂-1], k ∈ {0,1}}
```

#### 9.6.3 Edge and Face Generation สำหรับการแสดงผล 3D

**คำนิยาม:**
- **Vertex (จุด):** จุดแต่ละจุดใน Workspace
- **Edge (เส้น):** เส้นที่เชื่อมต่อระหว่างสอง vertices
- **Face (หน้า):** พื้นผิวสี่เหลี่ยมที่สร้างจาก 4 vertices

**Point Mapping Function:**

ให้ Pₘₐₚ เป็นฟังก์ชันที่แมพ indices ไปยังตำแหน่งของ vertex:

```
Pₘₐₚ(i, j, k) ∈ ℤ,  (i, j, k) ∈ [0, N₁-1] × [0, N₂-1] × {0, 1}
```

**Edge Generation:**

เชื่อมต่อ edges เฉพาะที่ขอบของ Workspace เพื่อลดความซับซ้อนในการแสดงผล:

1. **Edges ตามแนว d₃** (เชื่อม d₃ᵐⁱⁿ กับ d₃ᵐᵃˣ ที่ขอบผิว):
```
Eₐ₃ = {(Pₘₐₚ(i,j,0), Pₘₐₚ(i,j,1)) : i ∈ {0, N₁-1} ∨ j ∈ {0, N₂-1}}
```

2. **Edges ตามแนว θ₂** (เชื่อมจุดที่ θ₂ ต่อเนื่องกัน):
```
Eθ₂ = {(Pₘₐₚ(i,j,k), Pₘₐₚ(i,j+1,k)) : i ∈ [0, N₁-1], j ∈ [0, N₂-2], k ∈ {0,1}}
```

3. **Edges ตามแนว θ₁** (เชื่อมจุดที่ θ₁ ต่อเนื่องกัน):
```
Eθ₁ = {(Pₘₐₚ(i,j,k), Pₘₐₚ(i+1,j,k)) : i ∈ [0, N₁-2], j ∈ [0, N₂-1], k ∈ {0,1}}
```

**Face Generation:**

สร้าง faces สำหรับผิวของ Workspace โดยหลีกเลี่ยงการซ้ำซ้อนด้วยการใช้ set union:

1. **Faces ในระนาบ θ₁-θ₂** (d₃ คงที่):
```
Fₐ = {[Pₘₐₚ(i,j,k), Pₘₐₚ(i+1,j,k), Pₘₐₚ(i+1,j+1,k), Pₘₐₚ(i,j+1,k)] : 
      i ∈ [0, N₁-2], j ∈ [0, N₂-2], k ∈ {0,1}}
```

2. **Faces ที่ θ₂ = θ₂ᵐⁱⁿ หรือ θ₂ = θ₂ᵐᵃˣ:**
```
Fᵦ = {[Pₘₐₚ(i,j,0), Pₘₐₚ(i+1,j,0), Pₘₐₚ(i+1,j,1), Pₘₐₚ(i,j,1)] : 
      i ∈ [0, N₁-2], j ∈ {0, N₂-1}}
```

3. **Faces ที่ θ₁ = θ₁ᵐⁱⁿ หรือ θ₁ = θ₁ᵐᵃˣ:**
```
Fᴄ = {[Pₘₐₚ(i,j,0), Pₘₐₚ(i,j+1,0), Pₘₐₚ(i,j+1,1), Pₘₐₚ(i,j,1)] : 
      i ∈ {0, N₁-1}, j ∈ [0, N₂-2]}
```

**เซตของ Faces ทั้งหมด:**
```
F = Fₐ ∪ Fᵦ ∪ Fᴄ
```

#### 9.6.4 การปรับปรุงการแสดงผล

**คุณสมบัติเพิ่มเติม:**
1. **Toggle Visibility:** สามารถซ่อน/แสดง vertices และ faces เพื่อความชัดเจนในการมองในมุมต่างๆ
2. **View Control:** Sliders สำหรับควบคุมมุมมอง (Elevation และ Azimuth) เพื่อป้องกันการบิดเบือนของมุมมอง
3. **Adjustable Resolution:** สามารถปรับความละเอียดของ samples เพื่อสมดุลระหว่างความละเอียดและประสิทธิภาพการคำนวณ

**Computational Complexity สำหรับการสร้าง Workspace:**
- Workspace Points: O(N₁ × N₂ × 2)
- Edge Generation: O(N₁ × N₂)
- Face Generation: O(N₁ × N₂)
- Overall: O(N₁ × N₂)

---

## สัญลักษณ์ที่ใช้ (Notation)

### Configuration Space Variables
- θ₁: มุมหมุนของข้อต่อฐาน (Base rotation angle)
- θ₂: มุมหมุนของข้อต่อหัวไหล่ (Shoulder angle)
- d₃: ระยะยืดของข้อต่อเลื่อน (Prismatic extension)
- q: Configuration space vector = [θ₁, θ₂, d₃]ᵀ
- q̇: Configuration space velocity = [θ̇₁, θ̇₂, ḋ₃]ᵀ

### Task Space Variables
- Pₓ, Pᵧ, Pᵧ: ตำแหน่งของ end effector ในแกน X, Y, Z
- x: Task space position vector = [Pₓ, Pᵧ, Pᵧ]ᵀ
- ẋ: Task space velocity = [ẋ, ẏ, ż]ᵀ

### DH Parameters
- aᵢ: Link length (ระยะห่างตามแกน X)
- αᵢ: Link twist (มุมระหว่างแกน Z)
- dᵢ: Link offset (ระยะห่างตามแกน Z)
- θᵢ: Joint angle (มุมระหว่างแกน X)

### Transformation Matrices
- ⁱ⁻¹Tᵢ: Homogeneous transformation from frame i-1 to frame i
- ⁰Tₑ: Transformation from base frame to end effector frame
- J(q): Jacobian matrix
- J⁻¹: Jacobian inverse
- J⁺: Moore-Penrose pseudoinverse

### Trigonometric Notation
- cᵢ ≡ cos(θᵢ)
- sᵢ ≡ sin(θᵢ)
- Cθ ≡ cos(θ)
- Sθ ≡ sin(θ)

### Link Geometry
- Xᵢ, Yᵢ, Zᵢ: ตำแหน่งโกลบอลของข้อต่อที่ i (Joint Global Position)
- dx, dy, dz: ระยะห่างระหว่างข้อต่อ

---

## License

**Copyright © 2025 by FIBO**  
Institute of Field Robotics, King Mongkut's University of Technology Thonburi  
Senior Thesis FRAB (FIBO Robotics and Automation: Bachelor)

All rights reserved. This document and associated software are proprietary materials developed as part of an academic thesis project.