# Python Toolbox for Kinematics of 3-DOF (RRP) Polar Robotic Arm

## Project Team
- ชยากร ชื่นประเสริฐ (66340500012)
- ศักดิ์ไชย ธนภาสวัฒน์ (66340500052)
- ธัญนพ ศรีวานิช (66340500073)

**Institution:** Institute of Field Robotics, King Mongkut's University of Technology Thonburi  
**Program:** Senior Thesis FRAB (FIBO Robotics and Automation: Bachelor)  
**Year:** 2025

---

## บทคัดย่อ (Abstract)

โปรเจคนี้ เป็นโปรเจคสำหรับการสร้าง Toolbox ในภาษา Python เพื่อใช้ในการคำนวณหา Configuration Space และอัตราการเปลี่ยนแปลงของ Configuration Space ของแต่ละข้อต่อโดยอ้างอิงจาก Task Space และระยะเวลาที่ใช้ในการเคลื่อนที่ของ End Effector และทำการคำนวณหา Task Space และอัตราการเปลี่ยนแปลงของ Task Space ของ End Effector โดยอ้างอิงจาก Configuration Space และระยะเวลาที่ใช้ในการเคลื่อนที่ของแต่ละข้อต่อ

**คำสำคัญ:** Python Toolbox, Polar Robot Arm, Configuration Space และ Task Space

---

## 1. จุดประสงค์ของโครงการ (Project Objectives)

1. เพื่อศึกษา Polar Robot Arm ในปริภูมิ 3 มิติ
2. เพื่อพัฒนา Python Toolbox สำหรับการทำ Forward Kinematics, Inverse Kinematics และ Differential Kinematics ของ Polar Robot Arm ในปริภูมิ 3 มิติ ที่สามารถ Visualization ได้ทั้ง Pose และ Animation ของ Polar Robot Arm

---

## 2. ขอบเขต (Project Scope)

### 2.1 Python Toolbox Features
- **Pose Kinematics เท่านั้น**
  - Forward Kinematics
  - Inverse Kinematics
  - Forward Differential Kinematics
  - Inverse Differential Kinematics
  - ไม่ศึกษาในส่วนของ Acceleration

### 2.2 Polar Robotic Arm Specifications
- **ปริภูมิ 3 มิติ**
  - คำนวณตำแหน่งและความเร็ว Configuration Space
  - คำนวณตำแหน่งและความเร็ว Task Space ใน X-Y-Z Axis
  - คำนวณ Work Space
  - ไม่มีการศึกษาในส่วนของ End Effecter
  - ไม่มีสิ่งกีดขวางในปริภูมิ 3 มิติ

### 2.3 Robot Configuration
- **จำนวน Link:** 4 links
- **จำนวน Actuator:** 3 ตัว (RRP configuration)
- **จำนวน Joint:** สูงสุด 2 joints ต่อ 1 link

### 2.4 Input Parameters
1. **ระบุขนาดของ Link:** โดยระบุเป็นเซตของเวกเตอร์ โดยที่เซตของเวกเตอร์จะเป็นระยะห่างของ joint<sub>i-1</sub> กับ joint<sub>i</sub> โดยแต่ละเวกเตอร์เป็นระยะทางที่ต่อเนื่องกัน โดยมี v₁ เป็นเวกเตอร์ที่ต่อกับ joint<sub>i-1</sub> และมี v<sub>n</sub> เป็นเวกเตอร์ที่ต่อกับ joint<sub>i</sub>

2. **ระบุข้อจำกัดของแต่ละ Joint:** เป็นเมทริกซ์ของแกน X-Y-Z โดยแต่ละแกนจะเก็บค่าเป็นคู่
   - **Revolute joint:** ระบุค่าเป็นมุมที่ทำได้สูงสุดในทิศทางบวกลบตามกฎมือขวาในแต่ละแกน
   - **Prismatic joint:** ระบุค่าเป็นระยะผลักออกที่ไกลที่สุดและระยะดึงกลับ

3. **ระบุ Task Space:** ของ End Effector และระบุระยะเวลาที่ใช้ในการเคลื่อนที่ของ End Effector

4. **ระบุ Configuration Space:** ของแต่ละข้อต่อ และระบุระยะเวลาที่ใช้ในการเคลื่อนที่ของแต่ละข้อต่อ

---

## 3. ทฤษฎีที่เกี่ยวข้อง (Theoretical Background)

### 3.1 Robot Specification
3-DOF (RRP) Polar Robotic Arm เป็นแขนหุ่นยนต์แบบทรงกลม (Spherical Configuration) ประกอบด้วย:
- **3 ข้อต่อ (joints):** 
  - 1 Revolute base (R) - θ₁
  - 1 Revolute joint (R) - θ₂
  - 1 Prismatic joint (P) - d₃
- **4 ลิงก์ (links)**

### 3.2 DH Parameters (Denavit-Hartenberg)
การกำหนด Frame ของหุ่นยนต์โดยใช้หลักการ DH Parameters:

| i | a<sub>i</sub> | α<sub>i</sub> | d<sub>i</sub> | θ<sub>i</sub> |
|---|---------------|---------------|---------------|---------------|
| 1 | x₁ | +90° | x₂ | θ₁ |
| 2 | 0 | -90° | 0 | θ₂ |
| 3 | 0 | 0 | d₃ | 0 |

### 3.3 Transformation Matrix
Transformation Matrix จากเฟรม i-1 ไปเฟรม i:

```
ⁱTᵢ₋₁ = Tᵤ(θᵢ) × Tᵤ(dᵢ) × Tₓ(aᵢ) × Tₓ(αᵢ)
```

หรือ

```
ⁱ⁻¹Tᵢ = [Cθᵢ  -SθᵢCαᵢ   SθᵢSαᵢ   aᵢCθᵢ]
        [Sθᵢ   CθᵢCαᵢ  -CθᵢSαᵢ   aᵢSθᵢ]
        [0     Sαᵢ      Cαᵢ      dᵢ   ]
        [0     0        0        1    ]
```

### 3.4 Workspace Calculation
การคำนวณ Workspace โดยใช้ Jacobian Matrix และ Singularity Analysis:

**Jacobian Matrix:**
```
J(q) = [∂Φ(q)/∂qᵢ]
```

**Singularity Condition:**
```
|Φ_q| = 0, Where Φ_q = ∂Φ(q)/∂qᵢ
```

**Singularity Sets:**
- S₁ = {p∈Rⁿ ; dim Null(J(q)) ≥ 1, for some q}
- S₂ = {p∈Rⁿ ; dim Null(J(q*)) ≥ 1, for some q*}
- S₃ = {p∈Rⁿ ; [qᵢ⁰, qⱼ⁰], for i,j = 1 to n; i ≠ j}

**Total Workspace:**
```
S = S₁ ∪ S₂ ∪ S₃
Ψᵢ(μᵢ) = [Ψ₁(μ₁), Ψ₂(μ₂), ..., Ψₘ(μₘ)]
```

---

## 4. System Overview

### 4.1 System Diagram
```
Input Parameters → Verify Input → Calculate Functions → Output
     ↓                  ↓               ↓                ↓
- Link Vectors    - Validate     - 3D Workspace    - Task Space
- Joint Limits    - Check Range  - Forward Kin.    - Config Space
- Task Space                      - Inverse Kin.    - Velocity
- Config Space                    - Diff. Kin.      - Visualization
- Time Duration                   - Error Handle    
```

### 4.2 Main Functions

#### Function 1: Robot Specification Setup
- **Input:** 
  - ขนาดของ Link (Vector set)
  - ข้อจำกัดของแต่ละ Joint (X-Y-Z Matrix)
- **Output:** 
  - Robot configuration object
- **Description:** 
  - รับค่าขนาดของ Link ทุก Link และข้อจำกัดของแต่ละข้อต่อ
  - Revolute joint: มุมหมุนสูงสุด/ต่ำสุดในทิศทางบวกลบ
  - Prismatic joint: ระยะยืด/หดสูงสุด/ต่ำสุด

#### Function 2: 3D Workspace Calculation
- **Input:** Robot configuration
- **Output:** 3D Workspace boundaries with singularity points
- **Description:** 
  - คำนวณ 3D Workspace ของแขนหุ่นยนต์
  - กำหนดช่วงสูงสุดและต่ำสุดของเส้นทางที่ไปได้
  - คำนวณหามุมของข้อต่อที่ทำให้เกิด Singularity
  - ทดแทน Singularity points ลงบน Workspace

#### Function 3: Forward Kinematics
- **Input:** Configuration Space
- **Output:** Task Space
- **Description:** 
  - รับค่า Configuration Space (joint angles/positions)
  - แปลงไปเป็น Task Space (end effector position)

#### Function 4: Inverse Kinematics
- **Input:** Task Space
- **Output:** Configuration Space
- **Description:** 
  - รับค่า Task Space (end effector position)
  - แปลงไปเป็น Configuration Space (joint angles/positions)
  - ตรวจสอบว่า Task Space อยู่ใน Workspace
  - Error: "No Configuration Space" หากไม่มีคำตอบ

#### Function 5: Forward Differential Kinematics
- **Input:** Configuration Space velocity (q̇)
- **Output:** Task Space velocity (ẋ)
- **Description:** 
  - รับอัตราการเปลี่ยนแปลงของ Configuration Space
  - แปลงไปเป็นอัตราการเปลี่ยนแปลง Task Space

#### Function 6: Inverse Differential Kinematics
- **Input:** Task Space velocity (ẋ)
- **Output:** Configuration Space velocity (q̇)
- **Description:** 
  - รับอัตราการเปลี่ยนแปลงของ Task Space
  - แปลงไปเป็นอัตราการเปลี่ยนแปลง Configuration Space

#### Function 7: Task Space Calculation (Complete)
- **Input:** Configuration Space + Time duration
- **Output:** Task Space + Task Space velocity
- **Process:**
  1. คำนวณ Forward Kinematics → Task Space
  2. คำนวณ velocity จาก time duration
  3. คำนวณ Forward Differential Kinematics → Task Space velocity
- **Error Handling:**
  - "Invalid input" หากข้อมูลรับผิด
  - "No Task Space" หาก Configuration Space ไม่อยู่ในช่วงที่ข้อต่อไปได้

#### Function 8: Configuration Space Calculation (Complete)
- **Input:** Task Space + Time duration
- **Output:** Configuration Space + Configuration Space velocity
- **Process:**
  1. ตรวจสอบ Task Space บน Workspace
  2. คำนวณ Inverse Kinematics → Configuration Space
  3. คำนวณ velocity จาก time duration
  4. คำนวณ Inverse Differential Kinematics → Configuration Space velocity
- **Error Handling:**
  - "Invalid input" หากข้อมูลรับผิด
  - "No Configuration Space" หาก Task Space ไม่อยู่บน Workspace หรือ Inverse Kinematics ไม่มีคำตอบ

#### Function 9: 3D Model Display
- **Input:** Robot configuration + Current pose
- **Output:** 3D visualization of robot
- **Description:** แสดง 3D Model ของแขนหุ่นยนต์

#### Function 10: 3D Animation
- **Input:** Trajectory data (from Function 7 or 8)
- **Output:** Animated 3D visualization
- **Description:** 
  - แสดงการเคลื่อนไหวของ 3D Model ของแขนหุ่นยนต์
  - รับข้อมูลจาก Task Space Calculation หรือ Configuration Space Calculation

---

## 5. เนื้อหาในรายวิชาที่เกี่ยวข้อง (Related Course Topics)

1. Transformation of Coordinate Frame
2. Forward Kinematics
3. Inverse Kinematics
4. Forward Differential Kinematics
5. Inverse Differential Kinematics

---

## 6. ผลการศึกษาที่คาดหวัง (Expected Results)

1. สามารถสร้าง Python Toolbox ที่สามารถใช้ในการคำนวณและได้ผลลัพธ์ตำแหน่งเชิงมุมของแต่ละข้อต่อได้ตำแหน่งของแขนหุ่นยนต์ปลายมือได้และความเร็วของแต่ละข้อต่อได้

2. Python Toolbox สามารถคำนวณ Workspace ได้อย่างถูกต้อง

---

## 7. รายละเอียดโครงการ (Project Timeline)

| ลำดับ | การดำเนินงาน | สัปดาห์ที่ 1 | สัปดาห์ที่ 2 | สัปดาห์ที่ 3 | สัปดาห์ที่ 4 | สัปดาห์ที่ 5 | สัปดาห์ที่ 6 | สัปดาห์ที่ 7 |
|-------|---------------|--------------|--------------|--------------|--------------|--------------|--------------|--------------|
| 1 | จัดทำ Proposal | ✓ | ✓ | | | | | |
| 2 | แก้ไข Proposal | | | ✓ | | | | |
| 3 | ศึกษาหาข้อมูลเกี่ยวกับ Pose Kinematic | ✓ | ✓ | | | | | |
| 4 | ศึกษาหาข้อมูลเกี่ยวกับ 3D Workspace | ✓ | ✓ | | | | | |
| 5 | จัดทำ Forward Kinematic | | | ✓ | | | | |
| 6 | จัดทำ Inverse Kinematic | | | ✓ | | | | |
| 7 | จัดทำ Forward Differential Kinematic | | | ✓ | | | | |
| 8 | จัดทำ Inverse Differential Kinematic | | | ✓ | ✓ | | | |
| 9 | จัดทำ 3D Workspace | | | ✓ | ✓ | | | |
| 10 | ทำ Kinematic Visualization | | | | ✓ | ✓ | | |
| 11 | ทดสอบและปรับปรุง | | | | | ✓ | ✓ | |
| 12 | จัดทำ User Manual Document | | | | | | ✓ | |
| 13 | จัดทำรายงาน | | | | | | ✓ | ✓ |
| 14 | นำเสนอ และ Project Demo | | | | | | | ✓ |

---

## 8. เอกสารอ้างอิง (References)

[1] Adil Khan. "Forward Kinematic Modeling of a 3-DOF (RRP) Polar Robotic Arm Using OOPS", 2009

[2] Khushdeep Goyal, Davinder Sethi "AN ANALYTICAL METHOD TO FIND WORKSPACE OF A ROBOTIC MANIPULATOR", 2010

---

## 9. รายละเอียดการดำเนินการ
### 9.1 การคำนวณ Tranformations Matrix
เพื่อให้ระบบและฟังก์ชันทั้งหมดทำงานได้อย่างถูกต้อง จำเป็นต้องมีการการคำนวณ DH-Parameter ของแขนกลให้สอดคล้องกับ Input link ที่ใส่เข้ามา

รูป

จากรูปสามารถทำเป็น ตาราง DH-Parameter ได้ดังนี้
| i | a<sub>i</sub> | α<sub>i</sub> | d<sub>i</sub> | θ<sub>i</sub> |
|---|---------------|---------------|---------------|---------------|
| 1 | 0 | 0 | 0 | θ₁ |
| 2 | X₁ | 0 | Z<sub>1<sub> | 0 |
| 3 | 0 | 90° | Y<sub>1<sub> | θ₂ |
| 4 | X<sub>2<sub>  | 0 | Y<sub>2<sub> | 90° |
| 5 | Z<sub>2<sub> | 90° | 0 | 0 |
| 6 | Z<sub>3<sub> | 0 | X<sub>3</sub> + d₃| 90° |
| E | Y<sub>3<sub> | 90° | 0 | 90° |

จากนั้นทะการ Transfomation จากเฟรม 0 ไป เฟรม E จากสูตร

    ⁱTᵢ₋₁ = Tₓ(aᵢ) × Tₓ(αᵢ) × Tᵤ(θᵢ) × Tᵤ(dᵢ)

จะได้

    ⁰Tₑ = | c₁c₂   −s₁    −c₁s₂   c₁X₁ + s₁(Y₁ + Y₂ + Y₃) + c₁c₂(d + X₂ + X₃) − c₁s₂(Z₂ + Z₃) |
          | c₂s₁    c₁    −s₁s₂   s₁X₁ − c₁(Y₁ + Y₂ + Y₃) + c₂s₁(d + X₂ + X₃) − s₁s₂(Z₂ + Z₃) |
          | s₂      0      c₂     Z₁ + s₂(d + X₂ + X₃) + c₂(Z₂ + Z₃)                          |
          | 0       0      0      1                                                           |

### 9.2 การคำนวณ Forward Kinematics
จากการคำนวณ Transformation Matrix สามารถคำนวณ Forward Kinematics ได้ดังนี้
    
                   |c₁X₁ + s₁(Y₁ + Y₂ + Y₃) + c₁c₂(d + X₂ + X₃) − c₁s₂(Z₂ + Z₃)|
    FK(θ₁,θ₂,d₃) = |s₁X₁ − c₁(Y₁ + Y₂ + Y₃) + c₂s₁(d + X₂ + X₃) − s₁s₂(Z₂ + Z₃)|
                   |Z₁ + s₂(d + X₂ + X₃) + c₂(Z₂ + Z₃)                         |
### 9.3 การคำนวณ Inverse Kinematics
สมการที่ 1: Base Rotation

    θ₁ = atan2(y, x) - atan2(y₃, x₃)

สมการที่ 2: Coordinate Transformation

    x<sub>org</sub> = x·cos(-θ₁) - y·sin(-θ₁) - x₁
    y<sub>org</sub> = x·sin(-θ₁) + y·cos(-θ₁) - y₁
    z<sub>org</sub> = z - z₁

สมการที่ 3: Prismatic Extension

    d₃ = -Δx ± √(r<sub>target</sub>² - Δy² - Δz²)

โดยที่: 

       Δx = x₃ - x₁, Δy = y₃ - y₁, Δz = z₃ - z₁
       r<sub>target</sub>² = x<sub>org</sub>² + y<sub>org</sub>² + z<sub>org</sub>²

สมการที่ 4: Shoulder Angle

    θ₂ = atan2(z<sub>org</sub>, √(x<sub>org</sub>² + y<sub>org</sub>²)) - atan2(Δz, √((Δx + d₃)² + Δy²))

### 9.4 การคำนวณ Inverse Differential Kinematics

## License
**Copyright © 2021 by FIBO**  
Institute of Field Robotics, King Mongkut's University of Technology Thonburi  
Senior Thesis FRAB (FIBO Robotics and Automation: Bachelor)