# Python Toolbox for Kinematics of 3-DOF (RRP) Polar Robotic Arm

## 9. รายละเอียดการดำเนินการ
### 9.5 การคำนวณการแบ่ง sample สำหรับ plot 3D ใน workspace

#### Singularity Detection Sampling

สำหรับการหา singularity positions จะใช้การ sample จุดระหว่างค่าใน workspace ทั่วไปโดยอิงความละเอียดตามจำนวน input Sample ที่สามารถคำนวณได้

$$N_1^{sing} = \max(8, \lfloor N_1/2 \rfloor)$$

$$N_2^{sing} = \max(8, \lfloor N_2/2 \rfloor)$$

$$N_3^{sing} = \max(5, \lfloor N_3/2 \rfloor)$$

- $N_1^{sing}$ คือจำนวนจุดที่จะคำนวณ singularity ของ $\theta_1$: ใช้สูตร $\max(8, \lfloor N_1/2 \rfloor)$ ซึ่งหมายถึงใช้ค่าที่มากกว่า ระหว่าง 8 หรือครึ่งหนึ่งของ theta1_samples
- $N_2^{sing}$ คือจำนวนจุดที่จะคำนวณ singularity ของ $\theta_2$: ใช้สูตร $\max(8, \lfloor N_2/2 \rfloor)$ ซึ่งหมายถึงใช้ค่าที่มากกว่า ระหว่าง 8 หรือครึ่งหนึ่งของ theta2_samples
- $N_3^{sing}$ คือจำนวนจุดที่จะคำนวณ singularity ของ $d_3$: ใช้สูตร $\max(5, \lfloor N_3/2 \rfloor)$ ซึ่งหมายถึงใช้ค่าที่มากกว่า ระหว่าง 5 หรือครึ่งหนึ่งของ d3_samples

**ตัวอย่าง:**
- ถ้า workspace ใช้ (theta1_samples=25, theta2_samples=25, d3_samples=15)
- singularity detection จะใช้ samples: (max(8, 12)=12, max(8, 12)=12, max(5, 7)=7)

#### Workspace Generation (get_workspace)
ในการแบ่ง sample สำหรับค่าที่ plot ใน workspace จะอ้างอิงความละเอียดตาม sample ที่ตั้งค่าไว้ โดยใน $\theta_1,\theta_2$ จะคำนวนได้เป็นตามสมการนี้
$$\theta_1^{(i)} = \theta_1^{min} + \frac{\theta_1^{max} - \theta_1^{min}}{N_1 - 1} \cdot i, \quad i \in \{0, 1, \ldots, N_1-1\}$$

$$\theta_2^{(j)} = \theta_2^{min} + \frac{\theta_2^{max} - \theta_2^{min}}{N_2 - 1} \cdot j, \quad j \in \{0, 1, \ldots, N_2-1\}$$

$N$ คือจำนวณ sample ที่ตั้งค่าไว้

สำหรับ $d_3$ หากเป็นช่วงระหว่างกลาง จะใช้เป็นค่า $d$ ที่ต่ำที่สุด และค่า $d$ ที่สูงที่่สุด

$$d_3^{(k)} = \begin{cases} d_3^{min} & k = 0 \\ d_3^{max} & k = 1 \end{cases}$$

เมื่อได้ parameter ของทุกตัวแล้ว จะนำมาคำนวณหา end effector ด้วย forward kinematic ของทุกตัว และนำมา plot ค่าภายใน workspace โดยทุกค่าที่จะอยู่ใน workspace จะได้เป็นดังสมการนี้

$$\mathcal{W} = \{\mathbf{p}_{ee}(\theta_1^{(i)}, \theta_2^{(j)}, d_3^{(k)}) : i \in [0, N_1-1], j \in [0, N_2-1], k \in \{0,1\}\}$$


#### Edge and Face Generation

หลังจากที่ได้จุดใน workspace ทุกตัว จะนำค่าที่ได้ plot เป็น 3D โดยนิยามให้จุดแต่ละจุดเรียกว่า vertex และเส้นที่เชื่อมระหว่างสองจุดเรียกว่า edge และการ plot สี่เหลี่ยมด้วย 4 vertex ว่า face

ในการ plot แต่ละ vertex สามารถ plot ได้ดังนี้

ให้ $P_{map}$ คือจุดที่จะ plot เป็น vertex ใน 3D workspace

$$P_{map}(i, j, k) \in \mathbb{Z}, \quad (i, j, k) \in [0, N_1-1] \times [0, N_2-1] \times \{0, 1\}$$

สำหรับการเชื่อม edge จะมีอยู่ 3 รูปแบบ ได้แก่

ให้ $E$ คือ edge ที่จะเชื่อมในแต่ละ vertex

1. เชื่อม edge ของ d₃_min กับ d₃_max ในกรณีที่อยู่ ณ ตรงขอบผิวของ Workspace
$$E_{d_3} = \{(P_{map}(i,j,0), P_{map}(i,j,1)) : i \in \{0, N_1-1\} \lor j \in \{0, N_2-1\}\}$$

1. เชื่อม edge ในกรณีที่จุดที่ plot มี θ₂ ที่ต่อเนื่องกัน
$$E_{\theta_2} = \{(P_{map}(i,j,k), P_{map}(i,j+1,k)) : i \in [0, N_1-1], j \in [0, N_2-2], k \in \{0,1\}\}$$

1. เชื่อม edge ในกรณีที่จุดที่ plot มี θ₁ ที่ต่อเนื่องกัน
$$E_{\theta_1} = \{(P_{map}(i,j,k), P_{map}(i+1,j,k)) : i \in [0, N_1-2], j \in [0, N_2-1], k \in \{0,1\}\}$$

ในการ plot face จะคำนวณ face ของทุกรูปแบบมา union กัน ซึ่งมี 3 กรณีดังนี้

1. กรณี plot face ในช่วง θ₁ และ θ₂ ต่อเนื่องกัน และมีค่า d₃ เท่ากัน
$$F_A = \{[P_{map}(i,j,k), P_{map}(i+1,j,k), P_{map}(i+1,j+1,k), P_{map}(i,j+1,k)] : \forall i,j,k\}$$

2. กรณี plot face ในขณะที่ θ₁ อยู่ตรง limit
$$F_B = \{[P_{map}(i,j,0), P_{map}(i+1,j,0), P_{map}(i+1,j,1), P_{map}(i,j,1)] : i \in [0, N_1-2], j \in \{0, N_2-1\}\}$$

3. กรณี plot face ในขณะที่ θ₂ อยู่ตรง limit
$$F_C = \{[P_{map}(i,j,0), P_{map}(i,j+1,0), P_{map}(i,j+1,1), P_{map}(i,j,1)] : i \in \{0, N_1-1\}, j \in [0, N_2-2]\}$$

หลังจากนั้นจะทำทุก face มา union กันและ plot ค่า

$$\ F = F_A \cup F_B \cup F_C, \quad$$



