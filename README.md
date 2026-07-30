# 宇树 G1 搬箱仿真系统

## 概述

本项目构建于 MuJoCo 物理引擎之上，面向宇树 G1 人形机器人（含灵巧手）实现**拾取—搬运—放置**闭环仿真。系统包含 10 个待搬运箱体、拾取工作台与放货桌面，支持：

| 模式 | 入口 | 说明 |
|------|------|------|
| 手动遥操作 | `python scripts/run_sim.py` | 键盘设定路点，自主完成够取/抱箱/起身 |
| 自动演示 | `python scripts/run_sim.py --demo` | 有限状态机驱动全流程 |
| 无头评测 | `python scripts/run_sim.py --demo --headless` | 无 GUI，用于自动化测试 |

底层行走采用 `unitree_rl_gym` 预训练 TorchScript 策略；上层任务调度、抓取与上肢规划为自定义控制器，二者通过分层控制架构解耦。

---

## 环境要求

- 操作系统：Ubuntu 20.04 / 22.04（其他发行版未经充分验证）
- 运行时：[Conda](https://docs.conda.io/)（Miniconda 或 Anaconda）
- 依赖项：Python 3.10、MuJoCo、PyTorch（见 `environment.yml`）
- 网络：首次部署需下载 G1 机器人模型与行走策略权重

---

## 部署与运行

```bash
# 安装依赖、模型与策略（需 Conda；未安装时脚本会打印 Miniconda 安装指引）
bash setup.sh
# 可选：自动安装 Miniconda 到 ~/miniconda3 并继续部署
# bash setup.sh --install-miniconda

# 激活 Conda 环境
conda activate unitree_pick_place

# 手动遥操作
python scripts/run_sim.py

# 自动演示
python scripts/run_sim.py --demo

# 无头自动评测
python scripts/run_sim.py --demo --headless
```

---

## 演示视频

官方基线自动 demo（2 箱搬运），源文件：`assets/demo_2boxes.mp4`

https://github.com/user-attachments/assets/9517a277-e760-495d-aa7d-0f57cf401253

自行录制：

```bash
# 手动模式 + 录制（WASD / G / R 等照常使用）
python scripts/run_sim.py --record

# 自动 demo + 录制，搬完 2 箱自动停止
python scripts/run_sim.py --demo --record --record-output assets/demo_2boxes.mp4 --record-stop-boxes 2

# 指定输出路径
python scripts/run_sim.py --record --record-output assets/my_demo.mp4
```

无显示器批量生成演示片（维护者 / CI 用）：

```bash
MUJOCO_GL=egl python scripts/record_demo.py --target-boxes 2
```

---

## 任务定义

设仿真场景中共有 \(N=10\) 个箱体，编号 \(\mathcal{B}=\{0,1,\ldots,9\}\)。每个箱体 \(i\) 的状态由位姿 \(\mathbf{p}_{i}\in\mathbb{R}^3\) 与线速度 \(\mathbf{v}_{i}\in\mathbb{R}^3\) 描述。

| 符号 / 参数 | 取值 / 含义 |
|-------------|-------------|
| 机器人平台 | 宇树 G1 + 12-DoF 灵巧手 |
| 拾取区 \(\mathcal{P}\) | 中心 \((0.15, 0, 0.85)\) m，箱阵列 \(2\times5\) |
| 放置区 \(\mathcal{D}\) | 中心 \((3.5, 0, 0.85)\) m，半尺寸 \((0.45, 0.45, 0.12)\) m |
| 放货桌面高度 | \(h_{\mathrm{table}} = 0.73\) m |
| 成功放置集合 \(\mathcal{S}\subseteq\mathcal{B}\) | 初始为空，逐箱递增 |

**任务目标：** 将所有未放置箱体 \(i\in\mathcal{B}\setminus\mathcal{S}\) 依次搬运至放置区并完成稳定放置，即 \(|\mathcal{S}|=N\)。

---

## 系统架构

```
┌─────────────────────────────────────────────────────────┐
│  run_sim.py          主循环 / 按键队列 / 可视化           │
├─────────────────────────────────────────────────────────┤
│  DemoController      有限状态机（`src/controllers/demo.py`）│
│  ManualController    路点导航 + 够取/抱箱（`src/controllers/manual.py`）│
├─────────────────────────────────────────────────────────┤
│  PolicyWalker        RL 行走 + 速度辅助 + 上肢锁定（`src/control/`）│
│  G1LocomotionPolicy  观测构建 / 策略推理 / PD 力矩          │
├─────────────────────────────────────────────────────────┤
│  BoxPickPlaceEnv     MuJoCo 步进 / 抓取 / 箱体同步（`src/env/`）│
│  BoxPickPlaceTask    放置判定                             │
└─────────────────────────────────────────────────────────┘
```

每个仿真步 \(k\) 的执行顺序为：

`stabilize_boxes` → `pre_physics` → `mj_step(Δt)` → `post_physics` → `sync_carried_box` → `task.update`

其中 \(\Delta t = 0.002\) s（见 `config/task.yaml`）。

---

## 算法设计

### 1. 强化学习行走策略

#### 1.1 观测向量

策略网络 \(\pi_{\theta}\) 的输入观测 \(\mathbf{o}_{k} \in \mathbb{R}^{n_{\mathrm{obs}}}\) 由本体感知与步态相位构成（列向量拼接）：

```math
\mathbf{o}_k = \begin{bmatrix}
\mathbf{\omega}_k \\
\mathbf{g}_k \\
\mathbf{c}_k \odot \mathbf{s}_{\mathrm{cmd}} \\
\tilde{\mathbf{q}}_k \\
\mathbf{v}_k^{q,\mathrm{sc}} \\
\mathbf{a}_{k-1} \\
\sin\phi_k \\
\cos\phi_k
\end{bmatrix}
```

各分量定义如下：

| 符号 | 维度 | 含义 |
|------|------|------|
| \(\mathbf{\omega}_{k}\) | \(\mathbb{R}^3\) | 骨盆角速度（经 `ang_vel_scale` 缩放） |
| \(\mathbf{g}_{k}\) | \(\mathbb{R}^3\) | 重力方向在机体坐标系下的投影 |
| \(\mathbf{c}_{k}\) | \(\mathbb{R}^3\) | 速度指令 \(\left[v_{x}, v_{y}, \omega_{z}\right]^\top\)（经 `cmd_scale` 缩放） |
| \(\tilde{\mathbf{q}}_{k}\) | \(\mathbb{R}^{12}\) | 腿部关节角偏差 \((\mathbf{q}_{k} - \mathbf{q}_{0}) \cdot s_{q}\) |
| \(\mathbf{v}_{k}^{q,\mathrm{sc}}\) | \(\mathbb{R}^{12}\) | 腿部关节角速度 \(\mathbf{v}_{k}^{q} \cdot s_{dq}\)（`dof_vel_scale`） |
| \(\mathbf{a}_{k-1}\) | \(\mathbb{R}^{12}\) | 上一控制周期的策略输出 |
| \(\phi_{k}\) | 标量 | 步态相位 \(2\pi (t_{k} \bmod T_{p}) / T_{p}\)，\(T_{p} = 0.8\) s |

重力投影 \(\mathbf{g}_{k}\) 由单位四元数 \(\mathbf{q} = \left[q_{w}, q_{x}, q_{y}, q_{z}\right]^\top\) 解析求得：

```math
\mathbf{g}_k = \begin{bmatrix}
2(-q_z q_x + q_w q_y) \\
-2(q_z q_y + q_w q_x) \\
1 - 2(q_w^2 + q_z^2)
\end{bmatrix}
```

#### 1.2 动作与 PD 力矩控制

策略以降采样率 \(f_{\mathrm{ctrl}} = 1 / (N_{d} \Delta t)\)（\(N_{d} =\) `control_decimation`）输出动作 \(\mathbf{a}_{k} \in \mathbb{R}^{12}\)，映射为目标关节角：

```math
\mathbf{q}_k^{\mathrm{des}} = \mathbf{q}_0 + \sigma_a \mathbf{a}_k
```

其中 \(\mathbf{q}_{0}\) 为默认站姿，\(\sigma_{a}\) 为 `action_scale`。各关节 PD 力矩为：

```math
\mathbf{\tau}_k = \mathbf{K}_p (\mathbf{q}_k^{\mathrm{des}} - \mathbf{q}_k) + \mathbf{K}_d (\mathbf{v}_k^{q,\mathrm{des}} - \mathbf{v}_k^{q})
```

力矩经 `qfrc_applied` 施加于腿部自由度；position 执行器同步写入当前角以避免与高增益 PD 冲突。

#### 1.3 路点导航与速度指令生成

给定骨盆平面目标 \(\mathbf{x}^{\mathrm{tgt}} = \left[x^{\mathrm{tgt}}, y^{\mathrm{tgt}}\right]^\top\)，记当前位置 \(\mathbf{x}_{k} = \left[x_{k}, y_{k}\right]^\top\)，相对位移 \(\mathbf{d}_{k} = \mathbf{x}^{\mathrm{tgt}} - \mathbf{x}_{k}\)，距离 \(r_{k} = \|\mathbf{d}_{k}\|\)。

到达判定（阈值 \(\epsilon_{\mathrm{arr}}\)）：

```math
r_k \le \epsilon_{\mathrm{arr}} \Rightarrow \mathbf{c}_k = \mathbf{0}
```

否则，设当前偏航角 \(\psi_{k} = 2\arctan(q_{z}, q_{w})\)，期望偏航角 \(\psi_{k}^{\mathrm{des}} = \mathrm{atan2}(d_{y}, d_{x})\)，偏航误差 \(\Delta\psi_{k}\) 为 \(\psi_{k}^{\mathrm{des}} - \psi_{k}\) 归一化到 \(\left[-\pi, \pi\right]\)：

```math
\Delta\psi_k = \mathrm{atan2}(\sin(\psi_k^{\mathrm{des}} - \psi_k),\, \cos(\psi_k^{\mathrm{des}} - \psi_k))
```

分段速度指令：当 \(|\Delta\psi_{k}| > \psi_{\mathrm{turn}}\) 时（先转向）

```math
\mathbf{c}_k = \left[v_x^{\mathrm{turn}},\, 0,\, \min(\omega_{\max}, \max(-\omega_{\max}, \alpha_\omega \Delta\psi_k))\right]^\top
```

否则（边走边调向）

```math
\mathbf{c}_k = \left[\min(v_x^{\max}, \max(v_x^{\min}, \beta r_k)),\, v_y^{\mathrm{lat}},\, \min(\omega_{\max}, \max(-\omega_{\max}, \alpha_\psi \Delta\psi_k))\right]^\top
```

其中横向分量 \(v_{y}^{\mathrm{lat}}\) 为 \(\mathbf{d}_{k}\) 在机体左向 \(\mathbf{e}_{\mathrm{left}} = \left[-\sin\psi_{k}, \cos\psi_{k}\right]^\top\) 上的投影。持箱搬运时 \(v_{x}^{\max}\) 被限制为 `carry_cmd_max_forward`（默认 0.25），以降低偏载失稳风险。

---

### 2. 速度辅助（Velocity Assist）

为克服 RL 策略在长距离导航中的收敛延迟，对骨盆施加平面外力辅助：

```math
\mathbf{F}_k^{\mathrm{assist}} = m \left[\mathbf{K}_v (\mathbf{v}_k^{\mathrm{des}} - \mathbf{v}_k) - \mathbf{K}_d^{\mathrm{assist}} \mathbf{v}_k \right]
```

其中 \(m\) 为骨盆质量。期望速度与方向单位向量：

```math
\mathbf{v}_k^{\mathrm{des}} = \frac{\mathbf{d}_k}{\max(r_k, 10^{-6})} \cdot \min\left(v_s, \max(0.12, 0.8 r_k)\right)
```

该力写入 `xfrc_applied`，与 RL 摆腿力矩协同，**不采用 kinematic 平移**，保留物理真实性。

---

### 3. 抓取判定与箱体运动学跟踪

#### 3.1 抓取条件

设右手腕（或 palm 参考点）位置 \(\mathbf{h}_{k}\)，候选箱体 \(i\) 位置 \(\mathbf{p}_{i}\)。当

```math
\|\mathbf{h}_k - \mathbf{p}_i\|_2 \le d_{\mathrm{grasp}}
```

且当前无持箱对象时，触发抓取（阈值 \(d_{\mathrm{grasp}} = 0.48\) m）。抓取后对该箱体启用 `body_gravcomp`，并通过每步运动学写入消除 weld 约束，避免物理耦合导致机器人倾倒。

#### 3.2 跟踪模式

**腕部跟踪（`wrist`）：** 抓取过渡阶段，箱心位置为

```math
\mathbf{p}_i^{\mathrm{box}} = \mathbf{x}_{\mathrm{wrist}} + \mathbf{R}_{\mathrm{wrist}} (\mathbf{o}_{\mathrm{palm}} + \mathbf{o}_{\mathrm{box}})
```

**摇篮跟踪（`cradle`）：** 搬运阶段，箱心跟随双掌中点：

```math
\mathbf{p}_i^{\mathrm{box}} = \frac{1}{2}(\mathbf{p}_{\mathrm{palm}}^{\mathrm{R}} + \mathbf{p}_{\mathrm{palm}}^{\mathrm{L}}) + \mathbf{R}_{\mathrm{pelvis}} \mathbf{o}_{\mathrm{carry}}
```

其中 \(\mathbf{o}_{\mathrm{carry}} = \left[0.02, 0, 0.04\right]^\top\) m（`carry_palm_offset`）。箱体姿态取骨盆偏航角 \(\psi_{k}\) 对应的绕 \(z\) 轴旋转四元数。每步将箱体线速度置零（`qvel` 写入 \(\mathbf{0}\)）以实现运动学锁定。

---

### 4. 上肢姿态规划

操作阶段（够取 / 抱箱 / 放货）采用**关节空间线性插值**。给定起始姿态 \(\mathbf{q}^{\mathrm{from}}\) 与目标姿态 \(\mathbf{q}^{\mathrm{to}}\)，在第 \(t\) 步（共 \(T\) 步）：

```math
\mathbf{q}_t = (1-\alpha_t)\mathbf{q}^{\mathrm{from}} + \alpha_t \mathbf{q}^{\mathrm{to}}, \qquad \alpha_t = \min\left(1, \frac{t}{T}\right)
```

关键姿态常量（单位：rad）：

| 阶段 | 腰 | 右臂 / 左臂 |
|------|----|-------------|
| 够取 REACH | \(\mathbf{0}\) | \(\left[0.10, -0.45, 0.10, 0.30, 0, -0.70, 0\right]^\top\)（左臂 \(y\) 取反） |
| 抱箱 CARRY | \(\mathbf{0}\) | \(\left[0.15, -0.15, 0.10, 0.15, 0, -0.65, 0\right]^\top\)（左臂 \(y\) 取反） |
| 放货 PLACE | \(\mathbf{0}\) | \(\left[0.12, -0.40, 0.05, 0.35, 0, -0.55, 0\right]^\top\)（左臂 \(y\) 取反） |

持箱期间，腰/臂/手关节经 `_kinematic_upper` 直接写入 `qpos`，使上肢不受 RL 摆腿扰动。

---

### 5. 操作阶段骨盆锁定

够取、放货等静态操作阶段启用平面锁定。记锁定位置 \(\bar{x}, \bar{y}, \bar{z}\) 与锁定偏航 \(\bar{\psi}\)。每步后处理：

```math
(x_{k}, y_{k}) \leftarrow (\bar{x}, \bar{y}), \quad (v_{x}, v_{y}) \leftarrow 0
```

```math
z_{k} \leftarrow \bar{z}, \quad v_{z} \leftarrow 0
```

（\(z\) 方向锁定可选。）

```math
\mathbf{q}_{k}^{\mathrm{ori}} \leftarrow [\cos(\bar{\psi}/2), 0, 0, \sin(\bar{\psi}/2)]^\top, \quad \mathbf{\omega}_{k} \leftarrow \mathbf{0}
```

腿部执行器写入固定角 \(\mathbf{q}_{\mathrm{leg}}^{\mathrm{fix}}\)（`DEFAULT_LEG`），RL 策略停用。

---

### 6. 放置完成判定

箱体 \(i\) 在时刻 \(k\) 被标记为已放置，当且仅当同时满足：

**空间约束**

```math
|p_{i,x} - c_x| \le h_x,\quad |p_{i,y} - c_y| \le h_y,\quad |p_{i,z} - c_z| \le h_z + \delta_z
```

**速度约束**

```math
\|\mathbf{v}_i\|_2 < v_{\mathrm{th}}
```

**高度约束**

```math
p_{i,z} \ge h_{\min}
```

其中 \((c_{x}, c_{y}, c_{z})\) 为放置区中心，\((h_{x}, h_{y}, h_{z})\) 为放置区半尺寸；\(v_{\mathrm{th}} = 0.15\,\mathrm{m/s}\)，\(h_{\min} = 0.80\,\mathrm{m}\)，\(\delta_{z} = 0.05\,\mathrm{m}\)。持箱中的箱体（\(i = i_{\mathrm{grasp}}\)）不参与判定。

---

### 7. 持箱高度补偿

搬运阶段对骨盆施加竖直方向虚拟弹簧—阻尼力，抑制偏载下沉：

```math
F_z = K_z (z_{\mathrm{ref}} - z_{\mathrm{pelvis}}) - C_z v_{z,\mathrm{pelvis}}
```

默认 \(z_{\mathrm{ref}} = 0.79\) m，\(K_{z} = 1200\) N/m，\(C_{z} = 120\) N·s/m（`carry_elastic_band`）。

---

### 8. 摔倒检测与恢复

**摔倒判定：** 骨盆高度 \(z_{k} < z_{\mathrm{spawn}} - 0.22\) m（\(z_{\mathrm{spawn}} = 0.79\) m）。

**恢复算子 \(\mathcal{R}\)（手动模式 `U` 键）：** 保留当前平面位姿 \((x_{k}, y_{k}, \psi_{k})\) 与持箱索引 \(i_{\mathrm{grasp}}\)，将机器人关节重置为 MuJoCo 关键帧 `stand`，并令

```math
z_k \leftarrow z_{\mathrm{spawn}}, \qquad \mathbf{v}_k \leftarrow \mathbf{0}
```

同时调用 `sync_carried_box()`。恢复后可继续 RL 行走，持箱状态不丢失。

---

### 9. 自动演示状态机

`DemoController` 实现有限状态机 \(\mathcal{M} = (\mathcal{Q}, \Sigma, \delta, q_{0})\)，状态包括：

`WALK_TO_PICK` · `REACH` · `GRASP` · `STABILIZE` · `WALK_ROUTE` · `PLACE` · `RELEASE` · `RETRACT` · `WALK_BACK`

搬运路径 \(\Gamma = \left[\mathbf{\xi}_{1}, \ldots, \mathbf{\xi}_{m}\right]\) 由配置路点与动态生成的后退/入通道点拼接而成，逐点导航直至 \(\|\mathbf{x}_{k} - \mathbf{\xi}_{j}\|_{2} \le \epsilon_{\mathrm{route}}\) 后切换至 \(\mathbf{\xi}_{j+1}\)。

---

## 人机交互接口

### 手动模式

| 按键 | 功能 |
|------|------|
| `W` / `A` / `S` / `D` 或方向键 | 设定平面路点 \(\mathbf{x}^{\mathrm{tgt}}\) |
| `空格` | 清除路点，停止行走 |
| `G` | 启动够取—抓取—抱箱过渡序列 |
| `R` | 释放当前持箱 |
| `U` | 执行恢复算子 \(\mathcal{R}\) |
| `Backspace` | 全场景重置 |
| `Esc` | 退出仿真 |

按键事件经队列在主循环线程处理，避免 MuJoCo `mjData` 与 viewer 回调线程并发访问。

### 演示模式

| 按键 | 功能 |
|------|------|
| `R` | 强制释放 |
| `Backspace` | 重置场景及状态机 |
| `Esc` | 退出 |

---

## 目录结构

```
g1-mujoco-box-challenge/
├── config/
│   ├── task.yaml                 # 任务 / 行走 / 抓取参数
│   └── locomotion_g1.yaml        # RL 策略与 PD 增益
├── assets/robots/g1/             # G1 模型与搬箱场景
├── third_party/g1_policy/        # 预训练行走策略
├── src/
│   ├── core/                     # 配置加载
│   ├── env/                      # 仿真环境、任务判定
│   ├── control/                  # RL 行走、PD 力矩
│   ├── controllers/              # 基线 / 手动 / TeamController 接口
│   ├── planning/                 # 【预留】路径规划
│   ├── manipulation/             # 【预留】抓取操作
│   └── utils/                    # 【预留】工具函数
├── teams/                        # 参赛队代码（template 为提交模板）
├── scripts/
│   ├── run_sim.py                # 可视化入口（--record 可交互录制）
│   ├── evaluate.py               # 无头运行，输出任务完成情况
│   ├── record_demo.py            # 无头批量录制
│   └── tools/                    # 开发辅助脚本
├── assets/demo_2boxes.mp4        # 演示录像（README 内嵌需在 GitHub 网页拖入上传）
├── setup.sh
├── README.md
└── 赛题说明.md
```

参赛队开发请参阅 [赛题说明.md](赛题说明.md) 第 8.4 节。

---

## 配置参数

主配置文件：`config/task.yaml`。关键参数摘要：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `spawn.pos` | \(\left[-0.78, 0, 0.79\right]\) | 机器人初始位姿 |
| `walk.table_stand` | \(\left[2.85, -0.65\right]\) | 放货站位（桌南侧） |
| `walk.path_lane_y` | \(-0.90\) | 南侧搬运通道 |
| `walk.pick_south_y` | \(-0.52\) | 第 1 行箱够取站位 |
| `walk.pick_south_y_row2` | \(-0.32\) | 第 2 行箱够取站位 |
| `grasp.distance_threshold` | \(0.48\) m | 抓取距离阈值 \(d_{\mathrm{grasp}}\) |
| `table.top_height` | \(0.73\) m | 放货桌面高度 |

搬运路径示例（`carry_route` 段）：

```yaml
walk:
  path_lane_y: -0.90
  path_west_x: -0.90
  table_stand: [2.85, -0.65]
  carry_route:
    - [3.05, -0.90]
    - [2.85, -0.65]
  return_route:
    - [-0.78, 0.0]
```

---

## 常见问题

**`setup.sh` 无法获取 G1 模型**

确认网络连通；脚本默认从 Gitee 克隆 `mujoco_menagerie`。亦可指定本地路径：

```bash
MUJOCO_MENAGERIE_SRC=/path/to/mujoco_menagerie bash setup.sh
```

**PyTorch 相关报错**

```bash
conda activate unitree_pick_place
pip install torch typing_extensions
```

**场景文件未同步**

主场景路径：`assets/robots/g1/box_pick_place.xml`。修改 `assets/scenes/box_pick_place.xml` 后需重新执行 `setup.sh` 或手动复制。

---

## 许可证

G1 机器人模型遵循 [MuJoCo Menagerie / Unitree G1 LICENSE](assets/robots/g1/LICENSE)。
