# PCB 迁移与 EPFA 说明

## 迁移动机

- MVTec 转检测任务后，`mAP50-95` 长期偏低，说明“分割任务硬转检测”不是当前阶段最稳的主线。
- PCB 缺陷数据集天然是检测任务，更适合 YOLO11 和论文里的 `mAP / Recall / FPS` 叙事。
- 因此本轮迁移目标不是“换一个看起来更强的数据集”，而是优先获得稳定、可复现、可对比的实验闭环。

## 本轮策略

1. 先接入 `DeepPCB / PKU-Market-PCB / DsPCBSD+` 三个数据集。
2. 先跑稳 `student_plain baseline`。
3. 再落地一个轻量、PCB 领域导向的模块：`EPFA-Lite`。
4. 蒸馏只做保守扩展，避免再次出现“工程跑通但结果更差”的情况。

## EPFA-Lite 结构

输入特征记为 `F(B, C, H, W)`，模块只插在 `P3 / P4 / P5` 检测尺度前。

### 1. 通道门控

- 对 `F` 做 `GAP`
- 经过 `1D Conv`
- 再经过 `sigmoid`
- 得到 `Gc(B, C, 1, 1)`

这是一个 ECA 风格的轻量通道重标定，用来保留对细小缺陷更敏感的通道。

### 2. 边缘先验分支

- 从原始输入图像生成灰度图
- 使用固定 Sobel 卷积计算边缘幅值图 `E`
- 将 `E` 下采样到当前尺度 `H, W`
- 经过 `3x3 Conv + sigmoid`
- 得到 `Gs(B, 1, H, W)`

这个分支没有引入重型可学习 backbone，只把 PCB 缺陷中常见的“细边缘、断裂、缺口、毛刺”显式提示给检测特征。

### 3. 融合

- `F' = F * Gc * (1 + alpha * Gs)`
- 其中 `alpha` 是可学习标量，初值 `0.5`

这样设计的意图是：

- `Gc` 负责“哪些通道更重要”
- `Gs` 负责“哪些空间位置更像缺陷边缘”
- `alpha` 控制边缘提示不要一开始就压过原特征

## 损失函数

蒸馏阶段默认使用 response KD：

- `L_total = (1 - alpha) * L_det + alpha * L_kd`
- `L_kd` 当前使用 teacher / student 多尺度检测输出的 MSE
- `T` 由 `distill_temperature` 控制

额外预留了可选 `feature KD`：

- 通过对 `P3 / P4 / P5` 指定层挂 hook
- 对 shape 一致的特征做 MSE
- 默认关闭，避免在第一轮 PCB 实验里引入额外不稳定因素

## 实验设置

- Baseline：`student_plain`
- 改进模型：`student_epfa`
- 蒸馏：
  - `Distill(plain)`
  - `Distill(EPFA)`

统一对比指标：

- Params
- FLOPs
- FPS
- mAP50
- mAP50-95
- Recall

## 失败案例与风险点

- `PKU-Market-PCB` 的公开下载入口依赖外部分发站点，自动化下载稳定性较弱。
- `DsPCBSD+` 虽然可公开下载，但原始目录结构可能与 `DeepPCB` 不同，因此准备脚本做成了自动探测式而不是硬编码。
- `feature KD` 只有在 teacher / student 对应层 shape 一致时才参与，否则会自动跳过，避免训练直接报错。
- `EPFA-Lite` 依赖输入图像上下文，因此在 runner 里增加了前向 pre-hook 来传递当前 batch 图像；这部分实现尽量局部，避免破坏现有训练入口。
