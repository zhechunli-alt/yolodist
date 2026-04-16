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

## 第二轮蒸馏策略

由于第一轮 `response MSE + merge_teacher` 在 `DeepPCB` 上未能拉开与普通 student 的差距，第二轮蒸馏改为更贴近目标检测本身的组合策略：`Localization-aware + Foreground-weighted Feature KD`。

### 1. Localization-aware KD

- 不再把整个检测头输出当作一个整体做 MSE
- 将检测头输出拆分为：
  - 边框回归分布
  - 分类响应
- 对边框回归部分使用 DFL 风格的分布蒸馏
- 对分类部分使用 sigmoid 后的响应蒸馏

这样做的原因是：

- 检测任务里最难蒸馏的往往不是“有没有目标”，而是“框得准不准”
- 第一轮蒸馏没有显式强调定位知识，容易退化成对整体输出的弱约束

### 2. Foreground-weighted Feature KD

- 对 `P3 / P4 / P5` 的 teacher / student 特征继续做蒸馏
- 但不再做全图平均
- 改为使用 GT box 生成前景 mask
- 在前景区域内更强地对齐 teacher / student 特征
- 背景区域仅保留较低权重

这样做的原因是：

- PCB 缺陷目标通常小、稀疏
- 全图 feature KD 很容易被大面积背景稀释
- 前景加权更符合工业缺陷检测的实际需求

### 3. 当前配置

- `kd_strategy = "loc_fg_distill"`
- `distill_alpha = 0.4`
- `loc_kd_alpha = 1.0`
- `cls_kd_alpha = 0.25`
- `feature_kd_alpha = 0.10`
- `bg_weight = 0.05`
- `mask_expand_ratio = 0.10`

### 4. 预期目标

- 先验证第二轮蒸馏是否能在 `DeepPCB` 上真实超过对应 student
- 若有效，再看能否在 `PKU-Market-PCB` 与 `DsPCBSD+` 上复现
- 若仍不明显，则论文主结论继续以 `EPFA` 为核心，蒸馏作为探索性工作

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

## 模型关系与训练来源说明

这部分用于明确 `baseline_plain`、`student_plain`、`student_epfa`、`distill_plain`、`distill_epfa` 的关系，避免在论文写作时把“学生模型结构”和“蒸馏训练结果”混在一起。

### 1. baseline_plain

- 指标准 `YOLO11n` 基线模型
- 使用原始训练数据和真实标签做普通监督训练
- 训练完成后得到一个性能较强的检测模型
- 这个模型既是基线参考，也是后续蒸馏阶段的 `teacher`

### 2. student_plain

- 指轻量学生模型结构，不带 `EPFA`
- 同样使用原始训练数据和真实标签做普通监督训练
- 这是“只做轻量化、不加蒸馏、不加创新模块”的对照组

### 3. student_epfa

- 指加入 `EPFA-Lite` 的轻量学生模型结构
- 同样使用原始训练数据和真实标签做普通监督训练
- 这是“轻量化 + PCB 领域导向模块”的改进组

### 4. distill_plain

- 学生结构仍然是 `student_plain`
- 但训练时不再只依赖真实标签
- 而是在原始训练数据上，同时使用：
  - 真实标签监督
  - `baseline_plain` 提供的 teacher 指导
- 因此 `distill_plain` 不是“在 `student_plain` checkpoint 上继续微调”的概念，而是“使用同一学生结构，重新进行一条蒸馏训练线后得到的模型”

### 5. distill_epfa

- 学生结构仍然是 `student_epfa`
- 训练时同样在原始训练数据上，同时使用：
  - 真实标签监督
  - `baseline_plain` 提供的 teacher 指导
- 因此 `distill_epfa` 也是一条独立蒸馏训练线的最终模型，而不是简单从 `student_epfa` 继续训练得到

### 一句话总结

- `student_plain / student_epfa`：普通监督训练得到的学生模型
- `distill_plain / distill_epfa`：在相同原始数据上，引入 teacher 指导后重新训练得到的学生模型

### 关系示意

```text
原始数据 + 真实标签 ------------------------> student_plain
原始数据 + 真实标签 ------------------------> student_epfa

baseline_plain 作为 teacher

原始数据 + 真实标签 + teacher指导 ----------> distill_plain
原始数据 + 真实标签 + teacher指导 ----------> distill_epfa
```

写作时建议区分两层概念：

- `student_plain / student_epfa` 表示学生模型结构或普通训练结果
- `distill_plain / distill_epfa` 表示对应学生结构经过蒸馏训练后的最终模型

## 失败案例与风险点

- `PKU-Market-PCB` 的公开下载入口依赖外部分发站点，自动化下载稳定性较弱。
- `DsPCBSD+` 虽然可公开下载，但原始目录结构可能与 `DeepPCB` 不同，因此准备脚本做成了自动探测式而不是硬编码。
- `feature KD` 只有在 teacher / student 对应层 shape 一致时才参与，否则会自动跳过，避免训练直接报错。
- `EPFA-Lite` 依赖输入图像上下文，因此在 runner 里增加了前向 pre-hook 来传递当前 batch 图像；这部分实现尽量局部，避免破坏现有训练入口。
