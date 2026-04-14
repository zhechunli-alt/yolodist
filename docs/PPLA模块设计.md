# PPLA 模块设计文档

## 1. 模块目标

模块名称：`Prompt-Prior Lite Attention (PPLA)`  
定位：面向工业缺陷检测的轻量注意力增强模块。  
核心目标：在尽量不增加参数和推理开销的前提下，提升学生模型对小目标与长尾缺陷的召回能力。

## 2. 可行性评估（结论）

结论：**可行，且与当前项目路线匹配**。

可行性理由：

1. 结构可轻量实现：仅使用 `GAP/Conv1x1/1DConv/Sigmoid/Mul`，不引入 Transformer。
2. 可插入 YOLO 检测链路：放在 Neck 输出到 Detect 头之前，作用于 `P3/P4/P5`。
3. 与论文背景一致：解决小目标漏检、长尾类别弱、边缘端算力受限这三个核心问题。
4. 与当前工程兼容：你已经有 `baseline / modified / distill` 三条实验线，可直接做对比和消融。

前提条件：

1. 需要在 Ultralytics 侧注册自定义模块，才能在 YAML 中引用。
2. 先以检测性能提升为主，再逐步叠加蒸馏一致性损失。

## 3. 为什么不是“只用 mask 转框就够了”

你的判断是正确的：MVTec 的 mask 转框是强监督主标签，优先级最高。  
PPLA 不是替代真实标签，而是增强学生模型的“关注能力”：

1. 强监督标签告诉模型“哪里是缺陷”。
2. PPLA 让模型“更关注哪些通道和尺度信息”。
3. 两者作用不同：一个是监督信号，一个是特征重标定机制。

因此，论文叙述建议写为：

- `mask->box` 提供基础监督；
- `PPLA` 提供轻量注意力增强；
- 蒸馏提供教师知识迁移。

## 4. 模块定义（轻量版本）

输入：`P3, P4, P5` 三层特征 `F_l`。  
先验：prompt 先验向量 `p`（可由缺陷类型/纹理/严重程度编码而来）。

对每层特征做：

1. 通道描述：`z_l = GAP(F_l)`
2. 轻量通道建模：`a_l = ECA(z_l)`（1D 卷积实现局部通道交互）
3. 先验映射：`q_l = Conv1x1(z_l)`，与 `p` 做相似度得到 `s_l`
4. 门控融合：`g_l = sigmoid(a_l + lambda * s_l)`
5. 输出特征：`F'_l = F_l * g_l`

复杂度目标：

1. 参数增量：`0.1M ~ 0.3M`
2. FLOPs 增量：`< 3%`
3. FPS 降低：`<= 1~2`

## 5. 接入 YOLO 的推荐位置

推荐插入点：Neck 输出到 Detect 头之前。  
推荐作用层：`P3/P4/P5`。

原因：

1. `P3` 对小目标最敏感，优先受益。
2. `P4/P5` 保留中大尺度语义。
3. 在 Detect 前做门控，对检测头侵入最小。

## 6. 与当前三条模型线的关系

当前项目三线定义如下：

1. `baseline`：标准 `yolo11n.pt` 训练。
2. `modified_model`：轻量学生模型 YAML（当前是学生骨架，不含 PPLA）。
3. `distill`：在学生模型基础上做教师伪标签蒸馏。

引入 PPLA 后建议升级为：

1. `baseline`：保持不变（对照组）。
2. `modified_model_ppla`：学生模型 + PPLA。
3. `distill_ppla`：学生模型 + PPLA + 蒸馏。

## 7. 实施步骤（按工程落地顺序）

### 步骤 A：先做“无先验”PPLA（只通道门控）

目标：验证轻量注意力本身是否带来收益。  
改动：

1. 在 Ultralytics 注册 `PPLALite`（先不接 prompt 先验分支）。
2. 在学生 YAML 中将 P3/P4/P5 Detect 前插入 `PPLALite`。
3. 跑 `modified_model` 对比 baseline。

### 步骤 B：接入 prompt 先验向量

目标：验证“提示词先验引导”是否额外增益。  
改动：

1. 设计先验向量生成方式（离线 JSON / 类别嵌入表）。
2. 在 dataloader 或 model forward 中注入先验向量。
3. 在 `PPLALite` 中加入先验相似度门控分支。

### 步骤 C：与蒸馏联合训练

目标：验证 PPLA 与蒸馏是否互补。  
改动：

1. 先跑现有伪标签蒸馏流程。
2. 再扩展蒸馏损失（可选）：
   - 响应蒸馏（logits）
   - 特征蒸馏（P3/P4/P5）
3. 对比 `distill` vs `distill + PPLA`。

## 8. 代码接入路径（与你当前仓库对应）

### 方案 1：快速实验版（推荐先做）

1. 在 `ultralytics-src` 放入可编辑的 Ultralytics 源码。
2. 新增模块文件，例如：
   - `ultralytics/nn/modules/ppla.py`
3. 在模块导出位置注册 `PPLALite`。
4. 在学生 YAML（`configs/models/yolo11_student.yaml`）插入模块层。
5. 使用 `modified_model` 配置跑训练。

### 方案 2：纯外部封装版（不改源码，难度更高）

1. 在项目侧二次封装模型 forward。
2. 在 neck 输出后手工调用 PPLA。
3. 保持与 Ultralytics 训练器对接。

说明：方案 2 对训练器侵入更大，维护成本更高，不建议作为第一选择。

## 9. 训练损失建议

基础损失：

1. `L_det`：YOLO 原生检测损失（必须）。

蒸馏阶段可选：

1. `L_kd_resp`：响应蒸馏损失（温度 `T`）。
2. `L_kd_feat`：特征蒸馏损失（P3/P4/P5）。
3. `L_prior`：先验一致性损失（小权重）。

总损失示例：

`L = L_det + alpha * L_kd_resp + beta * L_kd_feat + gamma * L_prior`

## 10. 消融实验设计（建议最少 6 组）

1. baseline（yolo11n）
2. student（无 PPLA、无蒸馏）
3. student + PPLA（无先验）
4. student + PPLA（有先验）
5. student + distill（无 PPLA）
6. student + PPLA + distill

报告指标建议：

1. `mAP50`
2. `mAP50-95`
3. 小目标 Recall（可单独统计）
4. Params
5. FLOPs
6. FPS（统一硬件）

## 11. 风险与兜底

风险：

1. prompt 先验质量不稳定导致收益波动。
2. 模块引入后可能对某些类别过拟合。
3. ONNX/TensorRT 导出时若使用不常见算子可能失败。

兜底：

1. 先上“无先验 PPLA”，确认基础收益。
2. 先验分支默认可开关，失败时可退回纯 ECA-lite。
3. 导出阶段坚持基础算子组合（Conv/GAP/Sigmoid/Mul）。

## 12. 对论文可直接使用的创新描述

可写法：

“本文提出一种提示词先验引导的轻量缺陷注意力模块（PPLA），在保持低额外计算开销的前提下，对多尺度特征进行自适应重标定，从而提升了学生模型对小目标和长尾缺陷的检测性能。”

