# PCB 实验记录

这份文档用于沉淀 PCB 主线实验的真实执行记录，目标是后续可以直接作为毕业论文实验章节和方法章节的事实依据。

记录原则：

1. 只写已经发生的事实
2. 每条记录尽量包含命令、配置、结果、结论
3. 区分“计划中 / 进行中 / 已完成 / 已失败”

日志与实验目录命名规则：

1. 数据集名写在 `runs/<dataset>_.../` 目录前缀中，例如 `runs/deeppcb_*`、`runs/pku_market_pcb_*`
2. 模型路线写在目录后缀中，例如 `baseline_plain`、`student_plain`、`student_epfa`
3. 蒸馏超参数如有变化，直接写入目录名，例如 `distill_plain_alpha05`
4. `results.csv` 表示训练过程日志，`*_eval2/metrics_summary.json` 表示测试集正式评估结果
5. 后续写论文时，优先引用带 `eval` 目录下的测试集结果，避免把训练末尾验证指标和测试指标混用

## 一、实验总目标

- 数据集：`DeepPCB`、`PKU-Market-PCB`、`DsPCBSD+`
- 模型路线：
  - `baseline_plain`
  - `student_plain`
  - `student_epfa`
  - `distill_plain`
  - `distill_epfa`
- 统一指标：
  - `Params`
  - `FLOPs`
  - `FPS`
  - `mAP50`
  - `mAP50-95`
  - `Recall`

## 二、当前执行顺序

### P0：先获得稳定 teacher

1. `DsPCBSD+ baseline_plain`
2. `DeepPCB baseline_plain`
3. `PKU-Market-PCB baseline_plain`

### P1：轻量模型对比

4. `DeepPCB student_plain`
5. `DeepPCB student_epfa`
6. `PKU-Market-PCB student_plain`
7. `PKU-Market-PCB student_epfa`
8. `DsPCBSD+ student_plain`
9. `DsPCBSD+ student_epfa`

### P2：蒸馏对比

10. `DeepPCB distill_plain`
11. `DeepPCB distill_epfa`
12. `PKU-Market-PCB distill_plain`
13. `PKU-Market-PCB distill_epfa`
14. `DsPCBSD+ distill_plain`
15. `DsPCBSD+ distill_epfa`

## 三、数据接入状态

### 2026-04-15 DeepPCB

- 状态：`已完成 prepare / 已通过 preflight / 可开始训练`
- 原始目录：`datasets/deeppcb/`
- 处理后目录：`datasets/processed/deeppcb_detection/`
- 数据统计：
  - `train = 900`
  - `val = 100`
  - `test = 500`
- 类别：
  - `open`
  - `short`
  - `mousebite`
  - `spur`
  - `spurious_copper`
  - `missing_hole`
- 说明：
  - DeepPCB 原始标注中同时存在逗号分隔和空格分隔两种写法，准备脚本已兼容。
  - 官方 split 文件中的图片名与实际文件名存在 `_test.jpg` 后缀差异，已在准备逻辑中兼容。

### 2026-04-15 PKU-Market-PCB

- 状态：`已完成 prepare / 已通过 preflight / 可开始训练`
- 原始目录：`datasets/pku_market_pcb/`
- 处理后目录：`datasets/processed/pku_market_pcb_detection/`
- 数据统计：
  - `train = 555`
  - `val = 46`
  - `test = 92`
- 类别：
  - `open`
  - `short`
  - `mousebite`
  - `spur`
  - `spurious_copper`
  - `missing_hole`
- 说明：
  - 原始包为 `tar.gz`
  - 标注为 `COCO json`，位于 `pcb_cocoanno/train.json`、`val.json`
  - 准备脚本已兼容这种 COCO 目录布局

### 2026-04-15 DsPCBSD+

- 状态：`已完成 prepare / 已通过 preflight / baseline 已启动训练`
- 原始目录：`datasets/dspcbsd_plus/`
- 处理后目录：`datasets/processed/dspcbsd_plus_detection/`
- 数据统计：
  - `train = 8208`
  - `val = 683`
  - `test = 1368`
- 类别：
  - `short`
  - `spur`
  - `spurious_copper`
  - `open`
  - `mousebite`
  - `hole_breakout`
  - `conductor_scratch`
  - `conductor_foreign_object`
  - `base_material_foreign_object`
- 说明：
  - 原始包为公开 Figshare ZIP
  - 解压后为 `Data_COCO` 结构
  - 原始类别名为缩写：`SH / SP / SC / OP / MB / HB / CS / CFO / BMFO`
  - 已在准备脚本中完成 alias 映射

## 四、模型与模块状态

### EPFA-Lite

- 状态：`已实现 / 已注册 / 已完成前向验证`
- 文件：
  - [src/yolodist/models/epfa.py](/root/workspace/yolodist/src/yolodist/models/epfa.py)
  - [configs/models/yolo11_student_epfa.yaml](/root/workspace/yolodist/configs/models/yolo11_student_epfa.yaml)
- 当前验证结果：
  - `student_plain params = 1,286,179`
  - `student_epfa params = 1,286,221`
  - 参数增量仅 `42`
- 当前判断：
  - 参数增量远低于 `0.3M` 限制
  - 满足轻量化约束，可进入正式消融

## 五、实验执行记录

### 2026-04-15 DsPCBSD+ baseline_plain

- 状态：`中途暂停，可续跑`
- 配置：
  - [configs/train/baseline_dspcbsd_plus.toml](/root/workspace/yolodist/configs/train/baseline_dspcbsd_plus.toml)
- 命令：

```bash
source /root/workspace/.venv/bin/activate
PYTHONPATH=src python experiments/baseline/train.py --config configs/train/baseline_dspcbsd_plus.toml
```

- 当前观察：
  - 训练已正常推进到 `epoch 26`
  - `results.csv` 已保存
  - `best.pt` 与 `last.pt` 已生成
  - 最近一条记录对应验证指标约为：
    - `precision = 0.7599`
    - `recall = 0.7416`
    - `mAP50 = 0.7937`
    - `mAP50-95 = 0.4460`
- 暂停原因：
  - 为了更快拿到论文主实验结果，优先把 `DeepPCB baseline` 切上 GPU
  - `DsPCBSD+` 作为大数据集保留为后续泛化验证，可基于当前权重继续训练

### 2026-04-15 DeepPCB baseline_plain

- 状态：`已完成训练 / 已完成 test 评估 / 已生成 teacher`
- 配置：
  - [configs/train/baseline_deeppcb.toml](/root/workspace/yolodist/configs/train/baseline_deeppcb.toml)
- 命令：

```bash
source /root/workspace/.venv/bin/activate
PYTHONPATH=src python experiments/baseline/train.py --config configs/train/baseline_deeppcb.toml
```

- 启动观察：
  - 模型正常加载 `yolo11n.pt`
  - 自动将 `nc` 覆盖为 `6`
  - 数据扫描正常：
    - `train = 900`
    - `val = 100`
    - `corrupt = 0`
  - `AMP checks passed`
  - 已创建 `train.cache`
- 当前判断：
  - DeepPCB baseline 已具备完整训练条件
  - 可直接作为 teacher 基线使用
- 早期训练信号：
  - `epoch 4` 验证约为：
    - `precision = 0.7550`
    - `recall = 0.7471`
    - `mAP50 = 0.8021`
    - `mAP50-95 = 0.4340`
  - 说明：
    - 基线训练在早期就能稳定收敛
    - 该数据集比 MVTec 转检测更适合当前 YOLO11 路线
  - `epoch 8` 验证约为：
    - `precision = 0.8741`
    - `recall = 0.8499`
    - `mAP50 = 0.9310`
    - `mAP50-95 = 0.6496`
  - 进一步说明：
    - DeepPCB 基线在 10 个 epoch 内已显示出较强的可分性
    - 后续 `student_plain` 和 `student_epfa` 有较明确的对比价值
- 训练完成情况：
  - 训练目录：`runs/deeppcb_baseline/baseline_plain/`
  - 教师权重：`weights/teacher_deeppcb.pt`
- test 评估：
  - 评估目录：`runs/deeppcb_baseline/baseline_plain_eval2/`
  - 指标：
    - `precision = 0.9615`
    - `recall = 0.9484`
    - `mAP50 = 0.9755`
    - `mAP50-95 = 0.7492`
- 记录结论：
  - `baseline_plain` 是当前 DeepPCB 最强参考线
  - 该结果来自测试集，不是仅验证集结果

### 2026-04-15 DeepPCB student_plain

- 状态：`已完成训练 / 已完成 test 评估`
- 配置：
  - [configs/train/modified_model_deeppcb_plain.toml](/root/workspace/yolodist/configs/train/modified_model_deeppcb_plain.toml)
  - [configs/eval/modified_model_deeppcb_plain.toml](/root/workspace/yolodist/configs/eval/modified_model_deeppcb_plain.toml)
- 命令：

```bash
source /root/workspace/.venv/bin/activate
PYTHONPATH=src python experiments/modified_model/train.py --config configs/train/modified_model_deeppcb_plain.toml
PYTHONPATH=src python experiments/modified_model/evaluate.py --config configs/eval/modified_model_deeppcb_plain.toml
```

- 训练目录：`runs/deeppcb_modified/student_plain/`
- test 评估目录：`runs/deeppcb_modified/student_plain_eval2/`
- test 指标：
  - `precision = 0.9449`
  - `recall = 0.8758`
  - `mAP50 = 0.9405`
  - `mAP50-95 = 0.6247`
- 记录结论：
  - 轻量化学生模型能在 DeepPCB 上稳定工作
  - 但相较 `baseline_plain` 存在明显性能损失
  - 该组可作为 EPFA 与蒸馏的直接对照组

### 2026-04-15 DeepPCB student_epfa

- 状态：`已完成训练 / 已完成 test 评估`
- 配置：
  - [configs/train/modified_model_deeppcb_epfa.toml](/root/workspace/yolodist/configs/train/modified_model_deeppcb_epfa.toml)
  - [configs/eval/modified_model_deeppcb_epfa.toml](/root/workspace/yolodist/configs/eval/modified_model_deeppcb_epfa.toml)
- 命令：

```bash
source /root/workspace/.venv/bin/activate
PYTHONPATH=src python experiments/modified_model/train.py --config configs/train/modified_model_deeppcb_epfa.toml
PYTHONPATH=src python experiments/modified_model/evaluate.py --config configs/eval/modified_model_deeppcb_epfa.toml
```

- 训练目录：`runs/deeppcb_epfa/student_epfa/`
- test 评估目录：`runs/deeppcb_epfa/student_epfa_eval2/`
- test 指标：
  - `precision = 0.9500`
  - `recall = 0.9134`
  - `mAP50 = 0.9619`
  - `mAP50-95 = 0.6678`
- 与 `student_plain(test)` 对比增量：
  - `precision +0.0051`
  - `recall +0.0376`
  - `mAP50 +0.0214`
  - `mAP50-95 +0.0431`
- 记录结论：
  - `EPFA-Lite` 在 DeepPCB 测试集上带来真实提升
  - 提升最明显的是 `Recall` 和 `mAP50-95`
  - 当前已满足“至少一个数据集有可见提升”的验收目标

### 2026-04-15 DeepPCB distill_plain

- 状态：`已完成训练 / 已完成 test 评估`
- 配置：
  - [configs/train/distillation_deeppcb_plain.toml](/root/workspace/yolodist/configs/train/distillation_deeppcb_plain.toml)
  - [configs/eval/distillation_deeppcb_plain.toml](/root/workspace/yolodist/configs/eval/distillation_deeppcb_plain.toml)
- 命令：

```bash
source /root/workspace/.venv/bin/activate
PYTHONPATH=src python experiments/distillation/train.py --config configs/train/distillation_deeppcb_plain.toml
```

- 训练目录：`runs/deeppcb_distill/distill_plain/`
- 训练结束后 best checkpoint 验证指标：
  - `precision = 0.901`
  - `recall = 0.903`
  - `mAP50 = 0.957`
  - `mAP50-95 = 0.646`
- test 评估目录：`runs/deeppcb_distill/distill_plain_eval2/`
- test 指标：
  - `precision = 0.9111`
  - `recall = 0.8673`
  - `mAP50 = 0.9290`
  - `mAP50-95 = 0.5920`
- 与 `student_plain(test)` 对比增量：
  - `precision -0.0338`
  - `recall -0.0085`
  - `mAP50 -0.0115`
  - `mAP50-95 -0.0327`
- 记录结论：
  - 这版 `distill_plain` 在 DeepPCB 测试集上未优于 `student_plain`
  - 说明当前 `plain student + KD` 组合并不稳定
  - 后续若要保留蒸馏主线，应优先观察 `distill_epfa` 是否更适合承接 teacher 知识
- 备注：
  - 本轮为 `80 epoch` 初筛结果
  - 为保证与其他组公平，后续将统一补跑 `100 epoch` 版本

### 2026-04-15 DeepPCB distill_plain（100 epoch 统一口径重跑）

- 状态：`已完成训练 / 已完成 test 评估`
- 配置：
  - [configs/train/distillation_deeppcb_plain.toml](/root/workspace/yolodist/configs/train/distillation_deeppcb_plain.toml)
  - [configs/eval/distillation_deeppcb_plain.toml](/root/workspace/yolodist/configs/eval/distillation_deeppcb_plain.toml)
- 训练目录：`runs/deeppcb_distill/distill_plain/`
- test 评估目录：`runs/deeppcb_distill/distill_plain_eval2/`
- test 指标：
  - `precision = 0.9449`
  - `recall = 0.8758`
  - `mAP50 = 0.9405`
  - `mAP50-95 = 0.6247`
- 与 `student_plain(test)` 对比：
  - 指标完全一致
- 记录结论：
  - 在统一 `100 epoch` 口径下，`distill_plain` 未体现出蒸馏增益
  - 当前 `plain student + KD` 近似退化为普通 student 训练

### 2026-04-15 DeepPCB distill_epfa（100 epoch）

- 状态：`已完成训练 / 已完成 test 评估`
- 配置：
  - [configs/train/distillation_deeppcb_epfa.toml](/root/workspace/yolodist/configs/train/distillation_deeppcb_epfa.toml)
  - [configs/eval/distillation_deeppcb_epfa.toml](/root/workspace/yolodist/configs/eval/distillation_deeppcb_epfa.toml)
- 训练目录：`runs/deeppcb_distill_epfa/distill_epfa/`
- test 评估目录：`runs/deeppcb_distill_epfa/distill_epfa_eval2/`
- test 指标：
  - `precision = 0.9500`
  - `recall = 0.9134`
  - `mAP50 = 0.9619`
  - `mAP50-95 = 0.6678`
- 与 `student_epfa(test)` 对比：
  - 指标完全一致
- 记录结论：
  - 在统一 `100 epoch` 口径下，`distill_epfa` 未体现出额外蒸馏增益
  - 当前应优先将论文主结论聚焦为 `EPFA` 对轻量学生有效，而非蒸馏有效

### 2026-04-16 第二轮蒸馏策略调整

- 状态：`已完成代码实现 / 已通过 DeepPCB 启动冒烟验证 / 已排入连续训练`
- 调整原因：
  - 第一轮蒸馏在 `DeepPCB` 上与普通 student 几乎无差异
  - 说明 `response MSE + merge_teacher` 没有有效传递检测任务里更关键的定位知识
- 新策略：
  - `Localization-aware KD`
  - `Foreground-weighted Feature KD`
- 关键配置：
  - `kd_strategy = "loc_fg_distill"`
  - `distill_alpha = 0.4`
  - `loc_kd_alpha = 1.0`
  - `cls_kd_alpha = 0.25`
  - `feature_kd_alpha = 0.10`
  - `bg_weight = 0.05`
  - `mask_expand_ratio = 0.10`
- 新实验命名：
  - `distill_plain_locfg`
  - `distill_epfa_locfg`
- 已新增配置：
  - `DeepPCB / PKU-Market-PCB / DsPCBSD+` 各自的 `plain_locfg` 与 `epfa_locfg` train/eval 配置
- 当前判断：
  - 第二轮蒸馏是“策略升级实验”
  - 与第一轮蒸馏分目录、分配置管理，避免结果混淆

### 2026-04-16 DeepPCB distill_plain_locfg

- 状态：`已完成训练 / 已完成 test 评估`
- 配置：
  - [configs/train/distillation_deeppcb_plain_locfg.toml](/root/workspace/yolodist/configs/train/distillation_deeppcb_plain_locfg.toml)
  - [configs/eval/distillation_deeppcb_plain_locfg.toml](/root/workspace/yolodist/configs/eval/distillation_deeppcb_plain_locfg.toml)
- 训练目录：`runs/deeppcb_distill_locfg/distill_plain_locfg/`
- test 评估目录：`runs/deeppcb_distill_locfg/distill_plain_locfg_eval2/`
- test 指标：
  - `precision = 0.9449`
  - `recall = 0.8758`
  - `mAP50 = 0.9405`
  - `mAP50-95 = 0.6247`
- 与 `student_plain(test)` 对比：
  - 指标完全一致
- 记录结论：
  - 第二轮 `loc_fg_distill` 在 `DeepPCB plain student` 上仍未带来增益
  - 当前蒸馏策略对该数据集仍然退化为普通学生训练

### 2026-04-16 DeepPCB distill_epfa_locfg

- 状态：`已完成训练 / 已完成 test 评估`
- 配置：
  - [configs/train/distillation_deeppcb_epfa_locfg.toml](/root/workspace/yolodist/configs/train/distillation_deeppcb_epfa_locfg.toml)
  - [configs/eval/distillation_deeppcb_epfa_locfg.toml](/root/workspace/yolodist/configs/eval/distillation_deeppcb_epfa_locfg.toml)
- 训练目录：`runs/deeppcb_distill_epfa_locfg/distill_epfa_locfg/`
- test 评估目录：`runs/deeppcb_distill_epfa_locfg/distill_epfa_locfg_eval2/`
- test 指标：
  - `precision = 0.9500`
  - `recall = 0.9134`
  - `mAP50 = 0.9619`
  - `mAP50-95 = 0.6678`
- 与 `student_epfa(test)` 对比：
  - 指标完全一致
- 记录结论：
  - 第二轮 `loc_fg_distill` 在 `DeepPCB EPFA student` 上仍未体现蒸馏收益
  - DeepPCB 当前最稳的结论依旧是 `EPFA` 有效、蒸馏无增益

### 2026-04-16 PKU-Market-PCB distill_plain_locfg

- 状态：`已完成训练 / 已完成 test 评估`
- 配置：
  - [configs/train/distillation_pku_market_pcb_plain_locfg.toml](/root/workspace/yolodist/configs/train/distillation_pku_market_pcb_plain_locfg.toml)
  - [configs/eval/distillation_pku_market_pcb_plain_locfg.toml](/root/workspace/yolodist/configs/eval/distillation_pku_market_pcb_plain_locfg.toml)
- 训练目录：`runs/pku_market_pcb_distill_locfg/distill_plain_locfg/`
- test 评估目录：`runs/pku_market_pcb_distill_locfg/distill_plain_locfg_eval2/`
- test 指标：
  - `precision = 0.815`
  - `recall = 0.689`
  - `mAP50 = 0.758`
  - `mAP50-95 = 0.309`
- 记录结论：
  - 第二轮 `loc_fg_distill` 在 `PKU plain student` 上表现不理想
  - 后续若继续保留蒸馏线，需优先尝试更稳的 teacher 监督权重或标签分配策略

### 2026-04-16 PKU-Market-PCB distill_epfa_locfg

- 状态：`已完成训练 / 已完成 test 评估`
- 配置：
  - [configs/train/distillation_pku_market_pcb_epfa_locfg.toml](/root/workspace/yolodist/configs/train/distillation_pku_market_pcb_epfa_locfg.toml)
  - [configs/eval/distillation_pku_market_pcb_epfa_locfg.toml](/root/workspace/yolodist/configs/eval/distillation_pku_market_pcb_epfa_locfg.toml)
- 训练目录：`runs/pku_market_pcb_distill_epfa_locfg/distill_epfa_locfg/`
- test 评估目录：`runs/pku_market_pcb_distill_epfa_locfg/distill_epfa_locfg_eval2/`
- test 指标：
  - `precision = 0.918`
  - `recall = 0.731`
  - `mAP50 = 0.838`
  - `mAP50-95 = 0.368`
- 记录结论：
  - 第二轮 `loc_fg_distill` 在 `PKU EPFA student` 上较 `plain_locfg` 更好
  - 但是否优于 `student_epfa` 仍需等待 `PKU student_epfa` 正式 test 结果统一比对

### 2026-04-16 DsPCBSD+ locfg 续跑

- 状态：`已完成训练 / 已完成 test 评估`
- 问题与处理：
  - 原 `weights/teacher_dspcbsd_plus.pt` 损坏，报错 `failed finding central directory`
  - 已从 `runs/dspcbsd_plus_baseline/baseline_plain/weights/best.pt` 重新复制 teacher
- plain 配置：
  - [configs/train/distillation_dspcbsd_plus_plain_locfg.toml](/root/workspace/yolodist/configs/train/distillation_dspcbsd_plus_plain_locfg.toml)
  - [configs/eval/distillation_dspcbsd_plus_plain_locfg.toml](/root/workspace/yolodist/configs/eval/distillation_dspcbsd_plus_plain_locfg.toml)
- plain 训练目录：`runs/dspcbsd_plus_distill_locfg/distill_plain_locfg/`
- plain test 评估目录：`runs/dspcbsd_plus_distill_locfg/distill_plain_locfg_eval2/`
- plain test 指标：
  - `precision = 0.8044`
  - `recall = 0.7629`
  - `mAP50 = 0.8174`
  - `mAP50-95 = 0.4810`
- epfa 配置：
  - [configs/train/distillation_dspcbsd_plus_epfa_locfg.toml](/root/workspace/yolodist/configs/train/distillation_dspcbsd_plus_epfa_locfg.toml)
  - [configs/eval/distillation_dspcbsd_plus_epfa_locfg.toml](/root/workspace/yolodist/configs/eval/distillation_dspcbsd_plus_epfa_locfg.toml)
- epfa 训练目录：`runs/dspcbsd_plus_distill_epfa_locfg/distill_epfa_locfg/`
- epfa test 评估目录：`runs/dspcbsd_plus_distill_epfa_locfg/distill_epfa_locfg_eval2/`
- epfa test 指标：
  - `precision = 0.7698`
  - `recall = 0.7829`
  - `mAP50 = 0.8188`
  - `mAP50-95 = 0.4750`
- 与已有 `DsPCBSD+ student` 对比：
  - `student_plain(test) = 0.6142 / 0.6029 / 0.6282 / 0.3345`
  - `student_epfa(test) = 0.7346 / 0.7419 / 0.7883 / 0.4485`
- 记录结论：
  - 第二轮 `loc_fg_distill` 在 `DsPCBSD+` 上确实有提升
  - `plain_locfg` 相比 `student_plain` 提升明显，`mAP50-95 +0.1465`
  - `epfa_locfg` 相比 `student_epfa` 也有小幅提升，`Recall +0.0410`, `mAP50 +0.0305`

## 六、后续实验就绪状态

### DeepPCB student_plain

- 状态：`已通过 preflight，等待 teacher 训练完成后启动`

### DeepPCB student_epfa

- 状态：`已通过 preflight，等待 teacher 训练完成后启动`

### DeepPCB distill_plain / distill_epfa

- 状态：`distill_plain 已完成训练 / distill_epfa 可立即启动`
- 说明：
  - `teacher_deeppcb.pt` 已生成
  - 已决定把蒸馏实验统一改为 `100 epoch`
  - `80 epoch` 结果保留作为第一轮筛选记录

## 七、下一步

1. 刷新三数据集统一结果表：
   - `runs/paper_tables/pcb_main_results.csv`
   - `runs/paper_tables/pcb_ablation.csv`
2. 统一比对第二轮蒸馏与普通 student 的跨数据集趋势
3. 若仅 `DsPCBSD+` 上蒸馏有效，论文中将蒸馏表述为“对复杂多类 PCB 数据更有效”

### 2026-04-15 PKU-Market-PCB baseline_plain

- 状态：`已完成训练 / 已完成 test 评估`
- 配置：
  - [configs/train/baseline_pku_market_pcb.toml](/root/workspace/yolodist/configs/train/baseline_pku_market_pcb.toml)
  - [configs/eval/baseline_pku_market_pcb.toml](/root/workspace/yolodist/configs/eval/baseline_pku_market_pcb.toml)
- 命令：

```bash
source /root/workspace/.venv/bin/activate
PYTHONPATH=src python experiments/baseline/train.py --config configs/train/baseline_pku_market_pcb.toml
```

- 启动观察：
  - 数据扫描正常：
    - `train = 555`
    - `val = 46`
    - `corrupt = 0`
  - `nc = 6` 覆盖正确
  - 已进入 `epoch 1`
- 训练末尾验证：
  - `precision = 0.9743`
  - `recall = 0.9039`
  - `mAP50 = 0.9497`
  - `mAP50-95 = 0.5210`
- test 评估目录：`runs/pku_market_pcb_baseline/baseline_plain_eval2/`
- test 指标：
  - `precision = 0.9307`
  - `recall = 0.8471`
  - `mAP50 = 0.9138`
  - `mAP50-95 = 0.4446`

### 2026-04-15 PKU-Market-PCB student_plain

- 状态：`已完成训练 / 已完成 test 评估`
- 配置：
  - [configs/train/modified_model_pku_market_pcb_plain.toml](/root/workspace/yolodist/configs/train/modified_model_pku_market_pcb_plain.toml)
  - [configs/eval/modified_model_pku_market_pcb_plain.toml](/root/workspace/yolodist/configs/eval/modified_model_pku_market_pcb_plain.toml)
- 训练目录：`runs/pku_market_pcb_modified/student_plain/`
- test 评估目录：`runs/pku_market_pcb_modified/student_plain_eval2/`
- 训练末尾验证：
  - `precision = 0.9304`
  - `recall = 0.8680`
  - `mAP50 = 0.9080`
  - `mAP50-95 = 0.4533`
- test 指标：
  - `precision = 0.8154`
  - `recall = 0.6892`
  - `mAP50 = 0.7575`
  - `mAP50-95 = 0.3089`
- 记录结论：
  - `PKU` 上 plain student 相比 baseline 有明显退化
  - 这个数据集继续验证 `EPFA` 和蒸馏是有必要的

### 2026-04-16 PKU-Market-PCB student_epfa

- 状态：`已完成训练 / 已完成 test 评估`
- 配置：
  - [configs/train/modified_model_pku_market_pcb_epfa.toml](/root/workspace/yolodist/configs/train/modified_model_pku_market_pcb_epfa.toml)
  - [configs/eval/modified_model_pku_market_pcb_epfa.toml](/root/workspace/yolodist/configs/eval/modified_model_pku_market_pcb_epfa.toml)
- 训练目录：`runs/pku_market_pcb_epfa/student_epfa/`
- test 评估目录：`runs/pku_market_pcb_epfa/student_epfa_eval2/`
- 训练末尾验证：
  - `precision = 0.9743`
  - `recall = 0.9039`
  - `mAP50 = 0.9497`
  - `mAP50-95 = 0.5210`
- test 指标：
  - `precision = 0.9181`
  - `recall = 0.7310`
  - `mAP50 = 0.8378`
  - `mAP50-95 = 0.3677`
- 与 `student_plain(test)` 对比：
  - `mAP50 +0.0803`
  - `mAP50-95 +0.0588`
  - `Recall +0.0418`
- 记录结论：
  - `EPFA` 在 `PKU` 上同样有效
  - 尽管仍低于 baseline，但能稳定把轻量 student 拉起来
