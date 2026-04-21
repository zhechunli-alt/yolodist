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

### 2026-04-16 DsPCBSD+ 公平性修正重跑

- 状态：`已启动`
- 触发原因：
  - 之前的 `DsPCBSD+ locfg` 蒸馏实验使用的 teacher 来自 baseline 中途阶段，不适合作为论文里的严格主表结论
  - 为消除这一问题，已将 `baseline_plain` 从 `epoch 26` 续跑至 `epoch 100`
  - 后续蒸馏统一改为：
    - 完整 `teacher`
    - `100 epoch`
    - 与对应 student 尽量一致的训练条件
- `baseline_plain(test)` 已补齐：
  - `precision = 0.8128`
  - `recall = 0.8033`
  - `mAP50 = 0.8500`
  - `mAP50-95 = 0.5133`
- 新配置：
  - [configs/train/distillation_dspcbsd_plus_plain_locfg_fair100.toml](/root/workspace/yolodist/configs/train/distillation_dspcbsd_plus_plain_locfg_fair100.toml)
  - [configs/train/distillation_dspcbsd_plus_epfa_locfg_fair100.toml](/root/workspace/yolodist/configs/train/distillation_dspcbsd_plus_epfa_locfg_fair100.toml)
  - [configs/eval/distillation_dspcbsd_plus_plain_locfg_fair100.toml](/root/workspace/yolodist/configs/eval/distillation_dspcbsd_plus_plain_locfg_fair100.toml)
  - [configs/eval/distillation_dspcbsd_plus_epfa_locfg_fair100.toml](/root/workspace/yolodist/configs/eval/distillation_dspcbsd_plus_epfa_locfg_fair100.toml)
- 新脚本：
  - [tools/run_dspcbsd_fair100_distill.sh](/root/workspace/yolodist/tools/run_dspcbsd_fair100_distill.sh)
- 当前执行顺序：
  1. `distill_plain_locfg_fair100`
  2. `distill_epfa_locfg_fair100`
  3. 自动刷新 `runs/paper_tables/`

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

## 七、DeepPCB 外部模型对比

### 2026-04-17 DeepPCB comparison 主线启动

- 状态：`已启动`
- 对比目标：
  - 与 `student_epfa` 做同数据集、同输入尺寸、同 epoch 的外部模型对比
- 当前纳入队列的模型：
  - `YOLOv8n`
  - `YOLOv10n`
  - `SSDLite320-MobileNetV3-Large`
  - `RetinaNet-R50-FPN`
  - `FCOS-R50-FPN`
  - `YOLOX-Nano`
- 统一设置：
  - 数据集：`DeepPCB`
  - 输入尺寸：`640`
  - 训练轮数：`100`
  - `batch = 16`
  - `seed = 42`
- 主要新增文件：
  - [src/yolodist/compare/runner.py](/root/workspace/yolodist/src/yolodist/compare/runner.py)
  - [src/yolodist/data/yolo_detection_dataset.py](/root/workspace/yolodist/src/yolodist/data/yolo_detection_dataset.py)
  - [experiments/comparison/train.py](/root/workspace/yolodist/experiments/comparison/train.py)
  - [experiments/comparison/evaluate.py](/root/workspace/yolodist/experiments/comparison/evaluate.py)
  - [tools/watch_deeppcb_comparison_queue.sh](/root/workspace/yolodist/tools/watch_deeppcb_comparison_queue.sh)
  - [tools/run_deeppcb_yolox_nano.sh](/root/workspace/yolodist/tools/run_deeppcb_yolox_nano.sh)
  - [tools/prepare_deeppcb_coco.py](/root/workspace/yolodist/tools/prepare_deeppcb_coco.py)
  - [tools/run_deeppcb_comparison_remaining.sh](/root/workspace/yolodist/tools/run_deeppcb_comparison_remaining.sh)

### 2026-04-17 DeepPCB YOLOv8n

- 状态：`已完成训练 / 已完成正式 test 评估`
- 配置：
  - [configs/train/comparison_deeppcb_yolov8n.toml](/root/workspace/yolodist/configs/train/comparison_deeppcb_yolov8n.toml)
  - [configs/eval/comparison_deeppcb_yolov8n.toml](/root/workspace/yolodist/configs/eval/comparison_deeppcb_yolov8n.toml)
- 训练目录：`runs/deeppcb_compare/yolov8n/`
- test 评估目录：`runs/deeppcb_compare/yolov8n_eval22/`
- 当前训练末尾验证（epoch 100）：
  - `precision = 0.9840`
  - `recall = 0.9771`
  - `mAP50 = 0.9897`
  - `mAP50-95 = 0.7553`
- 当前判断：
  - `YOLOv8n` 在 `DeepPCB` 上训练正常，且验证集表现强
- test 指标：
  - `precision = 0.9560`
  - `recall = 0.9267`
  - `mAP50 = 0.9688`
  - `mAP50-95 = 0.7519`
- 与 `student_epfa(test)` 对比：
  - `mAP50 +0.0069`
  - `mAP50-95 +0.0841`
  - `Recall +0.0133`
- 当前判断：
  - `YOLOv8n` 在 `DeepPCB` 上已形成正式 test 对照结果
  - 这组结果强于当前 `student_epfa(test)`，后续外部对比模型需要继续统一口径后再整体比较
  - Ultralytics 因目录名冲突，实际将 test 评估结果写入了 `yolov8n_eval22/`，后续队列脚本已改为按前缀识别，避免再次卡住

### 2026-04-17 DeepPCB comparison 执行方式修正

- 状态：`已切换为顺序脚本执行`
- 修正原因：
  - 旧 watcher 在 `eval2 / eval22` 目录名冲突情况下容易停在中间阶段
  - 为确保剩余实验连续跑完，改为一条显式顺序脚本，按固定顺序依次执行
- 新顺序：
  1. `YOLOv10n test`
  2. `SSDLite320-MobileNetV3-Large train/eval`
  3. `RetinaNet-R50-FPN train/eval`
  4. `FCOS-R50-FPN train/eval`
  5. `YOLOX-Nano pipeline`
- 新脚本：
  - [tools/run_deeppcb_comparison_remaining.sh](/root/workspace/yolodist/tools/run_deeppcb_comparison_remaining.sh)

### 2026-04-17 DeepPCB YOLOv10n

- 状态：`已完成训练 / 已完成正式 test 评估`
- 配置：
  - [configs/train/comparison_deeppcb_yolov10n.toml](/root/workspace/yolodist/configs/train/comparison_deeppcb_yolov10n.toml)
  - [configs/eval/comparison_deeppcb_yolov10n.toml](/root/workspace/yolodist/configs/eval/comparison_deeppcb_yolov10n.toml)
- 训练目录：`runs/deeppcb_compare/yolov10n/`
- test 评估目录：`runs/deeppcb_compare/yolov10n_eval22/`
- 当前训练末尾验证（epoch 100）：
  - `precision = 0.9576`
  - `recall = 0.9672`
  - `mAP50 = 0.9886`
  - `mAP50-95 = 0.7722`
- test 指标：
  - `precision = 0.9351`
  - `recall = 0.9314`
  - `mAP50 = 0.9667`
  - `mAP50-95 = 0.7490`
- 与 `student_epfa(test)` 对比：
  - `mAP50 +0.0048`
  - `mAP50-95 +0.0812`
  - `Recall +0.0180`
- 当前判断：
  - `YOLOv10n` 在 `DeepPCB` 上也强于当前 `student_epfa(test)`
  - 该组已形成第二条正式外部对照线
  - 旧顺序脚本停在 `YOLOv10n eval` 之后，但评估结果实际已落盘；后续从 `SSDLite` 继续顺序执行即可

### 2026-04-17 DeepPCB comparison 续跑

- 状态：`进行中`
- 当前剩余模型：
  - `SSDLite320-MobileNetV3-Large`
  - `RetinaNet-R50-FPN`
  - `FCOS-R50-FPN`
  - `YOLOX-Nano`
- 续跑策略：
  - 重新执行 [tools/run_deeppcb_comparison_remaining.sh](/root/workspace/yolodist/tools/run_deeppcb_comparison_remaining.sh)
  - 依靠已有结果目录自动跳过 `YOLOv10n test`
  - 从 `SSDLite train/eval` 开始顺序往后推进

### 2026-04-17 DeepPCB SSDLite320-MobileNetV3-Large

- 状态：`已完成训练 / 已完成正式 test 评估`
- 配置：
  - [configs/train/comparison_deeppcb_ssdlite320.toml](/root/workspace/yolodist/configs/train/comparison_deeppcb_ssdlite320.toml)
  - [configs/eval/comparison_deeppcb_ssdlite320.toml](/root/workspace/yolodist/configs/eval/comparison_deeppcb_ssdlite320.toml)
- 训练目录：`runs/deeppcb_compare/ssdlite320_mobilenet_v3_large/`
- test 评估目录：`runs/deeppcb_compare/ssdlite320_mobilenet_v3_large_eval2/`
- 训练过程观察：
  - 训练推进到 `epoch 100`
  - 自动队列首次停在该组，主要原因是首次运行时需要下载 `mobilenet_v3_large` 预训练权重
  - 重新触发后训练与评估均已完成
- test 指标：
  - `recall = 0.2671`
  - `mAP50 = 0.1709`
  - `mAP50-95 = 0.0461`
- 与 `student_epfa(test)` 对比：
  - `Recall -0.6463`
  - `mAP50 -0.7910`
  - `mAP50-95 -0.6217`
- 当前判断：
  - `SSDLite320-MobileNetV3-Large` 在 `DeepPCB` 上明显弱于当前 `student_epfa`
  - 该组可作为轻量非 YOLO 对照中的弱基线

### 2026-04-17 DeepPCB RetinaNet-R50-FPN

- 状态：`已完成训练 / 已完成正式 test 评估`
- 配置：
  - [configs/train/comparison_deeppcb_retinanet.toml](/root/workspace/yolodist/configs/train/comparison_deeppcb_retinanet.toml)
  - [configs/eval/comparison_deeppcb_retinanet.toml](/root/workspace/yolodist/configs/eval/comparison_deeppcb_retinanet.toml)
- 训练目录：`runs/deeppcb_compare/retinanet_r50_fpn/`
- test 评估目录：`runs/deeppcb_compare/retinanet_r50_fpn_eval2/`
- 训练过程观察：
  - 首次恢复时需要下载 `resnet50` 预训练权重
  - 训练推进到 `epoch 100`
  - `best.pt / last.pt` 均已生成
- test 指标：
  - `recall = 0.8038`
  - `mAP50 = 0.9676`
  - `mAP50-95 = 0.7522`
- 与 `student_epfa(test)` 对比：
  - `Recall -0.1096`
  - `mAP50 +0.0057`
  - `mAP50-95 +0.0844`
- 当前判断：
  - `RetinaNet-R50-FPN` 在 `DeepPCB` 上形成了第三条正式外部对照线
  - 精度指标强于当前 `student_epfa`，但推理更重

### 2026-04-17 DeepPCB FCOS-R50-FPN

- 状态：`已完成训练 / 已完成正式 test 评估`
- 配置：
  - [configs/train/comparison_deeppcb_fcos.toml](/root/workspace/yolodist/configs/train/comparison_deeppcb_fcos.toml)
  - [configs/eval/comparison_deeppcb_fcos.toml](/root/workspace/yolodist/configs/eval/comparison_deeppcb_fcos.toml)
- 训练目录：`runs/deeppcb_compare/fcos_r50_fpn/`
- test 评估目录：`runs/deeppcb_compare/fcos_r50_fpn_eval2/`
- 训练过程观察：
  - 依赖 `resnet50` 预训练权重，已通过提前下载解决等待问题
  - 训练推进到 `epoch 100`
  - `best.pt / last.pt` 均已生成
- test 指标：
  - `recall = 0.8101`
  - `mAP50 = 0.9445`
  - `mAP50-95 = 0.7430`
- 与 `student_epfa(test)` 对比：
  - `Recall -0.1033`
  - `mAP50 -0.0174`
  - `mAP50-95 +0.0752`
- 当前判断：
  - `FCOS-R50-FPN` 的 `mAP50-95` 高于当前 `student_epfa`
  - 但 `Recall` 和 `mAP50` 低于当前 `student_epfa`
  - 该组可作为 anchor-free 非 YOLO 对照

### 2026-04-17 DeepPCB YOLOX-Nano

- 状态：`未完成 / 已停止`
- 配置：
  - [configs/models/yolox_deeppcb_nano.py](/root/workspace/yolodist/configs/models/yolox_deeppcb_nano.py)
  - [configs/train/comparison_deeppcb_yolox_nano.toml](/root/workspace/yolodist/configs/train/comparison_deeppcb_yolox_nano.toml)
  - [configs/eval/comparison_deeppcb_yolox_nano.toml](/root/workspace/yolodist/configs/eval/comparison_deeppcb_yolox_nano.toml)
- 日志：
  - `runs/overnight_logs/deeppcb_yolox_nano_20260417T145643Z.log`
- 当前观察：
  - `DeepPCB COCO` 转换已完成
  - 管线停在 `pip install -e external/YOLOX`
  - 原因是 `pip` 的 build isolation 环境内无法导入 `torch`
- 错误结论：
  - 该次运行未进入 `YOLOX train.py`
  - 当前仓库已把安装方式修正为 `PIP_NO_BUILD_ISOLATION=1 pip install -e external/YOLOX`
  - 下次恢复时可以基于修正后的脚本直接重试

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

## 2026-04-20 晚间恢复与并行主线推进

- 修正了 `DeepPCB distill_epfa` 的公平蒸馏条件：
  - 伪标签数据目录改为独立 `*_distill_epfa_fair`
  - 图片改为真实复制，避免 Ultralytics 沿原始路径复用旧 `labels/train.cache`
  - `distill_alpha` 提高到 `0.5`
  - 打开 `feature_kd_loss`
  - `feature_kd_alpha = 0.1`
- 修正后正式 test 结果：
  - `distill_epfa_fair(test)`：`precision = 0.9404`
  - `recall = 0.9179`
  - `mAP50 = 0.9628`
  - `mAP50-95 = 0.6865`
- 记录结论：
  - 修正后 `distill_epfa` 不再与旧 `student_epfa` 完全重合
  - 当前仍未超过 `student_epfa(test)`，但蒸馏链路已经真正生效

- 新增 `PKU-Market-PCB` 与 `DsPCBSD+` 自动主线脚本：
  - [tools/run_dataset_mainline.sh](/root/workspace/yolodist/tools/run_dataset_mainline.sh)
- 统一主线顺序：
  - `baseline -> eval -> teacher复制 -> student_plain -> student_epfa -> distill_epfa -> summarize`
- 当前运行状态：
  - `PKU baseline` 已完成 100 epoch，后半程已自动接到 `student_plain`
  - `DsPCBSD+ baseline` 已启动并进入训练
- 公平性说明：
  - `PKU` 与 `DsPCBSD+` 的 `distill_epfa` 也已同步切换到 `fair` 条件
  - `epochs = 100`
  - `distill_alpha = 0.5`
  - `feature_kd_loss = true`
  - `feature_kd_alpha = 0.1`

- 新增过夜自动调度：
  - [tools/run_pcb_overnight_mainlines.sh](/root/workspace/yolodist/tools/run_pcb_overnight_mainlines.sh)
  - 作用：
    - 自动巡检 `PKU-Market-PCB` 与 `DsPCBSD+`
    - 若某条主线中断或单步完成后无后继任务，会自动接上下一步
  - 目标：
    - 夜间持续占用 GPU
    - 尽量减少人工守夜和手动续跑

## 2026-04-21 蒸馏策略 tuned 版

- 为进一步提升 `student_epfa` 的蒸馏收益，新开一条 `distill_epfa_tuned` 路线，不覆盖现有 `fair` 结果。
- tuned 版核心参数：
  - `kd_strategy = "loc_fg_distill"`
  - `distill_alpha = 0.35`
  - `distill_temperature = 3.0`
  - `enable_feature_kd_loss = true`
  - `feature_kd_alpha = 0.05`
  - `cls_kd_alpha = 0.20`
  - `loc_kd_alpha = 1.25`
  - `bg_weight = 0.02`
  - `mask_expand_ratio = 0.05`
  - `teacher_conf = 0.30`
  - `teacher_iou_threshold = 0.6`
- 设计意图：
  - 降低总蒸馏权重，减少 teacher 过强约束对学生主检测目标的压制
  - 提高定位蒸馏比重，争取同时改善 `Recall` 与 `mAP50-95`
  - 降低 feature KD 权重，减少中间特征过拟合 teacher 的风险
  - 提高 teacher 伪标签置信阈值，减少噪声框
- 新增配置：
  - `configs/train/distillation_deeppcb_epfa_tuned.toml`
  - `configs/train/distillation_pku_market_pcb_epfa_tuned.toml`
  - `configs/train/distillation_dspcbsd_plus_epfa_tuned.toml`
  - 对应 `configs/eval/*_tuned.toml`
- 新增调度脚本：
  - [tools/run_epfa_tuned_distill_all.sh](/root/workspace/yolodist/tools/run_epfa_tuned_distill_all.sh)
  - [tools/run_epfa_tuned_remaining.sh](/root/workspace/yolodist/tools/run_epfa_tuned_remaining.sh)
- 当前状态：
  - `DeepPCB distill_epfa_tuned` 已启动
  - `PKU` 与 `DsPCBSD+` 将在 `DeepPCB` tuned 训练结束后自动接续
