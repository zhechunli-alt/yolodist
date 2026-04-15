# PCB 实验记录

这份文档用于沉淀 PCB 主线实验的真实执行记录，目标是后续可以直接作为毕业论文实验章节和方法章节的事实依据。

记录原则：

1. 只写已经发生的事实
2. 每条记录尽量包含命令、配置、结果、结论
3. 区分“计划中 / 进行中 / 已完成 / 已失败”

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

## 五、正在运行的实验

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

- 状态：`进行中`
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
  - 接下来可直接作为 teacher 基线使用
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

## 六、后续实验就绪状态

### DeepPCB student_plain

- 状态：`已通过 preflight，等待 teacher 训练完成后启动`

### DeepPCB student_epfa

- 状态：`已通过 preflight，等待 teacher 训练完成后启动`

### DeepPCB distill_plain / distill_epfa

- 状态：`配置与数据已就绪`
- 当前唯一阻塞：
  - `weights/teacher_deeppcb.pt` 尚未生成
- 说明：
  - baseline 完成后复制 `best.pt -> teacher_deeppcb.pt` 即可启动

## 七、下一步

1. 等 `DsPCBSD+ baseline` 产出 teacher 权重
2. 立即启动 `DeepPCB baseline_plain`
3. DeepPCB baseline 完成后，优先跑：
   - `student_plain`
   - `student_epfa`
4. 如果 `student_epfa > student_plain`，再进入：
   - `distill_plain`
   - `distill_epfa`
