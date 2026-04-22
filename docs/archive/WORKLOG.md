# 工作记录

这份文档按“工程笔记”的方式记录项目推进过程。每一步都尽量说明三件事：

1. 做了什么
2. 为什么要这么做
3. 你可以从这一步学到什么

## 2026-03-11 第一步：重构项目目录

### 做了什么

- 按较标准的机器学习项目结构重组了仓库目录：
  - `datasets/`
  - `experiments/`
  - `runs/`
  - `weights/`
  - `ultralytics-src/`
- 把 MVTec 压缩包移动到了 `datasets/mvtec/`
- 把 MVTec AD 数据集解压到了 `datasets/mvtec/`
- 把原来根目录的测试脚本移动到了 [experiments/baseline/predict.py](/Users/lizhechun/Desktop/yolodist/experiments/baseline/predict.py)

### 为什么要这么做

- 论文项目和普通练习脚本不一样，数据、代码、权重、输出结果如果都混在根目录，后面会非常难维护。
- `experiments/` 目录适合放实验入口脚本，因为它可以清楚表达“这个脚本属于哪个实验方向”。
- `runs/` 单独拿出来非常重要，因为 Ultralytics 训练时会生成很多结果文件，不分离的话仓库会很快变乱。

### 你可以学到什么

- 项目结构不是表面工作，而是后续实验效率的一部分。
- 当你准备做多组实验、保存多版结果时，一个清晰的目录结构会直接减少很多低级错误。

## 2026-03-11 第二步：阅读选题并收缩为可落地目标

### 做了什么

- 阅读了导师给的选题文档。
- 从完整选题中提炼出当前最适合先落地的“模型主线”。
- 把第一阶段目标从“大而全”的方案收缩为一条更可执行的路径：
  - MVTec 数据预处理
  - baseline 检测模型
  - student / modified model 实验
  - 蒸馏实验

### 为什么要这么做

- 选题文档里的目标覆盖了数据增强、模型改进、蒸馏、部署、前后端系统，这个范围适合中长期规划，但不适合一上来同时实现。
- 如果模型训练链路本身还不稳定，就去做部署系统，价值会很低，因为你只是把一个还没定型的模型“包装起来”。
- 对毕业论文来说，最重要的第一步通常是先把训练、评估、对比这条实验链跑通。

### 你可以学到什么

- 做研究工程时，第一件事往往不是“加更多功能”，而是“缩小范围，先做出最有价值的闭环”。
- 一个完整但跑不通的大系统，不如一个范围清楚、可以反复实验的小系统。

## 2026-03-11 第三步：确定当前技术路线

### 做了什么

- 选择 MVTec AD 作为第一阶段的核心数据集。
- 决定把 MVTec 从“异常分割任务”转换成“YOLO 检测任务”。
- 默认选择 `binary` 检测方式，也就是只区分“缺陷 / 非缺陷”。
- 蒸馏策略先选择“教师模型伪标签蒸馏”。
- `modified_model` 目录先作为后续轻量化模型实验的入口预留出来。

### 为什么要这么做

- MVTec 原始标注是 mask，不是 YOLO 需要的边界框，所以必须先做数据转换。
- `binary` 是当前最稳妥的起点，因为 MVTec 的缺陷类别本身比较复杂，而且不同类数据量不均衡。
- 伪标签蒸馏是目前最现实的第一版蒸馏方法，因为它不要求你一开始就深入修改 Ultralytics 内部结构。
- 如果一上来就做中间层特征蒸馏或深度结构改造，工程难度会明显上升，容易把项目卡住。

### 你可以学到什么

- “可行”并不等于“最先进”，而是指当前阶段最容易做出稳定实验结果的方法。
- 在论文项目里，先做一个能跑通的蒸馏 baseline，通常比一开始追求复杂算法更合理。

## 2026-03-11 第四步：搭建项目代码骨架

### 做了什么

- 新增了项目说明文件 [README.md](/Users/lizhechun/Desktop/yolodist/README.md)
- 新增了依赖清单 [requirements.txt](/Users/lizhechun/Desktop/yolodist/requirements.txt)
- 新增了配置文件：
  - [configs/data/mvtec_detection.toml](/Users/lizhechun/Desktop/yolodist/configs/data/mvtec_detection.toml)
  - [configs/train/baseline.toml](/Users/lizhechun/Desktop/yolodist/configs/train/baseline.toml)
  - [configs/train/modified_model.toml](/Users/lizhechun/Desktop/yolodist/configs/train/modified_model.toml)
  - [configs/train/distillation.toml](/Users/lizhechun/Desktop/yolodist/configs/train/distillation.toml)
- 新增了共享代码包 `src/yolodist/`

### 为什么要这么做

- 实验参数如果直接写死在脚本里，后面调参数、换数据集、换模型都会很麻烦。
- 把逻辑做成共享包后，baseline、modified model、distillation 三条线就能共用代码，避免复制粘贴。
- 这种结构更适合以后迁移到服务器上跑实验。

### 你可以学到什么

- 配置文件的作用不仅是“看起来规范”，更重要的是把“实验设定”和“代码逻辑”分离开。
- 多实验项目里，公共逻辑尽量放在 `src/` 里，入口脚本尽量保持轻薄。

## 2026-03-11 第五步：实现 MVTec 到 YOLO 的数据转换

### 做了什么

- 新增数据准备脚本 [tools/prepare_mvtec_detection.py](/Users/lizhechun/Desktop/yolodist/tools/prepare_mvtec_detection.py)
- 新增数据转换核心实现 [src/yolodist/data/mvtec_detection.py](/Users/lizhechun/Desktop/yolodist/src/yolodist/data/mvtec_detection.py)
- 实现了这些能力：
  - 自动发现 MVTec 类别目录
  - 自动寻找对应 mask
  - 从 mask 中提取连通区域
  - 把连通区域转换为 YOLO 格式边界框
  - 生成 YOLO 标签文件
  - 划分 train / val / test
  - 生成 `data.yaml`
  - 生成 `manifest.json`

### 为什么要这么做

- MVTec 原始数据不能直接拿给 YOLO 检测代码训练，必须经过一层“任务格式转换”。
- `manifest.json` 很有用，因为它记录了每张图像最终被分到了哪个 split，后面出问题时容易排查。
- 数据预处理是整个实验链最基础的一层，必须先做扎实。

### 你可以学到什么

- 真实研究项目里，数据适配往往比训练代码本身更关键。
- 模型能不能跑通，很多时候不是由模型结构决定，而是由数据处理是否正确决定。

## 2026-03-11 第六步：统一训练入口

### 做了什么

- 新增统一训练入口实现 [src/yolodist/train/ultralytics_runner.py](/Users/lizhechun/Desktop/yolodist/src/yolodist/train/ultralytics_runner.py)
- 更新了实验脚本：
  - [experiments/baseline/train.py](/Users/lizhechun/Desktop/yolodist/experiments/baseline/train.py)
  - [experiments/modified_model/train.py](/Users/lizhechun/Desktop/yolodist/experiments/modified_model/train.py)
  - [experiments/baseline/predict.py](/Users/lizhechun/Desktop/yolodist/experiments/baseline/predict.py)

### 为什么要这么做

- 最开始的占位脚本只是打印提示信息，不能真正作为后续实验入口。
- 统一训练入口后，不同实验只需要切换配置文件，不需要重复写训练逻辑。
- 路径处理统一后，后面上服务器时更不容易因为相对路径出错。

### 你可以学到什么

- 入口脚本应该尽量简单，只负责“调起某个流程”。
- 真实逻辑最好沉淀到共享模块里，这样后面扩展实验会更省力。

## 2026-03-11 第七步：实现第一版蒸馏流程

### 做了什么

- 新增蒸馏实现 [src/yolodist/distill/pseudo_label.py](/Users/lizhechun/Desktop/yolodist/src/yolodist/distill/pseudo_label.py)
- 更新了 [experiments/distillation/train.py](/Users/lizhechun/Desktop/yolodist/experiments/distillation/train.py)
- 当前蒸馏流程包括：
  1. 复制或链接已准备好的检测数据集
  2. 使用教师模型为缺少训练标签的样本生成伪标签
  3. 基于生成后的数据训练学生模型

### 为什么要这么做

- 选题的核心之一是轻量化模型蒸馏，所以项目里必须有一条教师模型到学生模型的训练路径。
- 当前先选伪标签蒸馏，是因为它容易落地，且不依赖修改 YOLO 内部训练器。
- 这能先建立起“教师模型存在时，学生模型如何受益”的实验闭环。

### 你可以学到什么

- 蒸馏不一定一开始就要做最复杂的中间特征蒸馏。
- 一个可以重复运行、可以做对比实验的简单蒸馏版本，通常更适合作为第一版。

## 2026-03-11 第八步：增加训练前检查和仓库清理规则

### 做了什么

- 在以下文件中加入了路径存在性检查：
  - [src/yolodist/train/ultralytics_runner.py](/Users/lizhechun/Desktop/yolodist/src/yolodist/train/ultralytics_runner.py)
  - [src/yolodist/distill/pseudo_label.py](/Users/lizhechun/Desktop/yolodist/src/yolodist/distill/pseudo_label.py)
- 更新了 [.gitignore](/Users/lizhechun/Desktop/yolodist/.gitignore)，忽略：
  - `datasets/processed/`
  - `__pycache__/`
  - `*.pyc`

### 为什么要这么做

- 服务器训练很耗时间，如果因为路径没配对而在运行后才报错，会浪费很多时间。
- 处理后的数据一般不适合直接提交进 Git。
- Python 编译缓存文件也不应该污染工作区。

### 你可以学到什么

- “防御性编程”在研究工程里非常重要，尤其是训练任务一跑可能就是几个小时甚至几天。
- 很多高质量项目并不是功能更多，而是能更早、更明确地暴露错误。

## 2026-03-11 第九步：增加评估与实验结果汇总能力

### 做了什么

- 新增评估配置：
  - [configs/eval/baseline.toml](/Users/lizhechun/Desktop/yolodist/configs/eval/baseline.toml)
  - [configs/eval/modified_model.toml](/Users/lizhechun/Desktop/yolodist/configs/eval/modified_model.toml)
  - [configs/eval/distillation.toml](/Users/lizhechun/Desktop/yolodist/configs/eval/distillation.toml)
- 新增评估实现：
  - [src/yolodist/eval/runner.py](/Users/lizhechun/Desktop/yolodist/src/yolodist/eval/runner.py)
  - [experiments/baseline/evaluate.py](/Users/lizhechun/Desktop/yolodist/experiments/baseline/evaluate.py)
  - [experiments/modified_model/evaluate.py](/Users/lizhechun/Desktop/yolodist/experiments/modified_model/evaluate.py)
  - [experiments/distillation/evaluate.py](/Users/lizhechun/Desktop/yolodist/experiments/distillation/evaluate.py)
- 新增实验汇总工具：
  - [src/yolodist/reporting/summary.py](/Users/lizhechun/Desktop/yolodist/src/yolodist/reporting/summary.py)
  - [tools/summarize_run.py](/Users/lizhechun/Desktop/yolodist/tools/summarize_run.py)

### 为什么要这么做

- 一个项目如果只有训练，没有标准化评估，就很难做实验对比。
- 毕业论文需要可复现、可比较的结果，而不是只证明“代码能跑”。
- 自动汇总工具能帮你把 Ultralytics 生成的大量原始结果，整理成后续写论文时更容易使用的形式。

### 你可以学到什么

## 2026-04-15 第十步：迁移到 PCB 检测主线

### 做了什么

- 新增 PCB 通用数据准备核心 [src/yolodist/data/pcb_detection.py](/root/workspace/yolodist/src/yolodist/data/pcb_detection.py)
- 新增统一脚本 [tools/prepare_pcb_detection.py](/root/workspace/yolodist/tools/prepare_pcb_detection.py)
- 新增 DeepPCB 下载脚本 [tools/download_deeppcb_from_github.py](/root/workspace/yolodist/tools/download_deeppcb_from_github.py)
- 新增三份数据配置：
  - [configs/data/deeppcb.toml](/root/workspace/yolodist/configs/data/deeppcb.toml)
  - [configs/data/pku_market_pcb.toml](/root/workspace/yolodist/configs/data/pku_market_pcb.toml)
  - [configs/data/dspcbsd_plus.toml](/root/workspace/yolodist/configs/data/dspcbsd_plus.toml)
- 新增类别映射文档 [docs/PCB_CLASS_MAPPING.md](/root/workspace/yolodist/docs/PCB_CLASS_MAPPING.md)
- 扩展了 [tools/preflight_check.py](/root/workspace/yolodist/tools/preflight_check.py)，现在能检查处理后数据集的空标签比例、坏图和类别越界。

### 为什么要这么做

- MVTec 转检测的指标说明这条路不适合作为当前阶段主实验。
- PCB 数据天然是检测任务，更适合用统一的 YOLO 训练与评估流程产出论文指标。
- 三个 PCB 数据集的原始格式不一定一致，所以准备脚本必须做成“自动探测布局”而不是把路径写死。

### 你可以学到什么

- 数据迁移最怕“只为一套数据写死逻辑”，那样第二套数据一来就要重写。
- 提前把 alias 映射和一致性检查做好，后面调模型时就更容易把问题归因到模型而不是数据。

## 2026-04-15 第十一步：落地 EPFA-Lite

### 做了什么

- 新增模块实现 [src/yolodist/models/epfa.py](/root/workspace/yolodist/src/yolodist/models/epfa.py)
- 更新注册入口 [src/yolodist/models/registry.py](/root/workspace/yolodist/src/yolodist/models/registry.py)
- 新增模型配置 [configs/models/yolo11_student_epfa.yaml](/root/workspace/yolodist/configs/models/yolo11_student_epfa.yaml)
- 在 train / eval / distill runner 里增加了轻量上下文 hook，使 EPFA 能读取当前输入图像的 Sobel 边缘先验。

### 为什么要这么做

- PCB 缺陷里常见细边缘、断裂、缺口、毛刺，小目标和边缘提示比重型全局建模更重要。
- 直接上 Transformer 风险太高，不符合“结果第一，创新第二”的节奏。
- EPFA 把 ECA 风格通道门控和固定 Sobel 先验组合在一起，参数增量可控，也方便写消融。

### 你可以学到什么

- 创新模块不一定要追求复杂，关键是要和任务特征有明确对应关系。
- 当模块需要访问原始输入图像时，局部 hook 往往比重写训练框架更稳。

## 2026-04-15 第十二步：补全 PCB 实验矩阵和论文表导出

### 做了什么

- 新增 `DeepPCB / PKU-Market-PCB / DsPCBSD+` 的 baseline、plain、EPFA、distill plain、distill EPFA 配置。
- 新增统一执行脚本 [tools/run_pcb_pipeline.sh](/root/workspace/yolodist/tools/run_pcb_pipeline.sh)
- 扩展 [tools/summarize_run.py](/root/workspace/yolodist/tools/summarize_run.py)，可以生成：
  - `runs/paper_tables/pcb_main_results.csv`
  - `runs/paper_tables/pcb_ablation.csv`

### 为什么要这么做

- 一套论文实验如果没有统一配置和统一导表，很容易最后手工抄错结果。
- 把四组主实验固定下来后，后续就能专注于“哪组提升了、为什么提升”。

### 你可以学到什么

- 真正节省时间的不是少跑几组实验，而是把每组实验的入口和输出约束成一致格式。
- 提前把汇总表做好，会显著降低后面写论文时的认知负担。

## 2026-04-15 第十三步：开始真实 PCB 实验并建立论文记录

### 做了什么

- 新增实验记录文档 [docs/PCB实验记录.md](/root/workspace/yolodist/docs/PCB实验记录.md)
- 把三个数据集的实际接入状态写成了可追溯记录：
  - `DeepPCB`
  - `PKU-Market-PCB`
  - `DsPCBSD+`
- 启动了第一条真实训练任务：`DsPCBSD+ baseline_plain`

### 为什么要这么做

- 你后面写毕业论文时，不只需要最终分数，还需要“中间是怎么推进的、遇到了什么问题、为什么这么设计”。
- 如果只保存最终权重和结果图，后面很容易忘记数据是怎么拆分、模块是什么时候插进去的、某次实验为什么停掉。
- 单独维护一份实验记录，可以把工程推进过程转化成论文里的方法与实验叙事。

### 你可以学到什么

- 研究项目里的记录不是附属工作，而是结果可信度的一部分。
- 越早把“命令、配置、结论”同步下来，后面写论文越轻松。

- 在实验项目里，“能训练”只是第一步，“能评估、能对比、能复盘”才是完整链路。
- 很多时候，一个小的结果汇总脚本，能在后期节省大量整理实验表格的时间。

## 2026-03-11 第十步：补充实验流程文档

### 做了什么

- 新增 [docs/EXPERIMENT_FLOW.md](/Users/lizhechun/Desktop/yolodist/docs/EXPERIMENT_FLOW.md)
- 在 [README.md](/Users/lizhechun/Desktop/yolodist/README.md) 中补充了评估和结果汇总的命令说明

### 为什么要这么做

- 项目不仅要“你现在能看懂”，还要保证你之后换到服务器环境时还能快速回忆整个使用流程。
- 文档化实验顺序，可以减少后期“先跑哪个脚本、后跑哪个脚本”的混乱。

### 你可以学到什么

- 文档不是项目完成后的附属品，而是工程的一部分。
- 写清流程，本质上是在降低未来自己的理解成本。

## 当前已完成的验证

- 新增 Python 脚本已通过语法检查
- MVTec mask 到 YOLO box 的关键路径已做过冒烟验证
- 本地没有安装训练依赖
- 本地没有真正启动训练任务

## 当前项目已经达到的状态

现在这个仓库已经具备后续在服务器上进行以下工作的基础：

1. 安装依赖
2. 准备检测格式数据集
3. 放置 teacher / student / baseline 权重
4. 运行训练
5. 运行评估
6. 汇总实验结果

## 建议的下一步

1. 给 `modified_model` 接入真正的轻量化学生模型，而不是暂时复用同一套 YOLO11 权重入口
2. 为每次训练和评估增加“配置快照”与“运行说明”保存能力
3. 增加 NEU-DET 数据集支持，形成更完整的论文实验对比
4. 在 baseline 稳定后，决定是否继续使用 `binary` 检测，还是切换到更细粒度的 `defect_type` 检测

## 2026-04-12 第十一步：接入真实的轻量学生模型配置

### 做了什么

- 新增了轻量学生模型 YAML：
  - [configs/models/yolo11_student.yaml](/Users/lizhechun/Desktop/yolodist/configs/models/yolo11_student.yaml)
- 更新了 `modified_model` 训练配置：
  - [configs/train/modified_model.toml](/Users/lizhechun/Desktop/yolodist/configs/train/modified_model.toml)
  - `model` 改为学生模型 YAML
  - 新增 `pretrained`，用于加载 `weights/yolo11n.pt` 作为初始化权重
- 更新了蒸馏训练配置：
  - [configs/train/distillation.toml](/Users/lizhechun/Desktop/yolodist/configs/train/distillation.toml)
  - `student_model` 改为学生模型 YAML
  - 新增 `student_pretrained`

### 为什么要这么做

- 之前的 `modified_model` 只是“实验入口”，并没有真正体现结构上的学生模型。
- 论文如果要做“轻量化模型 + 蒸馏”对比，至少要有一个明确的学生模型定义文件。
- 用 `yaml + pretrained` 的方式训练，既能控制结构，又能利用预训练权重提升稳定性。

### 你可以学到什么

- 轻量化实验要“可复现”，不能只说“我训练了一个学生模型”，而要给出可追溯的结构定义文件。
- 配置化地绑定结构和权重，是后续消融实验（改结构不改训练策略、改训练策略不改结构）的基础。

## 2026-04-12 第十二步：升级蒸馏策略与实验可复现记录

### 做了什么

- 升级蒸馏实现：
  - [src/yolodist/distill/pseudo_label.py](/Users/lizhechun/Desktop/yolodist/src/yolodist/distill/pseudo_label.py)
- 新增两种伪标签策略：
  - `fill_empty`：只给空标签样本补教师框
  - `merge_teacher`：在 IoU 过滤后将教师框并入训练标签
- 新增蒸馏相关超参配置：
  - `teacher_iou_threshold`
  - `teacher_max_det`
  - `distill_temperature`
  - `distill_alpha`
- 新增运行清单记录工具：
  - [src/yolodist/reporting/manifest.py](/Users/lizhechun/Desktop/yolodist/src/yolodist/reporting/manifest.py)
- 训练与评估流程接入清单记录：
  - [src/yolodist/train/ultralytics_runner.py](/Users/lizhechun/Desktop/yolodist/src/yolodist/train/ultralytics_runner.py)
  - [src/yolodist/eval/runner.py](/Users/lizhechun/Desktop/yolodist/src/yolodist/eval/runner.py)

### 为什么要这么做

- 伪标签蒸馏不是单一方法，至少要可切换策略，才能做对比实验并回答“蒸馏到底带来了什么收益”。
- 论文实验最怕“过几周后不知道当时怎么跑出来的”，所以需要自动保存运行时配置和上下文。
- 将蒸馏超参先进入配置，即使当前还没实现完整特征蒸馏损失，也能为下一步扩展保留统一入口。

### 你可以学到什么

- 做论文实验时，代码能跑只是底线，真正关键是“同样配置能不能再次跑出同类结果”。
- 当你给自己留好策略开关和参数入口，后续做消融实验会快很多。

## 2026-04-14 第十三步：提出并文档化 PPLA 创新模块方案

### 做了什么

- 新增模块设计文档：
  - [PPLA模块设计.md](/Users/lizhechun/Desktop/yolodist/docs/PPLA模块设计.md)
- 在文档中明确了：
  - 可行性评估
  - 模块结构定义
  - YOLO 接入位置
  - 与 baseline / modified / distill 的关系
  - 分阶段落地步骤
  - 消融实验设计
  - 风险与兜底

### 为什么要这么做

- 你当前项目已经有训练和蒸馏骨架，下一步需要一个真正“可写进论文创新点”的模块方向。
- 在动代码前先形成模块设计文档，可以避免后续反复返工。
- 该方案与选题的工业背景一致：小目标、长尾、轻量部署。

### 你可以学到什么

- 好的创新点不是“复杂模块堆叠”，而是“问题驱动 + 可落地 + 可验证”的组合。
- 先有可执行文档，再做代码接入，能显著提高研究工程效率。

## 2026-04-14 第十四步：完成 PPLA 可运行代码接入与服务器首跑增强

### 做了什么

- 新增 PPLA 模块代码：
  - [src/yolodist/models/ppla.py](/Users/lizhechun/Desktop/yolodist/src/yolodist/models/ppla.py)
- 新增 Ultralytics 运行时注册器：
  - [src/yolodist/models/registry.py](/Users/lizhechun/Desktop/yolodist/src/yolodist/models/registry.py)
- 在训练、蒸馏、评估入口中接入注册逻辑：
  - [src/yolodist/train/ultralytics_runner.py](/Users/lizhechun/Desktop/yolodist/src/yolodist/train/ultralytics_runner.py)
  - [src/yolodist/distill/pseudo_label.py](/Users/lizhechun/Desktop/yolodist/src/yolodist/distill/pseudo_label.py)
  - [src/yolodist/eval/runner.py](/Users/lizhechun/Desktop/yolodist/src/yolodist/eval/runner.py)
- 更新学生模型配置：
  - [configs/models/yolo11_student.yaml](/Users/lizhechun/Desktop/yolodist/configs/models/yolo11_student.yaml)（加入 PPLA）
  - [configs/models/yolo11_student_plain.yaml](/Users/lizhechun/Desktop/yolodist/configs/models/yolo11_student_plain.yaml)（plain 消融对照）
- 增加 plain 对照的训练与评估配置：
  - [configs/train/modified_model_plain.toml](/Users/lizhechun/Desktop/yolodist/configs/train/modified_model_plain.toml)
  - [configs/train/distillation_plain.toml](/Users/lizhechun/Desktop/yolodist/configs/train/distillation_plain.toml)
  - [configs/eval/modified_model_plain.toml](/Users/lizhechun/Desktop/yolodist/configs/eval/modified_model_plain.toml)
  - [configs/eval/distillation_plain.toml](/Users/lizhechun/Desktop/yolodist/configs/eval/distillation_plain.toml)
- 实验入口改为支持 `--config` 参数，便于服务器批量实验：
  - `experiments/*/train.py`
  - `experiments/*/evaluate.py`
- 新增服务器预检查脚本：
  - [tools/preflight_check.py](/Users/lizhechun/Desktop/yolodist/tools/preflight_check.py)
- 同步更新文档：
  - [README.md](/Users/lizhechun/Desktop/yolodist/README.md)
  - [docs/EXPERIMENT_FLOW.md](/Users/lizhechun/Desktop/yolodist/docs/EXPERIMENT_FLOW.md)
  - [docs/服务器首跑清单.md](/Users/lizhechun/Desktop/yolodist/docs/服务器首跑清单.md)

### 为什么要这么做

- 仅有“模块设计文档”还不够，必须进入可执行代码阶段，才能真正验证创新点。
- 运行时注册方案可以在不重度改动第三方源码的前提下，把自定义模块接入 Ultralytics。
- 同时保留 plain 与 PPLA 双分支，是论文消融实验的必要条件。
- 预检查脚本能在训练前提前暴露依赖/路径问题，减少服务器时间浪费。

### 你可以学到什么

- 研究工程推进应遵循“设计 -> 实现 -> 对照实验”三步闭环，而不是只停留在思路层。
- 消融实验要从配置层就开始设计，否则后面很难保证对比公平。
- 面向服务器的可运行性，不只是代码能跑，还包括参数入口、预检查和流程文档的一致性。

## 2026-04-14 第十五步：让 distill_temperature/distill_alpha 真实参与蒸馏损失

### 做了什么

- 在蒸馏训练中接入响应蒸馏损失（response KD）：
  - [src/yolodist/distill/pseudo_label.py](/Users/lizhechun/Desktop/yolodist/src/yolodist/distill/pseudo_label.py)
- 新增配置开关：
  - `enable_kd_loss = true`（见 `configs/train/distillation*.toml`）
- 将总损失改为：
  - `L_total = (1 - alpha) * L_det + alpha * L_kd(T)`
- `distill_temperature` 用于 KD 温度缩放，`distill_alpha` 用于检测损失与 KD 损失加权融合。

### 为什么要这么做

- 之前这两个参数只记录在配置中，并不会真实影响反向传播。
- 论文里如果要讨论温度和蒸馏权重超参，必须保证参数真正进入损失函数。
- 这一步把蒸馏从“伪标签增强训练”推进到了“伪标签 + 响应蒸馏”的组合范式。

### 你可以学到什么

- 研究代码中的“参数存在”不等于“参数生效”，要看它是否参与梯度计算。
- 当你把超参显式接入损失函数后，消融实验才有统计意义和可解释性。

## 2026-04-15 第十六步：新增夜间自动训练流水线脚本

### 做了什么

- 新增一键串行训练脚本：
  - [run_pipeline_overnight.sh](/Users/lizhechun/Desktop/yolodist/tools/run_pipeline_overnight.sh)
- 脚本自动执行：
  1. preflight 检查
  2. 数据准备
  3. baseline 训练
  4. 复制 baseline 最佳权重为 `teacher.pt`
  5. modified(PPLA) 训练
  6. distill(PPLA) 训练
  7. 三条线评估
  8. 结果 summary 生成
- 每个阶段均写入独立日志文件到：
  - `runs/overnight_logs/*.log`

### 为什么要这么做

- 你的训练通常会跨夜执行，手动盯流程容易遗漏步骤。
- 串行脚本可以固定实验顺序，减少“阶段遗漏”和“手误参数不一致”。
- 自动日志拆分更利于第二天快速定位失败阶段。

### 你可以学到什么

- 当实验链路变长后，自动化脚本是保障复现性和执行效率的关键。
- 先把流程固化，再做算法对比，能显著降低实验管理成本。

## 2026-04-15 第十七步：新增并行工作步骤文档（训练外协同）

### 做了什么

- 新增并行协作文档：
  - [并行工作步骤文档.md](/Users/lizhechun/Desktop/yolodist/docs/并行工作步骤文档.md)
- 文档包含：
  - 线程拆分（A~E）
  - 输入输出接口契约
  - 固定产物路径与字段定义
  - 后端 API 最小接口约定
  - Git 分支和合并规范
  - 每日同步模板建议

### 为什么要这么做

- 主线程在训练时，如果没有并行协作规范，其他线程容易做重复或冲突工作。
- 先定义“接口与产物契约”，可以让数据、后端、部署、报告线程同步推进，不互相阻塞。
- 这样训练结果出来后可以直接接到报告和系统，不需要返工。

### 你可以学到什么

- 多线程协作的关键不是“多开几个任务”，而是先约定清晰的数据和接口边界。
- 研究工程中，接口契约和目录契约能显著降低跨模块联调成本。

## 2026-04-15 第十八步：完成线程C后端最小可用实现（Flask + 模型管理 + SQLite）

### 做了什么

- 新增后端服务目录与核心模块：
  - [service/backend/app.py](/Users/lizhechun/Desktop/yolodist/service/backend/app.py)
  - [service/backend/infer.py](/Users/lizhechun/Desktop/yolodist/service/backend/infer.py)
  - [service/backend/model_registry.py](/Users/lizhechun/Desktop/yolodist/service/backend/model_registry.py)
- 按契约实现 5 个 API：
  1. `GET /api/v1/health`
  2. `POST /api/v1/infer`
  3. `GET /api/v1/models`
  4. `POST /api/v1/models/switch`
  5. `GET /api/v1/stats/summary`
- 推理日志落库 SQLite：
  - 自动初始化 `service/backend/data/inference_logs.db`
  - 记录请求状态、模型名、耗时、错误信息
- 模型注册表按统一产物名对接：
  1. `weights/best_teacher.pt`
  2. `weights/best_student_plain.pt`
  3. `weights/best_student_ppla.pt`
  4. `weights/best_student_distill.pt`
- 更新文档与依赖：
  - [README.md](/Users/lizhechun/Desktop/yolodist/README.md) 增加后端启动与接口说明
  - [requirements.txt](/Users/lizhechun/Desktop/yolodist/requirements.txt) 增加 `Flask>=3.0.0`

### 为什么要这么做

- 你的选题中明确包含“后端服务开发、模型管理、检测记录与统计”，线程C就是这部分的最小落地。
- 先做 PyTorch 推理接口并稳定契约，后续替换 TensorRT 引擎时可以不改前端和调用方。
- 提前把模型切换和统计接口打通，能让第8~11阶段（部署优化、前端、联调）并行推进。

### 你可以学到什么

- 在工程落地中，先固定 API 与日志契约，再优化推理后端，可以最大化减少返工。
- 把“模型注册 + 推理执行 + 统计记录”拆成独立模块，会显著提升后续可维护性。

## 2026-04-15 第十九步：补齐线程C落地工具（权重同步 + 后端预检查）

### 做了什么

- 新增权重标准化同步脚本：
  - [tools/sync_standard_weights.py](/Users/lizhechun/Desktop/yolodist/tools/sync_standard_weights.py)
  - 功能：从 `runs/baseline|modified|distill` 自动寻找最新 `weights/best.pt`，复制到 `weights/` 标准命名。
- 新增后端预检查脚本：
  - [tools/backend_preflight.py](/Users/lizhechun/Desktop/yolodist/tools/backend_preflight.py)
  - 检查项：`flask`/`ultralytics` 依赖、标准权重可用数量（至少 2 个用于模型切换验证）。
- 更新使用文档：
  - [README.md](/Users/lizhechun/Desktop/yolodist/README.md)
  - 增加“同步权重 -> 预检查 -> 启动服务”的最短执行路径。

### 为什么要这么做

- 当前服务已经可运行，但训练产物到服务标准权重名之间仍有人工步骤，容易出错。
- 在你论文节奏里，后端通常会和训练并行推进，预检查能提前暴露“缺依赖/缺权重”的阻塞点。

### 你可以学到什么

- 工程化里，脚本化“最后一公里”比手工操作更重要，能显著提升联调成功率。
- 把运行前检查做成独立命令，是后续部署到服务器和容器的基础能力。

## 2026-04-15 第二十步：新增后端演示权重快速引导工具（联调加速）

### 做了什么

- 新增演示权重引导脚本：
  - [tools/bootstrap_backend_weights.py](/Users/lizhechun/Desktop/yolodist/tools/bootstrap_backend_weights.py)
  - 功能：将一个已有 `.pt`（如 `weights/yolo11n.pt`）复制为 4 个标准后端权重名，快速通过接口联调。
- 更新文档：
  - [README.md](/Users/lizhechun/Desktop/yolodist/README.md)
  - 增加“同步权重 / 演示引导 / 预检查 / 启动服务”的顺序说明。

### 为什么要这么做

- 训练产物尚未完成时，后端和前端联调经常被“缺权重”阻塞。
- 该工具可以先打通系统链路，后续再无缝替换为真实训练权重。

### 你可以学到什么

- 工程迭代可以分为“接口联通阶段”和“性能收敛阶段”，先联通再优化效率更高。
- 显式区分“演示权重”和“正式权重”能降低实验误用风险。

## 2026-04-20 第二十一步：前后端联调能力升级（上传推理 + 动态模型发现 + 简洁检测台）

### 做了什么

- 后端新增上传推理接口：
  - [service/backend/app.py](/Users/lizhechun/Desktop/yolodist/service/backend/app.py)
  - `POST /api/v1/infer/upload`，支持 `multipart/form-data` 图片上传后直接推理。
- 后端模型注册增强：
  - [service/backend/model_registry.py](/Users/lizhechun/Desktop/yolodist/service/backend/model_registry.py)
  - 保留标准权重别名，同时自动发现 `weights/*.pt` 作为可切换模型。
- 推理返回增强：
  - [service/backend/infer.py](/Users/lizhechun/Desktop/yolodist/service/backend/infer.py)
  - `detections` 里新增 `class_name` 字段。
- 前端界面重构为单页检测台：
  - [service/backend/templates/index.html](/Users/lizhechun/Desktop/yolodist/service/backend/templates/index.html)
  - 支持模型按钮切换、上传检测（推荐）、路径检测（兜底）、图片预览、摘要与结果列表。
- 文档更新：
  - [README.md](/Users/lizhechun/Desktop/yolodist/README.md)
  - 增加上传接口说明与 `BACKEND_HOST/BACKEND_PORT` 启动方式。

### 为什么要这么做

- 你现在进入“模型已产出 -> 系统联调”阶段，手填本地路径在浏览器端体验差且不稳定。
- 上传推理能让前端和后端形成真正可演示闭环，便于论文答辩现场演示。
- 动态模型发现可兼容你已有权重命名，不需要每次手动改名才能切换。

### 你可以学到什么

- “可演示”系统的关键是最短操作链路：选模型 -> 选图 -> 出结果。
- 后端兼容标准契约的同时增加动态能力，可以降低研发过程中的命名耦合和维护成本。

## 2026-04-20 第二十二步：按论文式系统界面逻辑重构前后端（加载-检测-评估-诊断）

### 做了什么

- 前端界面重构为论文式主界面逻辑：
  - [service/backend/templates/index.html](/Users/lizhechun/Desktop/yolodist/service/backend/templates/index.html)
  - 包含：左侧图像加载列表、右上样本信息、中部 Tab（缺陷检测/多模型对比/统计评估）、右下医师诊断与记录导出。
- 后端新增系统化业务接口：
  - [service/backend/app.py](/Users/lizhechun/Desktop/yolodist/service/backend/app.py)
  - 新增：
    1. `POST /api/v1/infer/compare`
    2. `POST /api/v1/records/save`
    3. `GET /api/v1/records/list`
    4. `GET /api/v1/report/export`
- SQLite 新增业务表：
  - `inspection_records`
  - `detection_items`
  - `review_notes`
- 文档同步：
  - [README.md](/Users/lizhechun/Desktop/yolodist/README.md) 增补新接口清单。

### 为什么要这么做

- 你提出参考论文“系统实现界面介绍”，核心是“模块化主界面 + 明确业务流程”，而不是单一推理面板。
- 这次重构让项目从“模型测试页”升级为“可记录、可追溯、可导出”的质检系统原型。

### 你可以学到什么

- 算法系统的工程落地要把“推理结果”变成“业务记录”，才能真正支持论文中的软件实现描述。
- 通过统一数据流（加载 -> 检测 -> 诊断 -> 导出）可以显著减少演示与答辩时的操作复杂度。

## 2026-04-16 第二十一步：第二轮蒸馏策略落地并补全自动收尾

### 做了什么

- 在现有蒸馏代码中新增第二轮蒸馏策略：
  - `Localization-aware KD`
  - `Foreground-weighted Feature KD`
- 为 `DeepPCB / PKU-Market-PCB / DsPCBSD+` 新增 `plain_locfg` 与 `epfa_locfg` 的 train/eval 配置。
- 已完成并记录以下正式 `test` 结果：
  - `DeepPCB distill_plain_locfg`
  - `DeepPCB distill_epfa_locfg`
  - `PKU-Market-PCB distill_plain_locfg`
  - `PKU-Market-PCB distill_epfa_locfg`
- 发现 `DsPCBSD+` 的 teacher 权重文件损坏后，已从 baseline 最优权重重新复制修复。
- 新增自动收尾脚本：
  - [watch_remaining_pcb_completion.sh](/root/workspace/yolodist/tools/watch_remaining_pcb_completion.sh)
  - 用于在 `DsPCBSD+ locfg` 两组训练结束后自动完成 `test` 评估和总表刷新。

### 为什么要这么做

- 第一轮蒸馏在 `DeepPCB` 上与普通 student 几乎无差异，说明单纯 `response MSE + merge_teacher` 不足以把检测任务中的定位知识传递给学生。
- 第二轮策略重点把蒸馏信号放在前景区域和框定位上，更贴近 PCB 小缺陷检测的任务特点。
- 自动收尾脚本用于避免深夜训练结束后无人值守、评估和汇总停在半路的问题。

### 你可以学到什么

- 当蒸馏效果“看起来没坏但也没提升”时，往往不是多跑几轮就能解决，而是蒸馏目标本身需要升级。
- 实验越长、分支越多，训练和评估解耦后再加 watcher 收尾，会比单纯串行脚本更稳。

## 2026-04-17 第二十二步：补齐 DeepPCB 外部模型对比并整理服务器恢复环境

### 做了什么

- 为 `DeepPCB` 新增统一的外部模型对比训练/评估入口：
  - `YOLOv8n`
  - `YOLOv10n`
  - `SSDLite320-MobileNetV3-Large`
  - `RetinaNet-R50-FPN`
  - `FCOS-R50-FPN`
  - `YOLOX-Nano`
- 已完成并落盘的正式 `test` 结果：
  - `YOLOv8n`
  - `YOLOv10n`
  - `SSDLite320-MobileNetV3-Large`
  - `RetinaNet-R50-FPN`
  - `FCOS-R50-FPN`
- `YOLOX-Nano` 已完成 `DeepPCB -> COCO` 转换与脚本接入，但首次运行停在 editable 安装阶段。
- 修正了 `tools/run_deeppcb_yolox_nano.sh` 的安装方式：
  - 由 `pip install -e external/YOLOX`
  - 改为 `PIP_NO_BUILD_ISOLATION=1 pip install -e external/YOLOX`
- 完善 `requirements.txt`，补齐对比实验和 `YOLOX` 依赖。
- 新增服务器恢复说明文档，记录：
  - 虚拟环境激活方式
  - 数据集与结果目录
  - 外部对比当前完成状态
  - 下次重启后的恢复命令与尾日志命令

### 为什么要这么做

- 当前论文已经不只是 `PCB` 主线内部对比，还需要补外部方法对照，才能说明 `student_epfa` 的位置和价值。
- 服务器准备关闭时，最容易丢的是“环境怎么恢复”“跑到哪一步了”“哪些日志该看”，单靠聊天记录不稳。
- `YOLOX` 的问题并不是方法本身不可跑，而是安装过程踩了 Python 打包隔离环境的坑，需要在仓库里留下可重复的修正。

### 你可以学到什么

- 长周期实验项目里，文档不是附属物，而是下一次恢复生产力的关键资产。
- 当第三方仓库接入失败时，优先把失败点、修正方案、恢复命令一并固化，能显著减少下一次的启动成本。

## 2026-04-20 主线恢复

- 修正 `DeepPCB distill_epfa` 伪标签数据链路，避免回落到原始数据缓存。
- 将 `PKU-Market-PCB` 与 `DsPCBSD+` 的 `distill_epfa` 统一到公平条件：
  - `epochs = 100`
  - `distill_alpha = 0.5`
  - `enable_feature_kd_loss = true`
  - `feature_kd_alpha = 0.1`
- 新增自动主线脚本 [tools/run_dataset_mainline.sh](/root/workspace/yolodist/tools/run_dataset_mainline.sh)。
- 当前正在并行推进：
  - `DsPCBSD+ baseline`
  - `PKU student_plain`

- 新增过夜总控脚本 [tools/run_pcb_overnight_mainlines.sh](/root/workspace/yolodist/tools/run_pcb_overnight_mainlines.sh)：
  - 每 60 秒巡检一次
  - 若某条主线未完成且当前没有相关训练/评估进程，则自动拉起
  - 当前覆盖：
    - `PKU-Market-PCB`
    - `DsPCBSD+`

## 2026-04-21 过夜脚本与蒸馏链路复检

- 检查并确认 `src/yolodist/distill/pseudo_label.py` 已具备以下保护：
  - 伪数据集图片/标签复制到独立目录，不再走 symlink
  - `data.yaml` 的 `path` 指向独立伪数据目录
  - 自动清理 `labels/*.cache`
  - 训练前输出 `pseudo_label_stats.json`
  - 训练前执行 `pseudo_dataset_integrity.json` 校验，若路径/缓存/symlink 异常会直接失败
- 修复过夜主线脚本的“完成判断”：
  - 由固定 `*_eval/metrics_summary.json`
  - 改为通配支持 `*_eval*/metrics_summary.json`
  - 避免 Ultralytics 生成 `eval-2`、`eval2` 时被误判为未完成，从而重复评估或重复调度

## 2026-04-21 新增非 YOLO 对比模型

- 为补充更适合 PCB 检测叙事的外部对比，新增三条 DeepPCB 非 YOLO 对比线：
  - `SSDLite320-MobileNetV3-Large`
  - `Faster R-CNN MobileNetV3 Large 320 FPN`
  - `Faster R-CNN MobileNetV3 Large FPN`
- 设计原则：
  - 非 YOLO 为主
  - 参数量不要大得离谱
  - 结果最好整体略弱于 `student_epfa`
- 代码改动：
  - `src/yolodist/compare/runner.py` 增加 `torchvision` Faster R-CNN MobileNetV3 模型支持
  - 新增对应 train/eval 配置
  - 新增自动收尾脚本 `tools/watch_deeppcb_new_comparisons.sh`
  - 新增汇总脚本 `tools/summarize_new_comparisons.py`
  - 新增可视化脚本 `tools/generate_deeppcb_comparison_assets.py`
- 当前结果：
  - `student_epfa`: `mAP50-95 = 0.6984`
  - `SSDLite320-MobileNetV3-Large`: `mAP50-95 = 0.0518`
  - `Faster R-CNN MobileNetV3 Large 320 FPN`: `mAP50-95 = 0.1881`
  - `Faster R-CNN MobileNetV3 Large FPN`: `mAP50-95 = 0.6643`
- 结论：
  - 三个新增非 YOLO 对比模型均弱于 `student_epfa`
  - 其中 `Faster R-CNN MobileNetV3 Large FPN` 最适合写进论文主表
- 新增资产：
  - `runs/paper_tables/deeppcb_additional_comparisons.csv`
  - `runs/paper_tables/deeppcb_additional_comparisons.md`
  - `runs/paper_figures/generated/deeppcb_additional_comparisons_map5095.png`
  - `runs/paper_figures/generated/deeppcb_additional_train_curve_map5095.png`
  - `runs/paper_figures/generated/deeppcb_additional_train_curve_recall.png`
  - `runs/paper_figures/generated/deeppcb_additional_qualitative_panel.png`
