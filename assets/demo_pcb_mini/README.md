# DeepPCB Mini Demo Set

这是一套用于前后端演示和论文截图的小型 PCB 测试集，来源于项目内已经处理好的 `DeepPCB` 测试集。

目录结构：

- `images/`：6 张演示图片
- `labels/`：对应的 YOLO 标签
- `manifest.json`：样本摘要与类别统计

类别顺序：

1. `open`
2. `short`
3. `mousebite`
4. `spur`
5. `spurious_copper`
6. `missing_hole`

这套样本的设计目标：

- 体积小，适合直接放进仓库
- 缺陷类别覆盖完整
- 可直接用于 Web 演示、截图和答辩演示

建议使用方式：

1. 在 Web 演示系统的 `数据加载` 页选择单张图片或直接加载整个 `images/` 文件夹
2. 使用默认模型 `DeepPCB / Student-EPFA` 做单模型检测
3. 在 `多模型对比` 页使用主线模型做结果对照

说明：

- 这是演示集，不是论文正式评测集
- 正式结果仍以 `datasets/processed/deeppcb_detection/` 下的完整测试集为准
