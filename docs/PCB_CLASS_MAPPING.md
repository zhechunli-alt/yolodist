# PCB 类别映射

这份文档记录三个 PCB 数据集接入 yolodist 时采用的统一类别名，以及遇到原始标注命名不一致时的 alias 规则。

## 统一原则

- `DeepPCB` 和 `PKU-Market-PCB` 统一到 6 类：
  - `open`
  - `short`
  - `mousebite`
  - `spur`
  - `spurious_copper`
  - `missing_hole`
- `DsPCBSD+` 保持 9 类原任务语义，避免强行并类导致论文解释变差。

## DeepPCB

- 官方 id 映射：
  - `1 -> open`
  - `2 -> short`
  - `3 -> mousebite`
  - `4 -> spur`
  - `5 -> spurious_copper`
  - `6 -> missing_hole`
- 常见 alias：
  - `open_circuit -> open`
  - `short_circuit -> short`
  - `mouse_bite -> mousebite`
  - `pin-hole -> missing_hole`
  - `pin_hole -> missing_hole`
  - `copper -> spurious_copper`

## PKU-Market-PCB

- 采用与 DeepPCB 对齐的 6 类：
  - `open`
  - `short`
  - `mousebite`
  - `spur`
  - `spurious_copper`
  - `missing_hole`
- 常见 alias：
  - `open_circuit -> open`
  - `short_circuit -> short`
  - `mouse_bite -> mousebite`
  - `pin-hole -> missing_hole`
  - `pin_hole -> missing_hole`
  - `copper -> spurious_copper`

## DsPCBSD+

- 当前默认 9 类：
  - `short`
  - `spur`
  - `spurious_copper`
  - `open`
  - `mousebite`
  - `hole_breakout`
  - `conductor_scratch`
  - `conductor_foreign_object`
  - `base_material_foreign_object`
- 常见 alias：
  - `SH -> short`
  - `SP -> spur`
  - `SC -> spurious_copper`
  - `OP -> open`
  - `MB -> mousebite`
  - `HB -> hole_breakout`
  - `CS -> conductor_scratch`
  - `CFO -> conductor_foreign_object`
  - `BMFO -> base_material_foreign_object`

## 日志位置

- 每次 `prepare_pcb_detection.py` 运行后，alias 映射日志会写进对应输出目录下的 `manifest.json`。
- 字段名：`alias_log`
