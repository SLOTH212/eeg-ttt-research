# EEG TTT Research

连续 EEG 睡眠分期中的测试时更新效用、保留与回退，以及匹配的静态因果历史对照。

**状态：2026-10-06 起因磁盘空间不足暂停研究。此次公开发布只整理代码，不恢复实验。**

本仓库保存阶段 1–138 的研究代码与汇总证据。模型权重、原始及处理后 EEG、逐窗预测、虚拟环境和运行缓存没有上传；这不是一键复现全部结果的发行包。

## 当前结论

在已观察的 SleepEDF 30 人、三个种子、EMA=0.5 条件下：

| 载体 | 等容量重复当前头 | 静态历史头 | Macro-F1 差值（百分点，95%配对区间） |
|---|---:|---:|---:|
| CBraMod | 59.97 | 63.42 | +3.45 [2.18, 4.71] |
| LaBraM | 59.88 | 62.12 | +2.25 [1.57, 2.93] |
| EEGNet-style | 59.88 | 62.72 | +2.84 [1.84, 3.83] |

- 静态历史收益不等于 TTT 收益：历史头额外使用源标签训练，测试时参数冻结。
- 外部 ISRUC 收益不稳定；EEGNet 历史头相对重复头在健康/患者组分别下降 3.24/1.48pp。
- 梯度更新尚未胜过强静态对照，可观察更新效用与有效保留控制尚未成立。
- 先在人内平均种子指标，再做受试者配对统计；窗口和种子不是独立受试者。
- 目标队列已被反复观察，结果属于探索性证据；跨架构训练预算不同，不是预训练的因果消融。

完整说明：[阶段研究报告](reports/research_report_20261006.html) · [聚合证据](reports/evidence)。
GitHub 文件页可查看 HTML 源码；下载后用浏览器阅读。

## 仓库结构

| 路径 | 内容 |
|---|---|
| src/ | 452 个研究脚本，保留原阶段名称和模块依赖 |
| configs/pilot.json | 最初 CPU pilot 原始配置，不代表所有后续 GPU 实验 |
| docs/code_index.html | 按阶段分组的源码及本地模块依赖索引 |
| docs/reproduction.html | 数据/检查点依赖、阶段链与复现边界 |
| docs/protocols/ | 选定阶段的只读协议快照，原路径替换为占位符 |
| docs/source_manifest.json | 全部源码原始/发布 SHA256 与发布修改记录 |
| docs/third_party.html | 外部数据、代码与权重来源 |
| reports/ | 当前报告和小型聚合结果，无逐窗数据 |
| tools/check_repository.py | 无依赖的语法、完整性及发布边界校验 |
| tools/doctor.py | 只读环境和必要本地资源检查 |

## 从哪里读代码

| 问题 | 主要入口（均位于 src/） |
|---|---|
| 原始模型、固定五类指标 | common.py |
| 输入幅度/形状单窗状态 | phase102_stream.py |
| 静态残差头与因果历史 | phase119_models.py、phase127_stream.py |
| 三载体历史训练 | phase119_logit_history.py、phase121_labram_history.py、phase135_eegnet_history.py |
| 固定目标评价 | phase120_history_targets.py、phase123_labram_targets.py、phase136_eegnet_targets.py |
| 教师 WRITE / 历史分类器熵 WRITE | phase114_teacher_replication.py、phase126_history_entropy.py |
| 源选模敏感性 | phase130_selector_swap.py、phase131_selector_targets.py |
| 完整单窗工程审计 | phase133_waveform_stream.py、phase138_eegnet_stream.py |
| 同人群三模型比较 | phase137_carrier_comparison.py |

完整映射见 [代码索引](docs/code_index.html)。部分脚本会在导入时加载资源或执行工作，不要批量 import 所有实验脚本。

## 不运行实验的检查

    python3 tools/check_repository.py
    python3 tools/doctor.py

两个命令仅用 Python 标准库，不下载数据、不导入实验模块、不训练模型。
仓库没有自动实验或下载工作流；上述检查仅在显式调用时运行。

## 环境与复现

原环境为 Linux / Python 3.11.15，已记录版本见 [requirements.txt](requirements.txt) 和 [上游附加依赖](requirements-upstream.txt)。
Torch 原运行版本为 2.8.0+cu128；CPU/CUDA 发行包需与所选硬件匹配。版本记录不是跨硬件数值复现保证。

可在有足够空间的新环境中安装依赖；依赖安装不包含数据、上游模型仓库或检查点。
请先阅读 [复现说明](docs/reproduction.html)，当前项目仍暂停。

原运行约束：

- CPU 仅允许核心 8–23；重现原配置需在外层使用 taskset -c 8-23。
- OPENBLAS_NUM_THREADS=1；后期 EEG_THREADS=2、OMP_NUM_THREADS=2。
- common.py 的早期路径默认禁用 CUDA；明确的 GPU 脚本会自行启用 CUDA，不能将全项目视为 CPU-only。
- 没有相应核心编号的机器需要显式调整资源配置；此发布保留原 affinity 规则。

发布只改动18个文件：17个脚本的固定工作目录改为相对源码位置解析，另一个报告生成器去掉私人附件路径。
算法、数据拆分、超参数和 common.py 保持原样；原实验目录未修改。
历史锁可能包含旧源码哈希，不能把发布副本冒充逐位相同的原始代码，也不应覆盖旧锁绕过审计。

## 来源与许可

数据来源为 Sleep-EDF Expanded 与 ISRUC-Sleep；外部代码包括 CBraMod、LaBraM。
来源和锁定 revision 见 [第三方登记](docs/third_party.html)。
本仓库不再分发数据、上游代码或模型权重，不为它们授予新许可。
本研究代码快照暂未指定开源许可证；公开可见不替代明确的使用许可。
