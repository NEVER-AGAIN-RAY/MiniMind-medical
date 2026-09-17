# 情感分类 LoRA 实验（lora_sentiment_20260914）

在 MiniMind 64M 基座上用 LoRA 微调中文酒店评论情感二分类（ChnSentiCorp）。

## 为什么做这个实验

前一个实验 [`lora_medical_mcq_pilot`](../lora_medical_mcq_pilot/) 在医学多选题上失败：
64M 模型的生成式 MCQ 回答约 90% 无法解析，无从判断 LoRA 到底学没学到东西。
事后归因是**任务类型选错了**——MCQ 属于知识型任务，答案不在输入里，要求模型从参数中
召回它本来就没有的医学知识。

情感分类是对照组：判断所需的全部信息都在评论正文内，属于**映射型任务**。如果 LoRA 在
这个量级的模型上有效，应该先在映射型任务上表现出来。这个实验要回答的就是这一个问题。

## 数据流水线

```bash
python experiments/lora_sentiment_20260914/prepare_data.py            # 首次生成
python experiments/lora_sentiment_20260914/prepare_data.py --overwrite  # 重新生成
```

[`prepare_data.py`](prepare_data.py) 把冻结的原始 CSV 加工成切分，每一步都在挡一个具体的坑：

| 步骤 | 挡掉的问题 |
| --- | --- |
| 用 `csv` 模块解析 | 评论正文含逗号与引号，按逗号 split 会把评论切碎 |
| 空评论 / 过短过滤 | 无信息量样本 |
| 精确去重（归一化文本） | 同一条评论重复入库 |
| Token 长度过滤（真实 tokenizer 实测，**筛除而非截断**） | 截断会砍掉评论尾部的转折（"但是房间有异味"），标签与输入对不上 |
| 近重复去重（字符 bigram Jaccard ≥ 0.85） | 模板化评论导致训练集与测试集内容重叠，测出来的准确率虚高 |
| 平衡切分（正负各半） | 原始数据正面占 68.5%，不平衡则无脑全答"正面"就有 68.5%，分不清模型是学会了还是在偷懒 |
| 先取 test/val 再取 train | 评测集优先锁定，日后调整训练规模不会动到评测集，前后结果可比 |
| 跨切分泄漏校验 | 与其等结果好得离谱再回头查，不如当场抛异常 |

生成产物（均已入库，作为本实验的冻结输入）：

| 路径 | 内容 |
| --- | --- |
| `data/formal/{train,val,test}.jsonl` | 2000 / 400 / 400 条，正负各半 |
| `data/smoke/{train,val}.jsonl` | 40 / 20 条，云端流程冒烟 |
| `data/scale/train_{250,500,1000,2000}.jsonl` | 嵌套前缀子集，画"数据量 → 准确率"曲线 |
| `data/manifest.json` | 各文件 md5、过滤统计、生效配置、基线与判定阈值 |

### 语料容量上限

去重与长度过滤后可用池为负面 2266 / 正面 5178，**负面是瓶颈**。当前切分每类需
(2000+400+400+40+20)/2 = 1430 条，负面余量 836 条。若保持其余切分不变，平衡训练集
最多只能扩到 (2266-430)×2 = **3672 条**。想要更大的训练集，必须同时缩小评测集或放宽
`max_seq_len`（后者会让被筛除的 317 条超长评论回到池中）。

## 怎么读评测结果

平衡切分让**固定预测多数类**的准确率严格等于 0.5。**随机预测**的期望也是 0.5，但单次
实测服从 Binomial(n, 0.5)/n，会在抽样误差内波动——所以不要拿结果直接跟 0.5 比：

| 评测集 | n | 随机基线 95% 区间 | 判定"优于随机"所需准确率 |
| --- | --- | --- | --- |
| `formal_test` | 400 | [0.451, 0.549] | **> 0.549** |
| `formal_val` | 400 | [0.451, 0.549] | **> 0.549** |
| `smoke_val` | 20 | [0.281, 0.719] | > 0.719 |

冒烟集只有 20 条，区间宽到准确率数字基本不可解读——它只用于验证流程能跑通，不要拿它
的准确率下任何结论。这些阈值由 `prepare_data.py` 计算并写入 `data/manifest.json` 的
`baselines.per_split`。

除准确率外还需关注 `predicted_label_distribution`：医学实验中字母分布偏向 B 就是模型
塌缩到单一类别的信号，在这里对应"全答正面"。

## 训练与评测

本地无 GPU，训练与正式评测**只在云端执行**；本地只做数据准备与无权重的纯逻辑验证。

云端一条命令跑完全流程：

```bash
bash experiments/lora_sentiment_20260914/run_cloud.sh
```

[`run_cloud.sh`](run_cloud.sh) 的六个步骤：预检 → 冒烟 → 正式训练 → 正式评测 →
规模曲线 → 汇总报告。已完成的步骤依据产物自动跳过，`--force` 强制重跑，
`--step N` / `--from-step N` / `--skip-smoke` / `--skip-scale` 控制范围。
全部终端输出经 `tee` 落盘到 `logs/run_<时间戳>.log`（`latest.log` 指向最近一次）；
依赖快照只在与上一份不同时才落盘，避免堆积一堆逐字节相同的 `pip_freeze` 文件。

**预检会拒绝占位权重**：若 `out/PLACEHOLDER_README.txt` 存在（说明基座是本地干跑用的
随机初始化权重），脚本直接退出，除非显式加 `--allow-placeholder`。随机权重跑出来的
准确率毫无意义，这道闸门防止它混进正式结果。

也可以单独调用各脚本：

```bash
python experiments/lora_sentiment_20260914/train_sentiment_lora.py --smoke   # 40/20 冒烟
python experiments/lora_sentiment_20260914/train_sentiment_lora.py           # 正式训练
python experiments/lora_sentiment_20260914/train_sentiment_lora.py \
    --train-file experiments/lora_sentiment_20260914/data/scale/train_500.jsonl  # 规模曲线某点
python experiments/lora_sentiment_20260914/eval_sentiment.py --split formal \
    --checkpoint experiments/lora_sentiment_20260914/runs/formal/best_lora.pth
python experiments/lora_sentiment_20260914/make_report.py                    # 汇总 results.md
```

`--train-file` 的验证集固定为 `formal/val.jsonl`，否则规模曲线各点的 val_loss 无法横向比较。

超参与路径全部集中在 [`config.json`](config.json)；训练产物写入 `runs/`
（`allow_overwrite: false`，不会覆盖已有结果）。

## 测试

```bash
python -m pytest tests/test_sentiment_logic.py -q
```

37 个纯逻辑用例，不加载任何模型权重（只用分词器与几个元素的桩张量），2 核 CPU 上约 10 秒：
标签归一化、prompt/target 构建、数据集的监督掩码与「超长抛错而非截断」、生成结果解析
（含两个标签同时出现时判无效）、`candidate_score` 的切片位置、基线阈值换算、报告的
阈值判定与单类别塌缩检测，以及已入库切分本身的不变量（平衡、互斥、长度预算、嵌套前缀）。

## 当前状态

- ✅ 数据准备已完成并核验（切分互斥、类别平衡、md5 一致、重跑逐字节可复现）
- ✅ 纯逻辑单元测试（37 例全通过）
- ✅ `run_cloud.sh` 六步流程 + `make_report.py` 汇总
- ✅ 本地占位权重干跑：训练、评测、报告三条链路端到端跑通
- ✅ 云端全流程已跑完（RTX 3080 Ti，约 4.5 分钟，见 [results.md](results.md)）
- ✅ 正式评测 **准确率 0.8400**，远超判定阈值 0.549；规模曲线单调上升
- ✅ 结论：**LoRA 在映射型任务上有效**，与医学 MCQ 的失败形成对照

## 结论

| 训练条数 | 准确率 | val_loss |
| --- | --- | --- |
| 250 | 0.5600 | 0.2057 |
| 500 | 0.6675 | 0.1830 |
| 1000 | 0.7725 | 0.1309 |
| 2000 | **0.8400** | 0.1044 |

判定阈值 0.549（n=400）。250 条时准确率 0.56 仅勉强越过阈值，说明这个量级的模型
需要约 500 条以上才谈得上稳定学到东西；曲线到 2000 条仍在上升，尚未饱和——受限于
负面语料池，平衡训练集最多还能扩到 3672 条。

支撑"确实学会了"而非"碰巧蒙对"的三个旁证：
- **格式合规率 100%**：400 题全部输出了可解析的标签，对比医学 MCQ 实验约 90% 无法解析
- **预测分布 198/202**：没有塌缩到单一类别，两类 F1 分别为 0.8392 / 0.8408
- **候选打分与生成式准确率几乎一致**（0.8400 / 0.8375）：模型的判断与它实际吐出的字一致

这印证了立项时的假设：失败的原因不是 LoRA 或 64M 模型不行，而是医学 MCQ 属于知识型
任务——答案不在输入里。情感分类的判断依据全在评论正文内，同样的模型、同样的方法就有效。
