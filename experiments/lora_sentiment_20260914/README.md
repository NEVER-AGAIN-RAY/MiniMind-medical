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

本地无 GPU、无基座权重（`out/` 不存在），训练与评测**只在云端执行**；本地只做数据准备
与无权重的纯逻辑验证。

```bash
python experiments/lora_sentiment_20260914/train_sentiment_lora.py --smoke   # 40/20 冒烟
python experiments/lora_sentiment_20260914/train_sentiment_lora.py           # 正式训练
python experiments/lora_sentiment_20260914/eval_sentiment.py --split smoke
python experiments/lora_sentiment_20260914/eval_sentiment.py --split formal
```

超参与路径全部集中在 [`config.json`](config.json)；训练产物写入 `runs/{smoke,formal}/`
（`allow_overwrite: false`，不会覆盖已有结果）。

## 当前状态

- ✅ 数据准备已完成并核验（切分互斥、类别平衡、md5 一致、重跑逐字节可复现）
- ⬜ 云端冒烟与正式训练
- ⬜ 评测与规模曲线
- ⬜ `run_cloud.sh` 一键流程（参照 [医学实验的版本](../lora_medical_20260909/run_cloud.sh)）
- ⬜ 纯逻辑单元测试（参照 [`tests/test_eval_logic.py`](../../tests/test_eval_logic.py)）
