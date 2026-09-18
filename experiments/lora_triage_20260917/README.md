# 医疗科室分诊 LoRA 实验（lora_triage_20260917）

> 📄 **完整实验报告：[REPORT.md](REPORT.md)** —— 结果、结论与两次误诊的复盘。
> 本文件是实验设计与操作说明。

在 MiniMind 64M 基座上用 LoRA 微调中文医疗科室分诊（六分类）：输入患者自述，输出应挂的科室。

## 为什么做这个实验

这是三个实验里的第三个，用来补上前两个之间缺的那一环。

| 实验 | 任务类型 | 结果 |
| --- | --- | --- |
| [`lora_medical_mcq_pilot`](../lora_medical_mcq_pilot/) | 知识型（医学单选题） | ❌ 失败，六轮均未显著超过随机 |
| [`lora_sentiment_20260914`](../lora_sentiment_20260914/) | 映射型（酒店评论情感） | ✅ 0.8575，远超随机基线 |
| **本实验** | **映射型（医学）** | ✅ **0.8333** |

情感实验证明了「LoRA 在 64M 上对映射型任务有效」，但它用的是酒店评论——把医学 MCQ 的
失败归因于**任务类型**而非领域，这个推论跨了领域，没有被直接验证过。本实验把任务类型
换成映射型、领域留在医学，检验那条归因是否成立：

- **若有效** → 医学方向并非走不通，走不通的是知识召回类任务，分诊/抽取类任务可行；
- **若无效** → 归因需要修正，问题可能出在医学文本本身（术语密度、基座预训练覆盖），
  而不只是任务类型。

分诊属于映射型：判断所需的线索（症状、部位、性别与年龄暗示）全部在患者自述之内，
模型不需要具备"某病属于某科"之外的医学知识。

### 第二个可回答的问题：饱和点到底来自哪里

情感实验在 2000 条处饱和，但它的语料上限恰好是 3672 条——「模型容量饱和」与「语料耗尽」
纠缠在一起。本实验语料有 **79 万条**，训练集推到 12000 条也只用掉零头，因此如果同样出现
平台期，就只能是模型容量造成的。

## 数据

原始语料：[Chinese-medical-dialogue-data](https://github.com/Toyhom/Chinese-medical-dialogue-data)，
六个科室目录各一个 GB18030 编码的 CSV，共约 370MB，**不入库**——
按 [`data/raw/MANIFEST.json`](data/raw/MANIFEST.json) 的 URL 与 sha256 重新下载即可：

```bash
python experiments/lora_triage_20260917/fetch_raw.py    # 下载并逐个校验 sha256
```

标签取自**目录名**（顶层科室）；CSV 内的 `department` 列是更细的子科室（内科目录下有
58 个子科室，如"心血管科"），不作为标签。只使用 `ask` 列（患者自述），
`title` 与 `answer` 不进入输入。

### 流水线

```bash
python experiments/lora_triage_20260917/prepare_data.py              # 首次生成
python experiments/lora_triage_20260917/prepare_data.py --overwrite  # 重新生成
```

[`prepare_data.py`](prepare_data.py) 每一步都在挡一个具体的坑：

| 步骤 | 挡掉的问题 |
| --- | --- |
| 用 `csv` 模块解析 + GB18030 解码 | 自述含逗号与引号；编码猜错会整列变乱码 |
| **跨科室矛盾剔除**（全语料） | 同一段自述同时挂在两个科室下，标签自相矛盾，两边都丢（实测 1425 段） |
| 精确去重（归一化文本） | 同一段自述重复入库 |
| 空 / 过短过滤 | 无信息量样本 |
| 按类蓄水池抽样（每类 5000） | 79 万条全量载入会撑爆 4GB 内存的机器；固定种子下抽样完全可复现 |
| Token 长度过滤（真实 tokenizer 实测，**筛除而非截断**） | 截断会砍掉自述尾部的关键症状，标签与输入对不上 |
| 近重复去重（字符 bigram Jaccard ≥ 0.85，**类内**） | 模板化自述导致训练集与测试集内容重叠，测出来的准确率虚高 |
| 六类平衡切分 | 不平衡则无脑全答多数类就有高分，分不清模型是学会了还是在偷懒 |
| 先取 test/val 再取 train | 评测集优先锁定，日后调整训练规模不会动到评测集，前后结果可比 |
| 跨切分泄漏校验（id 与文本双重） | 与其等结果好得离谱再回头查，不如当场抛异常 |

生成产物（均已入库，作为本实验的冻结输入）：

| 路径 | 内容 |
| --- | --- |
| `data/formal/{train,val,test}.jsonl` | 12000 / 600 / 600 条，六类等量 |
| `data/smoke/{train,val}.jsonl` | 60 / 30 条，云端流程冒烟 |
| `data/scale/train_{600,1200,2400,4800,9600}.jsonl` | 嵌套前缀子集，画"数据量 → 准确率"曲线 |
| `data/manifest.json` | 各文件 md5、过滤统计、生效配置、基线与判定阈值 |

实测过滤量（种子 42）：全语料 792099 条 → 609161 段不同自述 → 剔除 1425 段跨科室矛盾 →
每类抽样 5000 → 长度过滤丢 103 条 → 近重复去重丢 145 条 → 每类可用池约 4950 条。

## 怎么读评测结果

六类平衡让**固定预测多数类**的准确率严格等于 1/6。**随机预测**的期望也是 1/6，但单次
实测服从 Binomial(n, 1/6)/n，会在抽样误差内波动——所以不要拿结果直接跟 0.1667 比：

| 评测集 | n | 随机基线 95% 区间 | 判定"优于随机"所需准确率 |
| --- | --- | --- | --- |
| `formal_test` | 600 | [0.1369, 0.1965] | **> 0.1965** |
| `formal_val` | 600 | [0.1369, 0.1965] | **> 0.1965** |
| `smoke_val` | 30 | [0.0334, 0.2999] | > 0.2999 |

冒烟集只有 30 条，区间宽到准确率数字基本不可解读——它只用于验证流程能跑通。
这些阈值由 `prepare_data.py` 计算并写入 `data/manifest.json` 的 `baselines.per_split`。

除准确率外还需关注 `predicted_label_distribution`：医学 MCQ 实验中字母分布偏向 B 就是
模型塌缩到单一类别的信号，在这里对应"全答内科"。

### 候选打分的长度偏置（本实验特有）

六个科室名的 token 数**不等**：内科/外科/儿科/男科各 4 个，妇产科 6 个，肿瘤科 7 个。
候选打分若直接比较各标签的 log-prob **之和**，长标签每多一个 token 就多乘一个小于 1 的
概率，会被系统性压低。情感实验没有这个问题（正面/负面等长），照搬会出错。

因此评测同时给出两个口径，**主指标取平均**：

| 指标 | 含义 |
| --- | --- |
| `candidate_scoring_accuracy_mean` | 每 token 平均 log-prob，消除长度偏置，**主指标** |
| `candidate_scoring_accuracy_sum` | log-prob 之和，与情感实验同口径，仅作对照 |
| `scoring_agreement` | 两个口径给出同一答案的题目占比 |

两者的差异本身是诊断信号：若 `sum` 口径明显偏向短标签（内科/外科/儿科/男科），
说明长度偏置确实在起作用。`tests/test_triage_logic.py` 用桩模型把这条性质固定成了断言。

## 已知局限

1. **标签来自论坛版块，存在不可消除的噪声。** 一例腹痛既可能挂内科也可能挂外科，
   这类样本的"正确答案"本身就不唯一。因此准确率天花板低于 100%，且上限未知——
   混淆矩阵能显示具体是哪几对科室在互相混淆。
2. **近重复去重只在类内做。** 跨类近重复（同一段自述的变体分属两个科室）是标签噪声
   而非泄漏——它压低准确率而不是抬高，因此不影响"是否显著优于随机"的结论。
   全局两两比较在 3 万条量级上不可行；exact 去重与矛盾剔除仍是全局的。

## 训练与评测

本地无 GPU，训练与正式评测**只在云端执行**；本地只做数据准备与无权重的纯逻辑验证。

云端一条命令跑完全流程：

```bash
bash experiments/lora_triage_20260917/run_cloud.sh
```

六个步骤：预检 → 冒烟 → 正式训练 → 正式评测 → 规模曲线 → 饱和分析。
已完成的步骤依据产物自动跳过，`--force` 强制重跑，
`--step N` / `--from-step N` / `--skip-smoke` / `--skip-scale` 控制范围。
**预检会拒绝占位权重**：若 `out/PLACEHOLDER_README.txt` 存在，脚本直接退出，
除非显式加 `--allow-placeholder`。

也可以单独调用各脚本：

```bash
python experiments/lora_triage_20260917/train_triage_lora.py --smoke     # 60/30 冒烟
python experiments/lora_triage_20260917/train_triage_lora.py             # 正式训练
python experiments/lora_triage_20260917/train_triage_lora.py \
    --train-file experiments/lora_triage_20260917/data/scale/train_2400.jsonl
python experiments/lora_triage_20260917/eval_triage.py --split formal \
    --checkpoint experiments/lora_triage_20260917/runs/formal/best_lora.pth
python experiments/lora_triage_20260917/analyze_scale.py                 # 生成 saturation.md
```

`--train-file` 的验证集固定为 `formal/val.jsonl`，否则规模曲线各点的 val_loss 无法横向比较。
`--rank` 可覆盖 LoRA rank；评测端的 rank 从 checkpoint 的 A 矩阵形状自动推断，不会错配。

超参与路径全部集中在 [`config.json`](config.json)；训练产物写入 `runs/`
（`allow_overwrite: false`，不会覆盖已有结果）。

## 测试

```bash
python -m pytest tests/test_triage_logic.py -q
```

44 个纯逻辑用例，不加载任何模型权重（只用分词器与几个元素的桩张量）：标签归一化、
prompt/target 构建、**长度预算按最长标签核算**、数据集的监督掩码与「超长抛错而非截断」、
生成结果解析（含六类互不为子串这一前提、复读判不合规、两个科室同时出现判无效）、
**候选打分双口径与长度偏置修正**、六类基线阈值换算、轮转平衡与近重复剪枝等准备期纯函数，
以及已入库切分本身的不变量（平衡、互斥、长度预算、嵌套前缀）。

## 对照实验

```bash
bash experiments/lora_triage_20260917/run_rank_check.sh   # rank 64 / 128
bash experiments/lora_triage_20260917/run_lr_check.sh     # lr 4e-4 / 8e-4
python experiments/lora_triage_20260917/analyze_scale.py  # 生成 saturation.md

# 学习率定下来之后，整条规模曲线在新学习率下重跑（12000 点复用 run_lr_check.sh 的产物）
bash experiments/lora_triage_20260917/run_scale_lr.sh --lr 0.0008
python experiments/lora_triage_20260917/analyze_scale.py --curve-lr 8e4 --runs-dir runs_3090
```

规模曲线在 9600 条处饱和，而语料还剩 98%——很容易据此写下「天花板来自模型容量」。
`lora_sentiment_20260914` 当初正是这么写的，而它错了两次（先错在基座容量，再错在适配器
容量）。这两组对照就是为了不重蹈覆辙：

- `run_rank_check.sh` 只改 rank → 证明 0.79 不是基座上限；
- `run_lr_check.sh` 只改学习率 → 证明 rank 的收益其实来自等效更新幅度。

结果：**rank 16 + lr 8e-4 = 0.8333**，与 rank 64（1.57M 参数）无显著差异，参数量只有 1/4。
瓶颈是一个没调过的超参数，不是容量。详见 [REPORT.md §3.3](REPORT.md)。

`run_scale_lr.sh` 是这条链的最后一环：学习率既然被证明偏低，原来那条曲线上的每一个点
都是在欠训练状态下测的。重跑的结论一分为二——**六个点全部被显著抬高，但饱和点没有移动**
（仍在 9600 条）。详见 [REPORT.md §3.4](REPORT.md) 与 [saturation_lr8e4.md](saturation_lr8e4.md)。

产物放在 `runs_3090/` 而不是 `runs/`：这一批跑在另一块显卡上，混在一起就查不出
「哪些数字来自同一台机器」了。`analyze_scale.py --runs-dir` 就是为此存在——
对照一节的 lr 2e-4 基线始终取自 `runs/`，重跑曲线取自指定目录。

## 当前状态

- ✅ 原始语料已下载并校验 sha256（370MB，不入库）
- ✅ 数据准备完成并核验（六类平衡、切分互斥、跨切分文本无重叠）
- ✅ 纯逻辑单元测试 46 例全通过
- ✅ 本地 CPU 冒烟干跑：训练与评测两条链路端到端跑通
- ✅ 云端全流程已跑完（RTX 3080 Ti，约 15 分钟）
- ✅ 正式评测 **准确率 0.7900**（rank 16, lr 2e-4），规模曲线在 9600 条饱和
- ✅ 对照实验：**rank 16 + lr 8e-4 达到 0.8333**，瓶颈是学习率而非容量
- ✅ lr 上界已兜住：8e-4 之后走平（1.6e-3 = 0.8250，3.2e-3 = 0.8283），8e-4 在拐点上
- ✅ 规模曲线已在 lr 8e-4 下重跑（RTX 3090，产物在 `runs_3090/`）：
  **六个点全部显著抬高，饱和点仍在 9600 条**——被推翻的是天花板的高度，不是位置
- ✅ 跨硬件锚点复现：12000 条 @ 2e-4 在 3090 上得 0.7917，与 3080 Ti 的 0.7900 差 1/600 题
- ⚠️ 最优学习率与数据量之间是否有交互作用未测（lr 扫描全部在 12000 条上做）
