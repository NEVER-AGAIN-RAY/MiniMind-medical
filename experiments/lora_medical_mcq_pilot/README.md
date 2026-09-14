# 医学单选题 LoRA 微调实验（已结束 / 结论：不继续）

**状态**：2026-09-14 归档，方向已放弃，后续转向情感分类。
**基座**：`out/full_sft_768.pth`（minimind-3，768 dim / 8 层 / ~64M，md5 前缀 `942a011c4b`）
**数据**：CMExam 中国医师资格考试题库（原始 CSV 不入库，见 `data/raw/MANIFEST.json`）

## 一句话结论

LoRA 能让 64M 基座学会任意输出格式，但**无法注入它本来没有的医学知识**。
六轮实验中没有任何一轮的未见题准确率显著超过随机水平（0.2015）。

## 完整结果

锁定验证集 200 题，确定性生成（`do_sample=False`），随机基线 **0.2015**。

| # | Run | 训练目标 | 训练量 | 未见题准确率 | 备注 |
|---|---|---|---|---|---|
| 0 | — | 随机基线 | — | **0.2015** | 五选一 |
| 1 | `formal` | 选项字母 | 1000 | 0.25 | 仅评了 val 前 100 题，样本量小 |
| 2 | `letter_reference_val200` | 选项字母 | 1000 | **0.21** | 锁定 200 题，作为主对照 |
| 3 | `answer_text_v1` | 答案文字 + 字母 | 1000 | **0.17** | 格式合规 0.99，自然终止 1.00 |
| 4 | `knowledge_5k_v1` | 答案文字 + CMExam 完整解析 | 5000 | 0.185 生成 / 0.195 候选打分 | 截断率 0.935，解析太长 |
| 5 | `concise_knowledge_5k_v1` | 答案文字 + 一句精炼知识 | 5000 | **0.235** 候选打分 / 0.20 生成 | 最好成绩，但 p=0.256 不显著 |

## 关键诊断：瓶颈是泛化，不是训练能力

这是整个实验最有价值的发现，来自 `knowledge_overfit50_v1` 和 `concise_overfit50_v1`
——用 50 道题反复训练，看模型**能不能记住**：

| 监督目标 | 配置 | 50 题训练集准确率 |
|---|---|---|
| 答案 + 完整长解析 | rank16, 50 epoch | 0.48 |
| 答案 + 精炼知识句 | rank16 | **1.00** |
| 答案 + 精炼知识句 | rank32 | **1.00** |
| 答案 + 精炼知识句 | 全参数（63.9M） | **1.00** |

含义：

1. **训练管线、LoRA 实现、学习率都没问题**——rank16（39 万可训练参数）足以把 50 题背到 100%。
2. **长解析会稀释训练信号**，同样 rank16 只能到 0.48；缩短监督目标就解决了。
3. 但记忆能力 100% 的同时未见题只有 0.235——**模型在记题，不在学医**。
4. 因此加大 rank、改全参数、扩到 20000 条都不会有用：可记忆性已经饱和，缺的是基座里根本不存在的医学知识。

另一个佐证：字母目标的输出分布严重偏斜（200 题中 B 占 76 个，38%），模型是在猜先验而非判断。

## 为什么换方向

LoRA 的低秩增量适合学**映射**（输入表面特征 → 输出），不适合注入**知识**。
医学单选题的答案不在题干里，在模型不具备的知识库里，所以这条路在 64M 基座上走不通。

上游 MiniMind 作者对 `minimind-3-exam` 的说明也印证了这点：那次 `lora_exam.jsonl`
训练"几乎没有额外注入新知识，只是对齐选择题的格式"。

后续方向应满足：**判断所需的全部信息已经在输入文本里**。

## 目录内容

```
config.json              实验配置（含 CMExam 官方下载 URL）
audit_record.json        代码审核记录
prepare_data.py          MCQ 数据切分（去重、排除历史评测题）
prepare_knowledge_data.py / prepare_curated_knowledge.py
                         知识目标数据构造
mcq_dataset.py           prompt 模板与 token 构造
train_mcq_lora.py        LoRA 训练入口
eval_mcq.py              生成式评测 + 候选打分评测
diagnose_concise_overfit.py / eval_concise_candidate.py
                         50 题可记忆性诊断
test_*.py                单元测试
run_cloud.sh             云端流水线
data/raw/MANIFEST.json   原始 CSV 的 URL + sha256（CSV 本身不入库）
data/<name>/             各轮次的冻结切分 + manifest
runs/<name>/             config_effective / train_summary / eval_results / 结论
```

## 复现

```bash
# 1. 重新下载原始题库（校验 data/raw/MANIFEST.json 中的 sha256）
python experiments/lora_medical_mcq_pilot/prepare_data.py --download

# 2. 训练（权重已清理，需重跑）
python experiments/lora_medical_mcq_pilot/train_mcq_lora.py

# 3. 评测
python experiments/lora_medical_mcq_pilot/eval_mcq.py
```

**归档时的路径变动**：各 run 的 `train_summary.json` 已从 `runs/<name>/checkpoints/`
移到 `runs/<name>/`——`checkpoints/` 被 `.gitignore` 整目录排除，留在原处这些训练记录
就无法入库。`train_mcq_lora.py` 仍写入 `checkpoints/train_summary.json`，因此**全新的
训练→评测流程不受影响**；只有对已归档 run 重跑 `eval_mcq.py` 会找不到该文件，而这些
run 的权重本就已删除。

**权重说明**：清理时只保留了 `runs/concise_overfit50_v1/lora_rank16_final.pth`
（复现"50 题记忆到 100%"这个关键正面结论所需）。其余 20 个 `.pth` 均为负面结果的
中间产物，已删除；各 run 的 `train_summary.json` 中记录了对应的 md5。
清理前的完整快照见提交 `b714f50`。
