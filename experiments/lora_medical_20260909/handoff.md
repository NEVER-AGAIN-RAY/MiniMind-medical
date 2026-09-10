# LoRA 医学微调训练与评估衔接说明（v2 数据与评估系统）

> 更新时间：2026-09-09 | 状态：本地实现与轻量无模型校验已就绪，云端全流程待执行

---

## 1. 当前已完成实现与交付物

### 1.1 独立评估脚本 `run_eval.py`
- **支持模式**：
  - `--mode base`: 加载基座模型（默认 `out/full_sft_768.pth`）进行评估。
  - `--mode lora`: 加载基座模型并挂载指定 LoRA（默认 `out/lora_medical_768.pth`）进行评估。
  - `--compare`: 对比 base 与 lora 产物，生成 Markdown 对比报告。
- **一致性生成配置**：
  - `model.eval()`, `torch.no_grad()`, `do_sample=False`, `open_thinking=False`。
  - 单轮对话无历史，`add_generation_prompt=True`。
  - MCQ `max_new_tokens=64`，医学问答 `max_new_tokens=512`，通用题 `max_new_tokens=256`。
  - 参考答案和解析仅用于保存与打分，不输入模型。
  - 严格上下文长度检查，防止输入加输出超长。
- **保守 MCQ 评分规则**：
  - 提示模型仅输出选项字母；只解析新生成回答开头，严禁全文随意检索首个 A-E。
  - 接受单独字母（`A`, `(B)`, `【C】`, `**D**` 等）与明确前缀（`答案：B`, `选A`, `正确答案是C` 等）。
  - 选项必须属于该题实际有效集合；遇到冲突选项（如 `选A或者B`, `AB`）或无法识别判定为无效（记入总题数并判错）。
- **验证集 Loss (`--val-loss`)**：
  - 复用 `SFTDataset` (max_length=512, `augment=False` 固定模板/固定标签，与训练脚本验证集口径一致)。
  - 按实际受监督的有效目标 token 数（`shift_labels != -100`）加权计算平均交叉熵，不直接平均 batch loss。
- **冒烟与输出隔离**：
  - 支持 `--limit N` 用于云端极速冒烟。
  - 冒烟产物默认存入 `smoke_base` / `smoke_lora`，不覆盖正式全量结果。
- **完整输出产物**：
  - `mcq_answers.jsonl`: 题目、选项、官方答案、模型原始回答、抽取答案、对错、解析状态。
  - `medqa_answers.jsonl`: 问题、参考答案、模型回答。
  - `general_answers.jsonl`: 问题、模型回答。
  - `summary.json`: 完成题数、MCQ 正确率、无效回答数、val_loss、生成配置、设备/dtype、模型与题集哈希身份。
- **报告生成 (`--compare`)**：
  - 强制校验 base 与 lora 的生成配置、题集版本、题目 ID 一致性；逐个核验实际题集哈希、完整题目及评分，检查相同基座、设备和 dtype。`--force` 仅允许比较冒烟结果，不能绕过完整性或配置检查。
  - 统计 MCQ 正确率、无效回答数、val_loss 差异，细分“错→对”与“对→错”题目清单。
  - 逐题并排展示医学问答与通用问题，预留标准人工临床评审栏（回答是否充分、事实错误、无依据诊疗建议、总体评价、备注），不自动编造人工评分。

### 1.2 Trainer 验证扩展 `trainer/train_lora.py`
- 新增参数 `--val_path` 与 `--val_interval`（默认不传，不传时完全保持原训练逻辑）。
- 训练前 step 0、每隔 `val_interval` 步、训练结束各评估一次 val loss。
- 末步防重：若最后一步已在 `val_interval` 中验证过，避免重复评估。
- 走现有 `Logger`，启用 `--use_wandb` 时同步记录。
- 与 `run_eval.py` 共用 `trainer/trainer_utils.py` 中的 `evaluate_val_loss` 函数，计算口径完全一致。
- 验证前后严格保持/恢复原有 `training` 状态，验证过程关闭梯度。

### 1.3 云端一键执行流 `experiments/lora_medical_20260909/run_cloud.sh`
- 单次云端 GPU 会话即可按既定顺序执行 7 个步骤：
  1. 环境、GPU、基座权重、数据与考卷预检
  2. 基座评估冒烟 (`--limit 2`)
  3. 4 步 LoRA 训练冒烟 + LoRA 权重重载与 `--limit 2` 评估冒烟
  4. 正式全量 Base 评估 (140 题 + val_loss)
  5. 正式全量 LoRA 训练 (5000 条医学数据, 1 epoch = 625 步, 全新初始化)
  6. 正式全量 LoRA 评估 (140 题 + val_loss)
  7. 生成前后对比报告 (`compare_report.md`)
- 包含自动跳过机制（已完成产物无需重复跑）与灵活调度参数（`--step`, `--from-step`, `--skip-smoke`, `--force`）。
- **日志落盘**：全流程终端输出（含 loss/val_loss 与报错堆栈）通过 `tee` 自动同步保存到 `experiments/lora_medical_20260909/logs/run_时间戳.log`（`latest.log` 软链指向最近一次），并在每次运行开头记录运行命令、GPU、关键依赖实测版本及完整 `pip freeze` 快照（`pip_freeze_时间戳.txt`）。纯 shell 层实现，不改变任何训练/评估代码逻辑。

---

## 2. 本地实际核验项与状态

由于本地 Mac 缺乏足够显存且受“不加载模型运行”硬性约束，本机仅完成轻量级与离线逻辑测试：

| 检查项 | 验证方式 | 状态 | 备注 |
| :--- | :--- | :---: | :--- |
| **Python 语法检查** | `py_compile` | ✅ 通过 | `trainer_utils.py`, `train_lora.py`, `run_eval.py` 均通过 |
| **命令行参数解析** | `--help` 无模型启动 | ✅ 通过 | 验证新增参数解析及默认值，不触发权重加载 |
| **MCQ 选项抽取逻辑** | 单元测试 (13 种模式) | ✅ 通过 | 单字母、各种中文前缀、冲突过滤、非选项纯文本均符合预期 |
| **val_loss 加权机制** | Mock 批次数学校验 | ✅ 通过 | 严格按 valid tokens 加权交叉熵计算，忽略 `-100` |
| **对比模式与一致性检查** | 5题合成数据测试 | ✅ 通过 | 错→对/对→错计数正确，ID错位与冒烟混用时能正确拦截报错 |
| **Shell 脚本语法** | `bash -n` | ✅ 通过 | `run_cloud.sh` 无语法错误 |
| **日志落盘与环境记录** | 替身进程 Shell 模拟运行 | ✅ 通过 | 中断/完成/重跑流程不变，终端输出完整落盘 logs/，含运行命令与 pip freeze 快照 |
| **锁定题集完整性** | `wc -l` 与哈希对比 | ✅ 通过 | 100 MCQ + 30 MedQA + 10 General + manifest.json 保持锁定 |

---

## 3. 云端待执行项（需租用 GPU）

以下所有步骤**必须在租用的云端 GPU 上执行**，本地不得伪造通过结论：

1. **真实权重与 CUDA 加载**：
   - 在 GPU 上加载 `out/full_sft_768.pth`，确认显存占用与 bfloat16/float16 推理正常。
2. **基座与 LoRA 极简冒烟**：
   - 执行 Step 2 (`--limit 2`) 与 Step 3 (4 步 LoRA 训练 + 保存重载评估)，确认云端环境与依赖无隐患。
3. **正式训练收敛**：
   - 观察 train_5000 在 625 步训练期间的 train loss 与周期性 val loss 曲线（观察是否在 1 epoch 内平稳下降）。
4. **正式前后对比与人工评分**：
   - 运行 Step 4、6、7 生成正式产物与 `compare_report.md`。
   - 由人工结合报告中的评分栏对 30 道医学问答进行临床质量审核。

---

## 4. 上传到云端的文件清单与依赖

### 4.1 需上传的文件与目录
- 项目代码及配置：
  - `model/`（包含架构定义与 tokenizer）
  - `trainer/`（包含训练与评估工具）
  - `dataset/`
  - `run_eval.py`
  - `experiments/lora_medical_20260909/`（包含 `data/v2/`, `eval/v1/`, `run_cloud.sh`）
- 基座模型权重：
  - `out/full_sft_768.pth`（约 137.7 MB）

> 打包上传建议命令（在本地终端）：
> ```bash
> tar -czvf minimind_cloud.tar.gz \
>   model/ trainer/ dataset/ run_eval.py \
>   out/full_sft_768.pth \
>   experiments/lora_medical_20260909/data/v2/ \
>   experiments/lora_medical_20260909/eval/v1/ \
>   experiments/lora_medical_20260909/run_cloud.sh
> ```

### 4.2 云端 Python 核心依赖
- Python >= 3.10
- PyTorch >= 2.1 (带 CUDA 支持)
- `transformers` >= 4.40
- `datasets` >= 2.18
- `accelerate`

---

## 5. 云端运行命令与下载指引

### 5.1 登录云端后一键执行
解压进入项目根目录后，执行：

```bash
# 执行全部 7 个步骤（环境检查 -> 冒烟测试 -> 正式基座评估 -> 正式LoRA训练 -> 正式LoRA评估 -> 对比报告）
bash experiments/lora_medical_20260909/run_cloud.sh
```

### 5.2 常用控制选项
- 仅在该云端环境已经通过冒烟验证时，可跳过冒烟步骤：
  ```bash
  bash experiments/lora_medical_20260909/run_cloud.sh --skip-smoke
  ```
- 若训练已完成，只需重新生成评估或报告：
  ```bash
  # 从步骤 6 (LoRA评估) 开始执行
  bash experiments/lora_medical_20260909/run_cloud.sh --from-step 6
  
  # 仅重新生成对比报告
  bash experiments/lora_medical_20260909/run_cloud.sh --step 7
  ```

### 5.3 独立执行命令参考（若不使用脚本）
```bash
# 1. 基座评估
python run_eval.py --mode base --val_loss --out_dir experiments/lora_medical_20260909/eval_results/base

# 2. 正式训练 (5000条, 1 epoch = 625步)
python trainer/train_lora.py \
  --data_path experiments/lora_medical_20260909/data/v2/train_5000.jsonl \
  --val_path experiments/lora_medical_20260909/data/v2/val.jsonl \
  --val_interval 50 --save_interval 125 --max_seq_len 512 \
  --epochs 1 --batch_size 8 --accumulation_steps 1 \
  --lora_name lora_medical_formal --from_weight full_sft --from_resume 0 --save_dir out

# 3. LoRA 评估
python run_eval.py --mode lora --lora_weight out/lora_medical_formal_768.pth --val_loss --out_dir experiments/lora_medical_20260909/eval_results/lora

# 4. 生成前后对比报告
python run_eval.py --compare \
  --base_dir experiments/lora_medical_20260909/eval_results/base \
  --lora_dir experiments/lora_medical_20260909/eval_results/lora \
  --save_report experiments/lora_medical_20260909/eval_results/compare_report.md
```

> 注：通过 `run_cloud.sh` 运行时终端输出会自动保存到 `experiments/lora_medical_20260909/logs/`；若手动执行上述独立命令，请自行追加 `2>&1 | tee -a experiments/lora_medical_20260909/logs/manual_$(date +%Y%m%d_%H%M%S).log` 留存日志。

### 5.4 产物位置与下载方式
执行完成后，生成的重要产物均集中在：
- `out/lora_medical_formal_768.pth`: 正式微调 LoRA 权重
- `out/lora_medical_formal_768.complete.sha256`: 训练完成标记与文件校验和（审计用，不能用于恢复训练）
- `experiments/lora_medical_20260909/logs/`: 完整终端日志（`run_时间戳.log`，含 loss/val_loss、报错堆栈、运行参数、实测环境版本）与 `pip_freeze_时间戳.txt` 依赖快照
- `experiments/lora_medical_20260909/eval_results/base/`: 基座全量评估产物 (4个文件)
- `experiments/lora_medical_20260909/eval_results/lora/`: LoRA 全量评估产物 (4个文件)
- `experiments/lora_medical_20260909/eval_results/compare_report.md`: Markdown 对比报告

**在云端打包命令**（含日志与完成标记）：
```bash
tar -czvf minimind_results.tar.gz \
  out/lora_medical_formal_768.pth \
  out/lora_medical_formal_768.complete.sha256 \
  experiments/lora_medical_20260909/logs/ \
  experiments/lora_medical_20260909/eval_results/
```

如需保留续训能力（日后 `--from_resume 1` 断点续训），打包时追加检查点目录：
```bash
tar -czvf minimind_results.tar.gz \
  out/lora_medical_formal_768.pth \
  out/lora_medical_formal_768.complete.sha256 \
  experiments/lora_medical_20260909/logs/ \
  experiments/lora_medical_20260909/eval_results/ \
  checkpoints/
```

**本地下载命令**（在本地电脑终端执行）：
```bash
scp -P <端口> <用户名>@<云端IP>:<云端工作路径>/minimind_results.tar.gz ./
```
解压后即可查阅 `compare_report.md` 并填写人工临床评估；训练与评估的完整过程日志见 `logs/` 目录。


## 6. 代码复核修复（2026-09-09）

- 训练 tokenizer、基座与 checkpoint 目录按项目根目录解析，支持文档中的根目录启动命令。
- 云端所有模型步骤必须检测到 CUDA；自动选择受支持的 bf16/fp16，训练和两次评估显式使用相同设置。
- 正式训练成功退出后才写 `out/lora_medical_formal_768.complete.sha256`，记录权重、数据和相关代码的校验和。中断留下的权重不能代表完成；再次执行 Step 5 会从头开始正式训练，不做自动断点续训。旧权重没有完成标记时，也需要重新执行 Step 5。
- 完成标记及其文件校验通过才能进入 Step 6/7。修改记录中的文件后需重新训练；不要手工补造标记。
- 跳过正式评估前使用 `run_eval.py --check-results` 核验实际结果、题目完整性、评分、当前权重、验证集和设备精度；过期或不完整结果会重新评估。
- 答案冲突按无效处理；评分版本 2 写入 summary。锁定题集及 manifest 保持不变，其中旧的首字母评分描述由本说明和版本 2 实现更正。
- 医学/通用回答中的换行、竖线和 Markdown 字符正确转义，避免破坏表格。
- 本地执行 `python -m unittest discover -s tests -v`：14 项通过，含真实 Shell + 替身进程模拟中断/完成/重跑，未加载真实模型、未训练或生成回答。
- 验证集一律使用固定模板固定标签：`SFTDataset` 新增 `augment` 开关，`train_lora.py` 验证集与 `run_eval.py --val-loss` 均传 `augment=False`，关闭随机 system 注入与空 think 随机移除，保证 val_loss 可复现、前后可比；训练集保持随机增强不变。
- Python 语法、CLI 帮助与 Shell 语法可在本地检查；真实 CUDA 冒烟、正式训练和模型效果仍待云端验证。
- 云端全流程终端日志通过 `tee` 自动落盘到 `experiments/lora_medical_20260909/logs/`（含报错堆栈），并在运行开头记录运行命令、GPU、关键依赖实测版本与完整 `pip freeze` 快照；下载打包包含日志与完成标记，需要续训时再追加 `checkpoints/`。

云端完整执行命令保持不变：

```bash
bash experiments/lora_medical_20260909/run_cloud.sh
```

下载时建议把完成标记一并保存以便审计；它不是模型权重，也不能单独用于恢复训练。
