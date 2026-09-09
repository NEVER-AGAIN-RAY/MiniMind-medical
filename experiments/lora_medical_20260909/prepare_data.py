#!/usr/bin/env python
"""LoRA 医学微调数据准备脚本 v2（可重复运行，输出逐字节一致）。

v2 相对 v1 的变更（对应 audit/review.md 整改要求）:
  1. 版本化输出: --out-dir（默认 data/v2），v1 结果保留在 data/v1 不动
  2. 读取结构化持久排除清单 audit/exclusions.jsonl（按稳定 q_md5 排除，不依赖行号）
  3. 验证/测试集固定: 成员 pin 到 val_members.json/test_members.json，改变 --train-size
     不影响 val/test；被剔除成员从候选池按 q_md5 升序确定性补充，补充记录写入 manifest
  4. 训练集: train_5000（全量）+ train_1000（入门子集，取前1000）+ smoke（从1000中取前32）
  5. text_form_score: 仅是文字形式筛选分数（字符丰富度/篇幅/分段/标点），
     不代表医学正确性；医学正确性以 audit/ 审核记录为准
  6. 近重复候选检查: 训练与验证/测试之间字符二元组 Jaccard>=0.5 的对输出到
     audit/near_duplicates.jsonl 供人工判断（不自动排除）
  7. 验收全量检查: JSON/非空问答/角色顺序/重复问题；全部 22 种模板变体
     （无system + 10个system）x（保留/移除空think）逐条核验 <=512 tokens、
     有效回答标签、受监督结束标记；SFTDataset 端到端加载；临时目录重跑一致性
  8. 失败返回非零退出码

用法:
  ../../.venv/bin/python prepare_data.py [--train-size 5000]
"""
import argparse
import hashlib
import json
import os
import random
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))

from transformers import AutoTokenizer  # noqa: E402
from dataset import lm_dataset as ld  # noqa: E402
from dataset.lm_dataset import SFTDataset  # noqa: E402

V1_DIR = HERE / 'data' / 'v1'
AUDIT_DIR = HERE / 'audit'
EXCLUSIONS_PATH = AUDIT_DIR / 'exclusions.jsonl'

# 文字形式评分说明（不代表医学正确性）
SCORE_DESC = ('text_form_score: 仅衡量文字形式（字符丰富度/篇幅/分段/标点/问题长度），'
              '不代表医学正确性；医学正确性以 audit/ 审核记录为准')
EOS_DESC = ('结束标记检查仅证明序列边界完整且结束标记参与监督，不代表原文语义完整')


# ---------- 基础工具 ----------
def md5_text(s: str) -> str:
    return hashlib.md5(s.encode('utf-8')).hexdigest()


def md5_file(p: Path) -> str:
    return hashlib.md5(p.read_bytes()).hexdigest()


def q_md5(row: dict) -> str:
    return md5_text(row['conversations'][0]['content'].strip())


def norm_q(q: str) -> str:
    """归一化问题文本: 去空白与常见标点、语气词，用于近似去重。"""
    return re.sub(r'[\s，。？！,.?!、的了吗呢吧啊]', '', q)


def distinct2(s: str) -> float:
    cs = list(s)
    if len(cs) < 2:
        return 0.0
    return len(set(zip(cs, cs[1:]))) / (len(cs) - 1)


def capture_system_prompts():
    """从 pre_processing_chat 中提取真实 SYSTEM_PROMPTS（单一事实来源）。"""
    captured = []
    orig_random, orig_choice = random.random, random.choice
    random.random = lambda: 0.0
    random.choice = lambda seq: (captured.extend(seq), seq[0])[1]
    try:
        ld.pre_processing_chat([{'role': 'user', 'content': 'x'}])
    finally:
        random.random, random.choice = orig_random, orig_choice
    assert captured, '未能捕获 SYSTEM_PROMPTS'
    return captured


def force_remove_empty_think(prompt: str) -> str:
    """调用训练同款 post_processing_chat，强制走移除空 think 标记分支。"""
    orig = random.random
    random.random = lambda: 1.0  # > empty_think_ratio(0.2) -> 移除
    try:
        return ld.post_processing_chat(prompt)
    finally:
        random.random = orig


# ---------- 文字形式评分（不代表医学正确性） ----------
def score_row(row: dict, tokens: int) -> float:
    q = row['conversations'][0]['content'].strip()
    a = row['conversations'][1]['content'].strip()
    info = min(max((distinct2(a) - 0.60) / 0.20, 0.0), 1.0)
    if 150 <= tokens <= 480:
        length = 1.0
    elif 80 <= tokens < 150:
        length = tokens / 150
    elif 480 < tokens <= 512:
        length = 1.0 - 0.4 * (tokens - 480) / 32
    else:
        length = 0.4
    paras = [p for p in a.split('\n') if p.strip()]
    comp = 0.6 * (1.0 if a.endswith(('。', '！', '？', '…', '）', ']', '"', '"')) else 0.0) \
        + 0.4 * (1.0 if len(paras) >= 2 else 0.0)
    l = len(q)
    qs = 1.0 if 8 <= l <= 60 else (0.8 if l <= 100 else 0.5)
    return 0.4 * info + 0.3 * length + 0.2 * comp + 0.1 * qs


# ---------- 排除清单 ----------
def load_exclusions():
    """返回 (条目列表, 全集排除 q_md5, 仅训练排除 q_md5)。"""
    entries, excl_all, excl_train = [], set(), set()
    if EXCLUSIONS_PATH.exists():
        for line in EXCLUSIONS_PATH.read_text(encoding='utf-8').splitlines():
            line = line.strip()
            if not line:
                continue
            e = json.loads(line)
            entries.append(e)
            if e.get('decision') == 'exclude':
                (excl_train if e.get('exclude_from') == 'train' else excl_all).add(e['q_md5'])
    return entries, excl_all, excl_train


def bootstrap_pins():
    """从 v1 验证/测试集引导初始 pin（其中的已排除成员会被剔除并补充）。"""
    pins = {}
    for name in ('val', 'test'):
        p = V1_DIR / f'{name}.jsonl'
        assert p.exists(), f'缺少 v1 {name}.jsonl，无法引导 pin'
        members = []
        for line in p.read_text(encoding='utf-8').splitlines():
            if line.strip():
                members.append(q_md5(json.loads(line)))
        pins[name] = members
    return pins


# ---------- 生成流程（纯函数: 相同输入 -> 相同输出） ----------
def generate(args, pin_source_dir: Path):
    """执行完整流程，返回 (contents, pins, manifest_meta)。不写任何文件。"""
    funnel = []

    # ===== 1. 读入 + 结构过滤 =====
    rows = []
    src = ROOT / 'dataset' / 'lora_medical.jsonl'
    with src.open(encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            convs = r.get('conversations')
            ok = (isinstance(convs, list) and len(convs) >= 2
                  and isinstance(convs[0], dict) and convs[0].get('role') == 'user'
                  and isinstance(convs[0].get('content'), str) and convs[0]['content'].strip()
                  and isinstance(convs[1], dict) and convs[1].get('role') == 'assistant'
                  and isinstance(convs[1].get('content'), str) and len(convs[1]['content'].strip()) >= 5)
            if ok:
                rows.append(r)
    funnel.append({'stage': 'structural', 'count': len(rows),
                   'reason': '剔除坏JSON/角色错误/空问答'})

    # ===== 2. 去重: 精确 + 归一化，保留首条 =====
    seen_e, seen_n, deduped = set(), set(), []
    for r in rows:
        e = q_md5(r)
        n = md5_text(norm_q(r['conversations'][0]['content']))
        if e in seen_e or n in seen_n:
            continue
        seen_e.add(e)
        seen_n.add(n)
        deduped.append(r)
    funnel.append({'stage': 'dedup', 'count': len(deduped),
                   'reason': '精确+归一化重复问题，保留首条'})

    # ===== 3. 硬性文字质量过滤 =====
    hard = []
    for r in deduped:
        a = r['conversations'][1]['content'].strip()
        if len(a) < 30:
            continue
        if len(a) >= 50 and distinct2(a) < 0.45:
            continue
        hard.append(r)
    funnel.append({'stage': 'hard_quality', 'count': len(hard),
                   'reason': '回答<30字或复读(distinct-2<0.45)'})

    # ===== 4. 排除清单（稳定 q_md5，不依赖行号） =====
    entries, excl_all, excl_train = load_exclusions()
    universe = [r for r in hard if q_md5(r) not in excl_all]
    funnel.append({'stage': 'exclusions', 'count': len(universe),
                   'reason': f'排除清单 {len(excl_all)} 个全量排除项 (audit/exclusions.jsonl)'})

    # ===== 5. 建训练同款助手; 最坏情况测长 + 评分 =====
    tok = AutoTokenizer.from_pretrained(ROOT / 'model', local_files_only=True)
    helper = object.__new__(SFTDataset)
    helper.tokenizer = tok
    helper.max_length = args.max_tokens
    helper.bos_id = tok(f'{tok.bos_token}assistant\n', add_special_tokens=False).input_ids
    helper.eos_id = tok(f'{tok.eos_token}\n', add_special_tokens=False).input_ids

    systems = capture_system_prompts()
    probe = [{'role': 'user', 'content': '问'}, {'role': 'assistant', 'content': '答'}]
    worst_system = max(
        systems,
        key=lambda s: len(tok(helper.create_chat_prompt([{'role': 'system', 'content': s}] + probe)).input_ids),
    )

    scored = []  # (score, q_md5, row, tokens)
    for i, r in enumerate(universe):
        prompt = helper.create_chat_prompt([{'role': 'system', 'content': worst_system}] + r['conversations'])
        t = len(tok(prompt).input_ids)  # 与 SFTDataset.__getitem__ 相同的默认 tokenize
        if t > args.max_tokens:
            continue  # 最坏情况(最长system+保留think)超限即剔除，保证全部22变体不超限
        scored.append((score_row(r, t), q_md5(r), r, t))
        if (i + 1) % 5000 == 0:
            print(f'  测长进度 {i + 1}/{len(universe)}')
    funnel.append({'stage': 'token_filter', 'count': len(scored),
                   'reason': f'最坏模板变体(最长system+保留think)>{args.max_tokens} tokens'})

    # ===== 6. text_form_score 淘汰底部 =====
    scored.sort(key=lambda x: (x[0], x[1]))
    cut = int(len(scored) * args.drop_bottom)
    score_cut = scored[cut][0] if cut < len(scored) else None
    quality_pool = scored[cut:]
    funnel.append({'stage': 'score_drop', 'count': len(quality_pool),
                   'reason': f'text_form_score 底部 {args.drop_bottom:.0%} 淘汰（分数仅代表文字形式）'})

    # ===== 7. 固定验证/测试集（pin 机制，与 train-size 无关） =====
    universe_by_md5 = {x[1]: x for x in scored}  # 通过全部硬过滤的候选（pin 成员资格）
    quality_by_md5 = {x[1]: x for x in quality_pool}  # 高分池（补充与训练候选来源）
    if (pin_source_dir / 'val_members.json').exists():
        base_pins = {name: json.loads((pin_source_dir / f'{name}_members.json').read_text(encoding='utf-8'))
                     for name in ('val', 'test')}
    else:
        base_pins = bootstrap_pins()  # 首次运行: 从 v1 成员引导
    kept_val = [m for m in base_pins['val'] if m in universe_by_md5][:args.val_size]
    kept_test = [m for m in base_pins['test'] if m in universe_by_md5][:args.test_size]
    cur_val, cur_test = set(kept_val), set(kept_test)
    # 被剔除/失效的成员从高分池按 q_md5 升序确定性补充（记录进 manifest 供人工审核）
    refills = {}
    for name, cur, want in (('val', cur_val, args.val_size), ('test', cur_test, args.test_size)):
        refill = []
        for m in sorted(quality_by_md5):
            if len(cur) >= want:
                break
            if m in cur or m in cur_val or m in cur_test:
                continue
            cur.add(m)
            refill.append(m)
        assert len(cur) == want, f'{name} 补充后应为 {want}，实际 {len(cur)}'
        refills[name] = refill
    pins = {'val': sorted(cur_val), 'test': sorted(cur_test)}
    val_set, test_set = cur_val, cur_test
    assert not (val_set & test_set), 'val/test pin 重叠'

    # ===== 8. 训练集: 候选池 - val - test - 仅训练排除，固定种子洗牌 =====
    train_cands = [x for x in quality_pool if x[1] not in val_set and x[1] not in test_set
                   and x[1] not in excl_train]
    train_cands.sort(key=lambda x: x[1])  # 按 q_md5 稳定排序，消除输入顺序影响
    rng = random.Random(args.seed)
    rng.shuffle(train_cands)
    assert len(train_cands) >= args.train_size, '训练候选不足'
    train_all = train_cands[:args.train_size]
    train_intro = train_all[:args.intro_size]
    smoke = train_intro[:args.smoke_size]

    # ===== 9. 序列化（不写盘，返回内容） =====
    def dump(xs):
        return ''.join(json.dumps(x[2], ensure_ascii=False) + '\n' for x in xs)

    contents = {
        'train_5000.jsonl': dump(train_all),
        'train_1000.jsonl': dump(train_intro),
        'val.jsonl': dump([universe_by_md5[m] for m in sorted(val_set)]),
        'test.jsonl': dump([universe_by_md5[m] for m in sorted(test_set)]),
        'smoke.jsonl': dump(smoke),
    }

    # ===== 10. 近重复候选（训练 vs 验证/测试，仅输出候选供人工判断） =====
    def bigrams(s):
        cs = list(s)
        return set(zip(cs, cs[1:])) if len(cs) >= 2 else {('<s>',)}

    train_bg = {x[1]: bigrams(x[2]['conversations'][0]['content'].strip()) for x in train_all}
    vt_rows = [universe_by_md5[m][2] for m in (sorted(val_set) + sorted(test_set))]
    near_dup = []
    for r in vt_rows:
        q = r['conversations'][0]['content'].strip()
        bg = bigrams(q)
        for tm, tbg in train_bg.items():
            inter = len(bg & tbg)
            if inter == 0:
                continue
            j = inter / len(bg | tbg)
            if j >= 0.5:
                near_dup.append((round(j, 4), r['conversations'][0]['content'].strip(),
                                 q_md5(r), tm))
    near_dup.sort(reverse=True)

    meta = {
        'funnel': funnel, 'score_cut': score_cut, 'pins': pins, 'refills': refills,
        'near_dup_count': len(near_dup), 'systems': systems, 'worst_system': worst_system,
        'excl_all': excl_all, 'excl_train': excl_train, 'source_md5': md5_file(src),
        'source_rows': len(rows), 'tok': tok, 'helper': helper,
    }
    return contents, meta, near_dup


# ---------- 全量验收 ----------
def run_acceptance(out_dir: Path, args, meta, near_dup, quiet=False):
    """对全部输出做全量检查，返回 (all_ok, 报告行列表)。"""
    tok, helper = meta['tok'], meta['helper']
    eos_id, bos_id = helper.eos_id, helper.bos_id
    reports, all_ok = [], True

    def log(ok, msg):
        nonlocal all_ok
        all_ok &= ok
        reports.append(f"[{'PASS' if ok else 'FAIL'}] {msg}")

    files = ['train_5000.jsonl', 'train_1000.jsonl', 'val.jsonl', 'test.jsonl', 'smoke.jsonl']
    data = {}
    for name in files:
        lines = [l for l in (out_dir / name).read_text(encoding='utf-8').splitlines() if l.strip()]
        rows = [json.loads(l) for l in lines]  # JSON 有效性
        data[name] = rows
        ok = all(isinstance(r.get('conversations'), list) and len(r['conversations']) >= 2
                 and r['conversations'][0].get('role') == 'user' and r['conversations'][0].get('content', '').strip()
                 and r['conversations'][1].get('role') == 'assistant' and r['conversations'][1].get('content', '').strip()
                 for r in rows)
        log(ok, f'{name}: {len(rows)} 条，JSON/非空问答/角色顺序 全量检查')

    # 数量
    for name, want in [('train_5000.jsonl', args.train_size), ('train_1000.jsonl', args.intro_size),
                       ('val.jsonl', args.val_size), ('test.jsonl', args.test_size),
                       ('smoke.jsonl', args.smoke_size)]:
        log(len(data[name]) == want, f'{name} 数量 = {len(data[name])} (期望 {want})')

    # 子集关系
    def qset(name):
        return {q_md5(r) for r in data[name]}
    log(qset('train_1000.jsonl') <= qset('train_5000.jsonl'), 'train_1000 ⊆ train_5000')
    log(qset('smoke.jsonl') <= qset('train_1000.jsonl'), 'smoke ⊆ train_1000')

    # 重复问题: 各文件内部 + 训练与验证/测试互斥
    for name in files:
        qs = [q_md5(r) for r in data[name]]
        log(len(qs) == len(set(qs)), f'{name} 内部无重复问题')
    qt, qv, qte = qset('train_5000.jsonl'), qset('val.jsonl'), qset('test.jsonl')
    log(not (qt & qv) and not (qt & qte) and not (qv & qte),
        f'训练/验证/测试互斥 (交集 {len(qt & qv)}/{len(qt & qte)}/{len(qv & qte)})')

    # 全部 22 种模板变体: (无system + 10个system) x (保留/移除空think)
    systems = meta['systems']
    bad, variants_total, think_diff = 0, 0, 0
    n_samples = sum(len(data[n]) for n in files)
    for name in files:
        for r in data[name]:
            for sys_cfg in [None] + systems:
                convs = ([{'role': 'system', 'content': sys_cfg}] if sys_cfg else []) + r['conversations']
                base = helper.create_chat_prompt(convs)
                variants = [base]
                removed = force_remove_empty_think(base)
                if removed != base:
                    think_diff += 1
                    variants.append(removed)
                for v in variants:
                    ids = tok(v).input_ids
                    labels = helper.generate_labels(ids)
                    variants_total += 1
                    if (len(ids) > args.max_tokens
                            or ids[-len(eos_id):] != eos_id
                            or not any(x != -100 for x in labels)
                            or labels[-1] == -100):
                        bad += 1
    log(bad == 0, f'全部 {n_samples} 条 x 22 种模板变体（实测 {variants_total} 个序列，'
                  f'其中 {think_diff} 处 think 变体产生不同序列）: '
                  f'≤{args.max_tokens} tokens、有效回答标签、受监督结束标记（异常 {bad}）')

    # 近重复候选输出（人工判断，不自动排除）
    nd_path = AUDIT_DIR / 'near_duplicates.jsonl'
    with nd_path.open('w', encoding='utf-8') as f:
        for j, vq, vmd5, tmd5 in near_dup:
            f.write(json.dumps({'jaccard': j, 'val_test_q_md5': vmd5, 'train_q_md5': tmd5,
                                'val_test_question': vq}, ensure_ascii=False) + '\n')
    log(True, f'近重复候选 {len(near_dup)} 对已输出至 audit/near_duplicates.jsonl 供人工判断')

    # SFTDataset 端到端加载
    for name in files:
        try:
            ds = SFTDataset(str(out_dir / name), tok, max_length=args.max_tokens)
            x, y = ds[0]
            ok = x.shape[0] == args.max_tokens and any(v != -100 for v in y.tolist())
        except Exception as e:
            reports.append(f'  SFTDataset 加载 {name} 异常: {e}')
            ok = False
        log(ok, f'SFTDataset 端到端加载 {name}')

    return all_ok, reports


# ---------- 主流程 ----------
def main():
    ap = argparse.ArgumentParser(description='LoRA 医学数据准备 v2')
    ap.add_argument('--out-dir', type=Path, default=HERE / 'data' / 'v2')
    ap.add_argument('--train-size', type=int, default=5000)
    ap.add_argument('--intro-size', type=int, default=1000)
    ap.add_argument('--val-size', type=int, default=100)
    ap.add_argument('--test-size', type=int, default=100)
    ap.add_argument('--smoke-size', type=int, default=32)
    ap.add_argument('--max-tokens', type=int, default=512)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--drop-bottom', type=float, default=0.15)
    args = ap.parse_args()

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    print('===== 生成数据（主运行）=====')
    contents, meta, near_dup = generate(args, out_dir)

    # 先写出 pin 成员文件，使临时目录重跑读取与主运行一致的固定成员
    for name in ('val', 'test'):
        p = out_dir / f'{name}_members.json'
        tmp = p.with_suffix('.tmp')
        tmp.write_text(json.dumps(meta['pins'][name]), encoding='utf-8')
        os.replace(tmp, p)

    # 临时目录重跑一致性（相同配置 -> 相同文件，纯内存比对）
    print('===== 临时目录重跑一致性检查 =====')
    contents2, _, _ = generate(args, out_dir)  # pin 文件已存在，读取固定成员
    same = contents == contents2
    if not same:
        for name in contents:
            if contents[name] != contents2[name]:
                print(f'  不一致文件: {name}')
    print(f"[{'PASS' if same else 'FAIL'}] 临时目录重跑输出逐字节一致")

    print('===== 写出文件 =====')
    files_meta = {}
    for name, c in contents.items():
        p = out_dir / name
        tmp = p.with_suffix(p.suffix + '.tmp')
        tmp.write_text(c, encoding='utf-8')
        os.replace(tmp, p)
        files_meta[name] = {'count': c.count('\n'), 'md5': md5_text(c),
                           'bytes': len(c.encode('utf-8'))}

    # manifest
    import transformers
    import torch
    tok_files = {f.name: md5_file(f) for f in sorted((ROOT / 'model').glob('*.json'))}
    manifest = {
        'version': 'v2',
        'seed': args.seed,
        'params': {'train_size': args.train_size, 'intro_size': args.intro_size,
                   'val_size': args.val_size, 'test_size': args.test_size,
                   'smoke_size': args.smoke_size, 'max_tokens': args.max_tokens,
                   'drop_bottom': args.drop_bottom},
        'provenance': {
            'source_dataset': {'path': 'dataset/lora_medical.jsonl', 'md5': meta['source_md5'],
                               'rows': meta['source_rows']},
            'tokenizer': {'path': 'model/', 'files': tok_files,
                          'note': 'chat_template 内嵌于 tokenizer_config.json'},
            'prepare_script': {'path': 'experiments/lora_medical_20260909/prepare_data.py',
                               'md5': md5_file(Path(__file__))},
            'exclusions': {'path': 'audit/exclusions.jsonl', 'md5': md5_file(EXCLUSIONS_PATH),
                          'entries': len(load_exclusions()[0])},
            'pinned_membership': {'val': 'data/v2/val_members.json', 'test': 'data/v2/test_members.json',
                                  'note': 'val/test 成员固定，改变 train-size 不影响'},
            'dependencies': {'python': sys.version.split()[0], 'torch': torch.__version__,
                             'transformers': transformers.__version__,
                             'tokenizers': __import__('tokenizers').__version__,
                             'datasets': __import__('datasets').__version__},
        },
        'funnel': meta['funnel'],
        'score': {'name': 'text_form_score', 'cut': round(meta['score_cut'], 4),
                  'description': SCORE_DESC},
        'refills': {name: [{'q_md5': m, 'reason': '确定性补充被剔除/失效的成员，需人工审核'}
                           for m in ms] for name, ms in meta['refills'].items()},
        'near_duplicates': {'candidates': meta['near_dup_count'],
                            'path': 'audit/near_duplicates.jsonl',
                            'note': '候选需人工判断；同属一种疾病不等于重复'},
        'files': files_meta,
        'disclaimers': [SCORE_DESC, EOS_DESC,
                        '训练集仅经过形式筛选与抽查审核，未全量人工审核，不宣称医学质量全部通过'],
    }
    mp = out_dir / 'manifest.json'
    tmp = mp.with_suffix('.tmp')
    tmp.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    os.replace(tmp, mp)

    # 全量验收
    print('===== 全量验收 =====')
    all_ok, reports = run_acceptance(out_dir, args, meta, near_dup)
    all_ok &= same
    for r in reports:
        print(r)

    # 汇总
    print('\n===== 漇斗 =====')
    for st in meta['funnel']:
        print(f"  {st['stage']}: {st['count']}  ({st['reason']})")
    print(f"补充成员: val {len(meta['refills']['val'])} 个, test {len(meta['refills']['test'])} 个"
          f"（需人工审核，见 manifest.refills）")
    print(f"近重复候选: {meta['near_dup_count']} 对（人工判断后如需排除写入 exclusions.jsonl，exclude_from=train）")
    for name, fm in files_meta.items():
        print(f"  {name}: {fm['count']} 条, {fm['bytes'] / 1024:.1f} KB, md5={fm['md5'][:12]}...")
    print('\n总体结果:', 'ALL PASS' if all_ok else 'EXISTS FAIL')
    sys.exit(0 if all_ok else 1)


if __name__ == '__main__':
    main()
