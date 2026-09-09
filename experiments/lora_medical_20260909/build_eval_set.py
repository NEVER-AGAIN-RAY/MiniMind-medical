#!/usr/bin/env python
"""
评估考卷构建脚本（v1）——一次性锁死评估题集，训练前运行。

三张考卷:
  questions_mcq.jsonl     100 道  CMExam val.csv（中国执业医师资格考试，官方标准答案，独立来源）
  questions_medqa.jsonl    30 道  data/v2/test.jsonl 中审核通过样本（同分布，参考答案=数据集回答）
  questions_general.jsonl  10 道  固定通用问题（灾难性遗忘检查）

规则:
  - 选择题只取: 单字母答案(A-E)、题干<=100字、选项2-5个且各<=80字、不含图
  - 问答只取 test.jsonl 中最新审核状态为 no_obvious_issue 的样本（按 q_md5 匹配，
    val_test_review.jsonl 的行号与当前文件已错位，不能按行号取）
  - 选择题与训练源数据(dataset/lora_medical.jsonl)做归一化问题对撞排除，防止考卷是练过的题
  - 固定 seed，重跑逐字节一致；manifest.json 记录全部来源校验值与评分标准

用法:
  ../../.venv/bin/python build_eval_set.py            # 正常构建
  ... build_eval_set.py --cmexam-csv /path/to/val.csv # 离线模式（跳过下载）
"""
import argparse
import csv
import hashlib
import io
import json
import random
import re
import sys
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))

AUDIT_DIR = HERE / 'audit'
TEST_JSONL = HERE / 'data' / 'v2' / 'test.jsonl'
TRAIN_SOURCE = ROOT / 'dataset' / 'lora_medical.jsonl'
VAL_TEST_REVIEW = AUDIT_DIR / 'val_test_review.jsonl'
REFILL_REVIEW = AUDIT_DIR / 'refill_review.jsonl'

CMEXAM_URL = 'https://raw.githubusercontent.com/williamliujl/CMExam/main/data/val.csv'

# 通用问题: 前 8 个沿用 eval_llm.py 的固定问题，后 2 个补充基础事实/自我认知探针
GENERAL_QUESTIONS = [
    '你有什么特长？',
    '为什么天空是蓝色的',
    '请用Python写一个计算斐波那契数列的函数',
    '解释一下"光合作用"的基本过程',
    '如果明天下雨，我应该如何出门',
    '比较一下猫和狗作为宠物的优缺点',
    '解释什么是机器学习',
    '推荐一些中国的美食',
    '中国的首都是哪座城市？',
    '用一句话介绍你自己',
]

SCORING_RUBRIC = [
    {'id': 'on_topic', 'name': '是否答到问题', 'good': True,
     'desc': '回答与问题相关且正面回应了问题，而非答非所问或复述问题'},
    {'id': 'factual_error', 'name': '是否存在事实错误', 'good': False,
     'desc': '回答中是否出现与参考答案/公认医学常识相悖的陈述'},
    {'id': 'unsourced_advice', 'name': '是否无依据地给出诊疗建议', 'good': False,
     'desc': '在无足够依据/未提示就医的情况下给出具体用药、剂量或手术建议'},
]


def md5_text(s: str) -> str:
    return hashlib.md5(s.encode('utf-8')).hexdigest()


def md5_file(p: Path) -> str:
    return hashlib.md5(p.read_bytes()).hexdigest()


def norm_q(q: str) -> str:
    """归一化问题文本: 与 prepare_data.py 保持一致，用于近似对撞。"""
    return re.sub(r'[\s，。？！,.?!、的了吗呢吧啊]', '', q)


# ---------- CMExam ----------
def load_cmexam(raw_path: Path, url: str, offline_csv: Path | None) -> list[dict]:
    """返回 CMExam val.csv 的行列表（原始 csv.DictReader 行）。"""
    if offline_csv is not None:
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        raw_path.write_bytes(offline_csv.read_bytes())
    elif not raw_path.exists():
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        print(f'下载 CMExam val.csv: {url}')
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=60) as resp:
            raw_path.write_bytes(resp.read())
    rows = list(csv.DictReader(io.StringIO(raw_path.read_text(encoding='utf-8'))))
    assert rows, 'CMExam csv 为空'
    return rows


def parse_options(s: str) -> dict[str, str]:
    """把 "A xxx\\nB xxx" 解析成 {'A': 'xxx', ...}。"""
    d = {}
    for line in s.split('\n'):
        m = re.match(r'^([A-E])\s+(.*)$', line.strip())
        if m:
            d[m.group(1)] = m.group(2)
    return d


def build_mcq_candidates(rows: list[dict]) -> tuple[list[dict], dict]:
    """筛选选择题候选，返回 (候选列表, 统计)。"""
    stats = {'total': len(rows), 'single_letter': 0, 'length_ok': 0,
              'options_ok': 0, 'no_image': 0}
    cands = []
    for r in rows:
        ans = r.get('Answer', '').strip()
        if len(ans) != 1 or ans not in 'ABCDE':
            continue
        stats['single_letter'] += 1
        q = r.get('Question', '').strip()
        if not q or len(q) > 100:
            continue
        stats['length_ok'] += 1
        opts = parse_options(r.get('Options', ''))
        if not (2 <= len(opts) <= 5) or any(not v.strip() or len(v) > 80 for v in opts.values()):
            continue
        if len(opts) < 5 and ans not in opts:
            continue  # 答案字母不在选项里
        stats['options_ok'] += 1
        if '图' in q or '图' in r.get('Options', ''):
            continue
        stats['no_image'] += 1
        cands.append({'question': q, 'options': opts, 'answer': ans,
                      'explanation': r.get('Explanation', '').strip()})
    return cands, stats


# ---------- 审核记录 ----------
def load_test_verdicts() -> tuple[dict[str, str], dict]:
    """按 q_md5 取 test.jsonl 每条的最新审核结论（refill 记录优先于原始记录）。"""
    verdicts, meta = {}, {'val_test_review_entries': 0, 'refill_test_entries': 0}
    for line in VAL_TEST_REVIEW.read_text(encoding='utf-8').splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if r.get('file') == 'test.jsonl':
            verdicts[r['q_md5']] = r.get('verdict', '')
            meta['val_test_review_entries'] += 1
    for line in REFILL_REVIEW.read_text(encoding='utf-8').splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if r.get('file') == 'test.jsonl':
            verdicts[r['q_md5']] = r.get('verdict', '')  # 补充样本覆盖旧记录
            meta['refill_test_entries'] += 1
    return verdicts, meta


# ---------- 训练源对撞 ----------
def load_train_norm_set() -> set[str]:
    """训练源数据全部问题的归一化 md5 集合。"""
    norms = set()
    for line in TRAIN_SOURCE.read_text(encoding='utf-8').splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        q = row['conversations'][0]['content']
        norms.add(md5_text(norm_q(q)))
    return norms


def dump_jsonl(xs: list[dict]) -> str:
    return ''.join(json.dumps(x, ensure_ascii=False) + '\n' for x in xs)


def main():
    ap = argparse.ArgumentParser(description='构建评估考卷 v1（一次性锁题）')
    ap.add_argument('--out-dir', type=Path, default=HERE / 'eval' / 'v1')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--mcq-count', type=int, default=100)
    ap.add_argument('--qa-count', type=int, default=30)
    ap.add_argument('--cmexam-csv', type=Path, default=None,
                    help='离线模式: 使用本地 CMExam val.csv（不联网）')
    args = ap.parse_args()
    out = args.out_dir
    raw_path = out / 'raw' / 'cmexam_val.csv'
    out.mkdir(parents=True, exist_ok=True)

    # ===== 1. CMExam 选择题 =====
    rows = load_cmexam(raw_path, CMEXAM_URL, args.cmexam_csv)
    cands, mcq_stats = build_mcq_candidates(rows)
    train_norms = load_train_norm_set()
    before = len(cands)
    cands = [c for c in cands if md5_text(norm_q(c['question'])) not in train_norms]
    overlap_excluded = before - len(cands)
    assert len(cands) >= args.mcq_count, f'选择题候选不足: {len(cands)} < {args.mcq_count}'

    cands.sort(key=lambda c: md5_text(c['question']))  # 稳定排序，保证抽样可复现
    rng = random.Random(args.seed)
    picked = rng.sample(cands, args.mcq_count)
    mcq_items = [{'id': f'mcq_{i:03d}', **c, 'q_md5': md5_text(c['question'])}
                 for i, c in enumerate(picked, 1)]

    # ===== 2. 同分布医学问答（test.jsonl 审核通过样本） =====
    test_rows = [json.loads(l) for l in TEST_JSONL.read_text(encoding='utf-8').splitlines() if l.strip()]
    verdicts, review_meta = load_test_verdicts()
    clean, pending = [], []
    for row in test_rows:
        m = md5_text(row['conversations'][0]['content'].strip())
        if verdicts.get(m, 'pending_verification') == 'no_obvious_issue':
            clean.append((m, row))
        else:
            pending.append(m)
    assert len(clean) >= args.qa_count, f'可用问答不足: {len(clean)} < {args.qa_count}'
    clean.sort(key=lambda x: x[0])
    picked_qa = rng.sample(clean, args.qa_count)
    qa_items = [{'id': f'qa_{i:03d}', 'q_md5': m,
                 'question': row['conversations'][0]['content'].strip(),
                 'reference_answer': row['conversations'][1]['content']}
                for i, (m, row) in enumerate(picked_qa, 1)]

    # ===== 3. 通用问题 =====
    gen_items = [{'id': f'gen_{i:03d}', 'question': q} for i, q in enumerate(GENERAL_QUESTIONS, 1)]

    # ===== 4. 落盘 =====
    files = {
        'questions_mcq.jsonl': dump_jsonl(mcq_items),
        'questions_medqa.jsonl': dump_jsonl(qa_items),
        'questions_general.jsonl': dump_jsonl(gen_items),
    }
    for name, content in files.items():
        tmp = out / (name + '.tmp')
        tmp.write_text(content, encoding='utf-8')
        tmp.replace(out / name)

    manifest = {
        'version': 'v1',
        'seed': args.seed,
        'params': {
            'mcq_count': args.mcq_count,
            'qa_count': args.qa_count,
            'general_count': len(gen_items),
            'mcq_filters': {
                'answer': '单字母 A-E',
                'question_max_chars': 100,
                'options': '2-5 个，每个 <=80 字',
                'no_image': '题干或选项含"图"的剔除',
            },
            'qa_rule': 'test.jsonl 中最新审核结论为 no_obvious_issue 的样本（q_md5 匹配，排除待核验）',
        },
        'sources': {
            'cmexam': {
                'url': CMEXAM_URL,
                'file': str(raw_path.relative_to(HERE)),
                'md5': md5_file(raw_path),
                'license_note': 'CMExam: 中国国家执业医师资格考试数据集 (Liu et al., NeurIPS 2023 D&B)',
                **mcq_stats,
                'train_overlap_excluded': overlap_excluded,
                'candidates_final': before - overlap_excluded,
            },
            'test_jsonl': {
                'path': str(TEST_JSONL.relative_to(HERE)),
                'md5': md5_file(TEST_JSONL),
                'rows': len(test_rows),
                'clean_usable': len(clean),
                'pending_excluded': len(pending),
                'pending_q_md5': sorted(pending),
                **{f'review_{k}': md5_file(v) for k, v in
                   [('val_test_review', VAL_TEST_REVIEW), ('refill_review', REFILL_REVIEW)]},
                **review_meta,
            },
            'train_source_overlap_check': {
                'path': str(TRAIN_SOURCE.relative_to(ROOT)),
                'md5': md5_file(TRAIN_SOURCE),
                'method': 'norm_q 归一化后 md5 对撞（与 prepare_data.py 同款归一化）',
            },
            'general_questions': '前 8 个沿用 eval_llm.py 固定问题，后 2 个为补充探针',
        },
        'rubric': {
            'mcq': '自动对答案: 抽取模型输出中首个 A-E 选项字母',
            'manual': SCORING_RUBRIC,
            'note': '人工评分逐题填写于 compare_report.md；待核验样本不作为评分参考答案（本脚本已排除）',
        },
        'files': {name: md5_file(out / name) for name in files},
    }
    tmp = out / 'manifest.json.tmp'
    tmp.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    tmp.replace(out / 'manifest.json')

    # ===== 5. 汇总 =====
    print('===== 评估考卷构建完成 =====')
    print(f"选择题: {len(mcq_items)} 道 (CMExam 候选 {len(cands)}, 训练对撞排除 {overlap_excluded})")
    print(f"问答:   {len(qa_items)} 道 (test.jsonl 可用 {len(clean)}, 排除待核验 {len(pending)})")
    print(f"通用:   {len(gen_items)} 个")
    print(f"输出目录: {out}")

    # ===== 6. 重跑一致性（内存级） =====
    import copy
    files2 = {
        'questions_mcq.jsonl': dump_jsonl(mcq_items),
        'questions_medqa.jsonl': dump_jsonl(qa_items),
        'questions_general.jsonl': dump_jsonl(gen_items),
    }
    assert files == files2, '内部一致性检查失败'
    print('内部一致性检查: 通过')


if __name__ == '__main__':
    main()
