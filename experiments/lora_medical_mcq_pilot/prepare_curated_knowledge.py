#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
构建"医学选择题精简知识训练集" curated_knowledge_v1。

只做数据筛选与清洗，不训练、不推理、不修改任何已有数据文件：
- 训练候选仅来自 data/raw/train.csv；
- raw val/test、data/formal/、data/knowledge_5k_v1/、历史评测题
  (experiments/lora_medical_20260909/eval/v1/) 与历史 QA 数据
  (dataset/lora_medical.jsonl) 全部仅用于排除；
- 训练目标格式："{正确选项文字}。知识点：{一句来自原解析的正向医学事实}。"
- 知识句只做标记清理，不编造医学内容；无法可靠提取知识句的题目直接排除。
- 随机种子固定为 42，全部流程确定性可复现。

输出目录: data/curated_knowledge_v1/
  train.jsonl / dev.jsonl / human_review_100.jsonl / manifest.json / rejected_stats.json
"""

import argparse
import csv
import hashlib
import json
import random
import re
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]

SEED = 42
ANSWER_LETTERS = ("A", "B", "C", "D", "E")
NEAR_DUP_JACCARD_THRESHOLD = 0.85  # 与 config.json data_config.deduplication 保持一致
TRAIN_TARGET = 5000
DEV_TARGET = 400
REVIEW_TARGET = 100
MAX_ANSWER_TEXT_CHARS = 30
MIN_KNOWLEDGE_CHARS = 8
MAX_KNOWLEDGE_CHARS = 100
MIN_QUESTION_CHARS = 6
MIN_EXPLANATION_CORE_CHARS = 6

OUTPUT_DIR = HERE / "data" / "curated_knowledge_v1"

# ---------------------------------------------------------------------------
# 基础文本工具（norm_q 与 prepare_data.py 完全一致，保证归一化口径统一）
# ---------------------------------------------------------------------------


def norm_q(q: str) -> str:
    """文本严格归一化：剥除标点、空白及停用语助词，提取核心语义骨架"""
    return re.sub(r"[\s，。？！,.?!、的了吗呢吧啊：:；;\-—_（）\(\)【】\[\]\"'“”‘’于在中为是有的]", "", q)


def compact(s: str) -> str:
    return re.sub(r"\s+", "", s)


def md5_text(s: str) -> str:
    return hashlib.md5(s.encode("utf-8")).hexdigest()


def md5_file(path: Path) -> str:
    hasher = hashlib.md5()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def question_bigrams(q: str) -> set:
    cleaned = norm_q(q)
    if len(cleaned) <= 2:
        return {cleaned} if cleaned else set()
    return {cleaned[i : i + 2] for i in range(len(cleaned) - 1)}


class NearDuplicateIndex:
    """基于字符 2-gram Jaccard 相似度的近重复检测倒排索引。"""

    def __init__(self, threshold: float = NEAR_DUP_JACCARD_THRESHOLD):
        self.threshold = threshold
        self._sets: dict[int, set] = {}
        self._inv: dict[str, list[int]] = {}
        self._next = 0

    def add(self, text: str) -> None:
        idx = self._next
        self._next += 1
        bgs = question_bigrams(text)
        self._sets[idx] = bgs
        for bg in bgs:
            self._inv.setdefault(bg, []).append(idx)

    def query(self, text: str) -> bool:
        """返回 True 表示存在近重复（Jaccard >= threshold）。"""
        bgs = question_bigrams(text)
        if not bgs:
            return False
        counts: Counter = Counter()
        for bg in bgs:
            for idx in self._inv.get(bg, ()):
                counts[idx] += 1
        for idx, c in counts.items():
            other = self._sets[idx]
            if c / max(len(bgs), len(other)) >= self.threshold:
                j = c / (len(bgs) + len(other) - c)
                if j >= self.threshold:
                    return True
        return False


# ---------------------------------------------------------------------------
# 判别规则（否定题 / 组合选项 / 图片表格 / 乱码 / 截断）
# ---------------------------------------------------------------------------

NEGATIVE_STEM_MARKERS = (
    "错误的是", "错误的", "不正确", "不包括", "不属于", "除外", "不是",
    "不符合", "无关", "没有的是", "不出现", "不发生", "不引起", "不宜",
    "不能", "禁止", "不得", "禁用", "不可能", "不常见", "不需要", "无助于",
)
EXCLUDE_STEM_RE = re.compile(r"除.{0,10}外")
EXPLANATION_NEGATIVE_RE = re.compile(r"错[，,]?\s*为(?:本题|该题)(?:的)?(?:正确|最佳)?答案")

COMBO_PHRASES = (
    "以上都对", "以上都不对", "以上均是", "以上均不是", "以上都不是",
    "以上都是", "以上均正确", "以上均错误", "以上各项均", "以上皆",
    "上述均是", "上述都不是", "其余均", "各项均正确", "各项均不正确",
    "全部正确", "全部错误", "均正确", "均不正确", "均错误", "均不是",
    "都不是", "都是", "都对",
)
COMBO_LETTER_RE = re.compile(r"^[A-E](?:\s*[和与、及＋+]\s*[A-E])+$")

VISUAL_RE = re.compile(
    r"如图|见图|下图|上图|图中|图示|附图|下表(?!面)|上表(?!面)|如表|见表|表中|表格|附表|处方|照片"
)
GARBAGE_RE = re.compile(r"[\ufffd□]|<[^>\n]{1,80}>|&[a-zA-Z]+;|[\x00-\x08\x0b\x0c\x0e-\x1f]")
TRUNCATION_TAIL_CHARS = "，,、：:；;…"

# 解析中的判题标记
PAREN_JUDGMENT_RE = re.compile(r"[（(]\s*([A-E])\s*(对|错|正确|错误)(?:[，,][^（）()]{0,20})?[）)]")
PAREN_LETTER_RE = re.compile(r"[（(]\s*[A-E]\s*[）)]")
JUDGMENT_MARKER_SEARCH = re.compile(
    r"(?<=[（(，,、；。：:！？\s])[A-E](?:对|错|正确|错误)(?=[）)，,、；。：:！？\s]|$)"
)
JUDGMENT_CLAUSE_RE = re.compile(r"^[A-E](?:对|错|正确|错误)$")
ANSWER_PHRASE_LETTERS_RE = re.compile(
    r"(?:正确答案|本题答案|答案)(?:是|为|应为|应选|当选|选|：|:)?\s*([A-E])(?![A-Za-z0-9])"
)
LEFTOVER_MARKER_RE = re.compile(r"本题|正确答案|故选|应选|答案")


def is_negative_question(question: str, explanation: str) -> bool:
    if any(p in question for p in NEGATIVE_STEM_MARKERS):
        return True
    if EXCLUDE_STEM_RE.search(question):
        return True
    if EXPLANATION_NEGATIVE_RE.search(explanation):
        return True
    return False


def is_combination_option(option_text: str) -> bool:
    if any(p in option_text for p in COMBO_PHRASES):
        return True
    if COMBO_LETTER_RE.match(option_text.strip()):
        return True
    return False


def has_judgment_marker(text: str) -> bool:
    """检测标点边界上的“A对 / B错 / C正确”等判题标记。

    通过标点/括号边界限定，避免误伤“维生素A对视觉有益”这类正文。
    """
    return bool(JUDGMENT_MARKER_SEARCH.search(" " + text + " "))


def extract_judgment_letters(explanation: str) -> tuple[set, set]:
    """从解析中提取 (被判为对的字母集合, 被判为错的字母集合)。"""
    dui, cuo = set(), set()
    for letter, judgement in PAREN_JUDGMENT_RE.findall(explanation):
        (dui if judgement in ("对", "正确") else cuo).add(letter)
    for clause in re.split(r"[，,、；;。．！!？?\n]", explanation):
        clause = clause.strip()
        m = JUDGMENT_CLAUSE_RE.match(clause)
        if m:
            judgement = clause[1:]
            (dui if judgement in ("对", "正确") else cuo).add(clause[0])
    return dui, cuo


def is_conflicted_explanation(explanation: str, answer: str) -> bool:
    """解析与正确答案冲突：对标记字母不是唯一答案、答案被标错、答案字母短语不一致。"""
    dui, cuo = extract_judgment_letters(explanation)
    if dui and dui != {answer}:
        return True
    if answer in cuo:
        return True
    for letter in ANSWER_PHRASE_LETTERS_RE.findall(explanation):
        if letter != answer:
            return True
    return False


def parse_options_strict(options_raw: str) -> dict | None:
    """严格解析选项：必须恰好解析出 A-E 五个非空选项，否则返回 None。"""
    opts: dict[str, str] = {}
    for line in options_raw.split("\n"):
        line = line.strip()
        if not line:
            continue
        m = re.match(r"^([A-Ea-eＡ-Ｅ])[\s.、:：]+(.*)$", line)
        if not m:
            return None
        letter = m.group(1).upper()
        text = m.group(2).strip()
        if letter in opts or not text:
            return None
        opts[letter] = text
    if set(opts) != set(ANSWER_LETTERS):
        return None
    return {L: opts[L] for L in ANSWER_LETTERS}


# ---------------------------------------------------------------------------
# 知识句清理与抽取
# ---------------------------------------------------------------------------

META_PREFIX_RE = re.compile(r"^本题(?:主要)?(?:考查|考核|考察)(?:的是|了)?")
CONNECTOR_RE = re.compile(r"^(?:所以|因此|故|因而|故此)")
PAREN_MARKER_RE = re.compile(r"[（(]\s*[A-E]\s*(?:对|错|正确|错误)(?:[，,][^（）()]{0,20})?[）)]")
PAREN_MARKER_COMBO_RE = re.compile(r"[（(]\s*(?:[A-E]\s*(?:对|错|正确|错误)\s*){1,8}[）)]")
COMBO_JUDGMENT_RE = re.compile(r"[A-E](?:对|错|正确|错误)[A-E](?:对|错|正确|错误)")
PAREN_CONTENT_RE = re.compile(r"[（(]([^（）()]{1,40})[）)]")
PAREN_INNER_CLASS_RE = re.compile(r"^[A-Ea-e对错正确错误均都，,、\s]+$")


def _strip_paren_judgment_blocks(text: str) -> str:
    """删除仅由字母、判词(对/错/正确/错误)与分隔符构成的括号块。

    覆盖 CMExam 常见标注：如 （E对BCD错）、（ACDE错）、（A对，为本题正确答案）。
    含其他正文文字的括号（如“对乙酰氨基酚”“A型”）不受影响。
    """
    def repl(m):
        inner = m.group(1)
        if PAREN_INNER_CLASS_RE.match(inner) and re.search(r"对|错|正确|错误", inner):
            return ""
        return m.group(0)
    return PAREN_CONTENT_RE.sub(repl, text)
PHRASE_REMOVAL_RES = (
    re.compile(r"(?:故|因此|所以|综上)?本题(?:的)?(?:正确答案|答案)(?:是|为|应为|应选)?\s*[A-E](?![A-Za-z0-9])"),
    re.compile(r"(?:故|因此|所以)\s*(?:本题)?(?:应)?选\s*[A-E](?![A-Za-z0-9])"),
    re.compile(r"(?:正确答案|答案)(?:是|为|应为|应选)?\s*[A-E](?![A-Za-z0-9])"),
    re.compile(r"为本题(?:的)?(?:正确|最佳)?答案"),
    re.compile(r"为本题(?:的)?正确选项"),
    re.compile(r"(?:正确答案|本题答案|答案)(?:是|为|应为|应选|当选)"),
)
KNOWLEDGE_NEGATIVE_RE = re.compile(
    "不是|不属于|不能|不会|不可|不宜|无法|没有|缺乏|除外|错误|而非|而不是"
    "|不正确|不符合|不出现|不发生|不引起|不包括|不存在|并无|以上"
)

# 知识句抽取失败的细分原因
FAIL_NO_CANDIDATE = "no_knowledge_sentence"
FAIL_WRONG_OPTIONS = "knowledge_mentions_wrong_options"
FAIL_NEGATIVE = "knowledge_negative_statement"
FAIL_MARKER = "knowledge_marker_remaining"
_FAIL_PRIORITY = (FAIL_NEGATIVE, FAIL_WRONG_OPTIONS, FAIL_MARKER, FAIL_NO_CANDIDATE)


def split_sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[。！？!?])", text)
    return [p.strip() for p in parts if p.strip()]


def split_clauses(sentence: str) -> list[str]:
    parts = re.split(r"(?<=[，、；;：:,])", sentence)
    return [p.strip() for p in parts if p.strip()]


def clean_fragment(fragment: str) -> str:
    """清理判题标记与元话语，仅做删除、不改写、不新增内容。"""
    text = fragment.strip()
    for _ in range(3):
        text = PAREN_MARKER_RE.sub("", text)
        text = PAREN_MARKER_COMBO_RE.sub("", text)
        text = _strip_paren_judgment_blocks(text)
    for pattern in PHRASE_REMOVAL_RES:
        text = pattern.sub("", text)
    text = META_PREFIX_RE.sub("", text)
    text = CONNECTOR_RE.sub("", text)
    # 丢弃纯判题短句（“A对”“B错”等独立小句）
    pieces = re.split(r"([，,、；;])", text)
    kept = [p for i, p in enumerate(pieces) if i % 2 == 1 or not JUDGMENT_CLAUSE_RE.match(p.strip())]
    text = "".join(kept)
    return text.strip().strip("，,、；;：:。．!！?？ ")


def _try_fragment(fragment: str, answer: str, options: dict) -> tuple[str | None, str]:
    """对单个句子/小句尝试提取知识句，返回 (知识句, 失败原因)。"""
    cleaned = clean_fragment(fragment)
    compact_answer = compact(options[answer])
    if not cleaned or compact_answer not in compact(cleaned):
        return None, FAIL_NO_CANDIDATE
    if len(compact(cleaned)) < len(compact_answer) + 2:
        return None, FAIL_NO_CANDIDATE
    if not MIN_KNOWLEDGE_CHARS <= len(cleaned) <= MAX_KNOWLEDGE_CHARS:
        return None, FAIL_NO_CANDIDATE
    if has_judgment_marker(cleaned) or PAREN_LETTER_RE.search(cleaned) \
            or COMBO_JUDGMENT_RE.search(cleaned) or LEFTOVER_MARKER_RE.search(cleaned):
        return None, FAIL_MARKER
    compact_clean = compact(cleaned)
    wrong_discussed = sum(
        1 for L in ANSWER_LETTERS
        if L != answer
        and compact(options[L]) not in compact_answer  # 选项文字是答案文字的子串时不算讨论错误选项
        and compact(options[L]) in compact_clean
    )
    if wrong_discussed >= 2:
        return None, FAIL_WRONG_OPTIONS
    if KNOWLEDGE_NEGATIVE_RE.search(cleaned):
        return None, FAIL_NEGATIVE
    return cleaned, ""


def extract_knowledge(explanation: str, answer: str, options: dict) -> tuple[str | None, str]:
    """从原解析中提取一句包含正确答案的正向医学陈述。

    先整句尝试，失败后按小句（顿号/逗号层级）再试；全部失败时按
    预设优先级返回最具代表性的失败原因，保证结果确定。
    """
    compact_answer = compact(options[answer])
    sentences = split_sentences(explanation)
    fail_counter: Counter = Counter()

    def scan(units) -> str | None:
        for unit in units:
            if compact_answer not in compact(unit):
                continue
            knowledge, fail = _try_fragment(unit, answer, options)
            if knowledge is not None:
                return knowledge
            fail_counter[fail] += 1
        return None

    knowledge = scan(sentences) or scan(cl for s in sentences for cl in split_clauses(s))
    if knowledge is not None:
        return knowledge, ""
    if fail_counter:
        for reason in _FAIL_PRIORITY:
            if fail_counter[reason]:
                return None, reason
    return None, FAIL_NO_CANDIDATE


def compose_target(answer_text: str, knowledge: str) -> str:
    a = answer_text.strip().rstrip("。．.！!？?")
    k = knowledge.strip().rstrip("。．.，,、；;！!？? ")
    return f"{a}。知识点：{k}。"


# ---------------------------------------------------------------------------
# 外部排除源加载
# ---------------------------------------------------------------------------


def default_external_sources() -> list[tuple[str, Path, str]]:
    """(来源标签, 文件路径, 加载方式)。加载方式: jsonl_question / conversations / csv_question"""
    return [
        ("formal_train", HERE / "data" / "formal" / "train.jsonl", "jsonl_question"),
        ("formal_val", HERE / "data" / "formal" / "val.jsonl", "jsonl_question"),
        ("formal_test", HERE / "data" / "formal" / "test.jsonl", "jsonl_question"),
        ("knowledge_5k_train", HERE / "data" / "knowledge_5k_v1" / "train.jsonl", "jsonl_question"),
        ("knowledge_5k_val", HERE / "data" / "knowledge_5k_v1" / "val.jsonl", "jsonl_question"),
        ("eval_v1_mcq", ROOT / "experiments" / "lora_medical_20260909" / "eval" / "v1" / "questions_mcq.jsonl", "jsonl_question"),
        ("eval_v1_general", ROOT / "experiments" / "lora_medical_20260909" / "eval" / "v1" / "questions_general.jsonl", "jsonl_question"),
        ("eval_v1_medqa", ROOT / "experiments" / "lora_medical_20260909" / "eval" / "v1" / "questions_medqa.jsonl", "jsonl_question"),
        ("hist_qa_lora_medical", ROOT / "dataset" / "lora_medical.jsonl", "conversations"),
        ("raw_val", HERE / "data" / "raw" / "val.csv", "csv_question"),
        ("raw_test", HERE / "data" / "raw" / "test.csv", "csv_question"),
    ]


def load_external_question_texts(sources: list[tuple[str, Path, str]]) -> tuple[dict[str, list[str]], list[str], dict[str, str]]:
    """读取全部外部排除源的问题原文。

    返回 (按来源的问题文本列表, 缺失来源标签, 输入文件MD5)。
    """
    by_source: dict[str, list[str]] = {}
    missing: list[str] = []
    input_md5: dict[str, str] = {}
    for label, path, kind in sources:
        if not path.is_file():
            missing.append(label)
            continue
        input_md5[str(path)] = md5_file(path)
        questions: list[str] = []
        if kind == "csv_question":
            with open(path, encoding="utf-8", errors="replace") as f:
                for row in csv.DictReader(f):
                    q = (row.get("Question") or "").strip()
                    if q:
                        questions.append(q)
        else:
            with open(path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    item = json.loads(line)
                    q = item.get("question", "")
                    if not q and kind == "conversations":
                        convs = item.get("conversations", [])
                        q = convs[0].get("content", "") if convs else ""
                    if q:
                        questions.append(q)
        by_source[label] = questions
    return by_source, missing, input_md5


def load_raw_train(path: Path) -> list[dict]:
    with open(path, encoding="utf-8", errors="replace") as f:
        return list(csv.DictReader(f))


# ---------------------------------------------------------------------------
# 单条题目筛选
# ---------------------------------------------------------------------------


def screen_row(row: dict, row_idx: int) -> tuple[str | None, dict | None, str]:
    """筛选单行数据。返回 (排除原因 or None, 候选记录 or None, 知识句 or '')。"""
    question = (row.get("Question") or "").strip()
    options_raw = row.get("Options") or ""
    answer = (row.get("Answer") or "").strip().upper()
    explanation = (row.get("Explanation") or "").strip()

    if not question or not options_raw or not answer:
        return "missing_or_incomplete_fields", None, ""
    options = parse_options_strict(options_raw)
    if options is None:
        return "invalid_options_format", None, ""
    if not re.fullmatch(r"[A-E]", answer) or answer not in options:
        return "answer_not_in_options", None, ""
    if is_negative_question(question, explanation):
        return "negative_question", None, ""
    answer_text = options[answer]
    if any(is_combination_option(t) for t in options.values()):
        return "combination_options", None, ""
    if (len(answer_text) > MAX_ANSWER_TEXT_CHARS
            or re.search(r"[。．.!！?？；;]", answer_text)
            or PAREN_LETTER_RE.search(answer_text)):
        return "answer_text_too_long_or_multi_sentence", None, ""
    if any(VISUAL_RE.search(x) for x in [question, options_raw, explanation]):
        return "visual_or_table_dependency", None, ""
    if any(GARBAGE_RE.search(x) for x in [question, options_raw, explanation]):
        return "garbled_or_html", None, ""
    if question[-1] in TRUNCATION_TAIL_CHARS:
        return "question_truncated", None, ""
    if explanation == "请等待更新" or len(explanation) < 10:
        return "explanation_missing_or_empty", None, ""
    core = clean_fragment(explanation)
    if len(re.sub(r"[\s，,、；;：:。．()（）]", "", core)) < MIN_EXPLANATION_CORE_CHARS:
        return "explanation_insufficient_medical_basis", None, ""
    if is_conflicted_explanation(explanation, answer):
        return "explanation_answer_conflict", None, ""
    if compact(answer_text) not in compact(explanation):
        return "answer_absent_from_explanation", None, ""
    knowledge, knowledge_fail = extract_knowledge(explanation, answer, options)
    if knowledge is None:
        return knowledge_fail, None, ""

    record = {
        "question": question,
        "options": options,
        "answer": answer,
        "answer_text": answer_text,
        "explanation": explanation,
        "source_row": row_idx,
    }
    return None, record, knowledge


# ---------------------------------------------------------------------------
# 构建、切分与写出
# ---------------------------------------------------------------------------


def make_record(base: dict, knowledge: str, prefix: str, seq: int, source: str) -> dict:
    record = {
        "id": f"{prefix}_{seq:05d}",
        "question": base["question"],
        "options": base["options"],
        "answer": base["answer"],
        "answer_text": base["answer_text"],
        "concise_knowledge": knowledge,
        "target_text": compose_target(base["answer_text"], knowledge),
        "source": source,
        "source_row": base["source_row"],
        "explanation": base["explanation"],
        "norm_hash": md5_text(norm_q(base["question"])),
    }
    return record


def stratified_split(pool: list[dict], seed: int, train_target: int, dev_target: int):
    """先 dev 后 train，按答案字母分层，各字母上限均分，保证字母均衡。"""
    rng = random.Random(seed)
    shuffled = list(pool)
    rng.shuffle(shuffled)
    by_letter: dict[str, list[dict]] = {L: [] for L in ANSWER_LETTERS}
    for item in shuffled:
        by_letter[item["answer"]].append(item)
    dev_per_letter = max(1, -(-dev_target // len(ANSWER_LETTERS)))  # ceil
    train_per_letter = max(1, train_target // len(ANSWER_LETTERS))
    dev, train = [], []
    for L in ANSWER_LETTERS:
        items = by_letter[L]
        dev.extend(items[:dev_per_letter])
        train.extend(items[dev_per_letter : dev_per_letter + train_per_letter])
    dev.sort(key=lambda r: r["source_row"])
    train.sort(key=lambda r: r["source_row"])
    train = [make_record(r, r.pop("_knowledge"), "curated_train", i + 1, "CMExam_train") for i, r in enumerate(train)]
    dev = [make_record(r, r.pop("_knowledge"), "curated_dev", i + 1, "CMExam_train") for i, r in enumerate(dev)]
    return train, dev


def build_review(train: list[dict], dev: list[dict], seed: int, count: int) -> list[dict]:
    population = sorted(train + dev, key=lambda r: (r["id"].split("_")[1], r["id"]))
    rng = random.Random(seed)
    take = min(count, len(population))
    picked = rng.sample(population, take)
    picked.sort(key=lambda r: (r["id"].split("_")[1], r["id"]))
    review = []
    for r in picked:
        item = dict(r)
        item["split"] = "train" if r["id"].startswith("curated_train") else "dev"
        review.append(item)
    return review


def write_jsonl(path: Path, records: list[dict]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for item in records:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def length_stats(records: list[dict], key: str) -> dict:
    values = [len(r[key]) for r in records]
    if not values:
        return {"min": 0, "max": 0, "mean": 0.0}
    return {
        "min": min(values),
        "max": max(values),
        "mean": round(sum(values) / len(values), 2),
    }


# ---------------------------------------------------------------------------
# 产物验证（对应任务“验证要求”1-8；第9条确定性由测试用例覆盖）
# ---------------------------------------------------------------------------

TARGET_FORBIDDEN_RE = re.compile(r"本题|正确答案|故选|应选|答案为|答案是")


def validate_records(records: list[dict], split_name: str) -> list[str]:
    problems = []
    for r in records:
        rid = r.get("id", "?")
        if set(r.get("options", {})) != set(ANSWER_LETTERS):
            problems.append(f"[{split_name}/{rid}] 选项键不完整")
            continue
        answer = r.get("answer", "")
        if answer not in ANSWER_LETTERS or answer not in r["options"]:
            problems.append(f"[{split_name}/{rid}] 答案字母不在options中: {answer}")
            continue
        if r.get("answer_text") != r["options"][answer]:
            problems.append(f"[{split_name}/{rid}] answer_text与正确选项不一致")
        if not (r.get("concise_knowledge") or "").strip():
            problems.append(f"[{split_name}/{rid}] concise_knowledge为空")
        target = r.get("target_text", "")
        if target != compose_target(r.get("answer_text", ""), r.get("concise_knowledge", "")):
            problems.append(f"[{split_name}/{rid}] target_text格式不符合模板")
        if has_judgment_marker(target) or TARGET_FORBIDDEN_RE.search(target):
            problems.append(f"[{split_name}/{rid}] target_text包含判题标记: {target[:60]}")
        if re.search(r"[（(]\s*[A-E]\s*[）)]", target):
            problems.append(f"[{split_name}/{rid}] target_text包含括号字母标记")
        if not r.get("question") or not r.get("explanation"):
            problems.append(f"[{split_name}/{rid}] question或explanation为空")
    return problems


def validate_output_dir(
    out_dir: Path,
    raw_train_path: Path,
    external_by_source: dict[str, set],
    external_index: NearDuplicateIndex,
    review_target: int = REVIEW_TARGET,
) -> list[str]:
    """对已写出的 train/dev/human_review_100 做全量一致性校验，返回问题列表。"""
    problems: list[str] = []

    def read(name):
        with open(out_dir / name, encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]

    train = read("train.jsonl")
    dev = read("dev.jsonl")
    review = read("human_review_100.jsonl")

    with open(raw_train_path, encoding="utf-8", errors="replace") as f:
        raw_rows = list(csv.DictReader(f))

    problems.extend(validate_records(train, "train"))
    problems.extend(validate_records(dev, "dev"))

    external_all: set = set()
    for hashes in external_by_source.values():
        external_all |= hashes

    for split_name, records in (("train", train), ("dev", dev)):
        for r in records:
            if r["norm_hash"] in external_all:
                problems.append(f"[{split_name}/{r['id']}] 与评测/历史数据精确重复（泄漏）")
            elif external_index.query(r["question"]):
                problems.append(f"[{split_name}/{r['id']}] 与评测/历史数据近重复（泄漏）")
            src = raw_rows[r["source_row"] - 1]
            if (src.get("Question") or "").strip() != r["question"]:
                problems.append(f"[{split_name}/{r['id']}] question与原始CSV不一致（疑似截断/改写）")
            if parse_options_strict(src.get("Options") or "") != r["options"]:
                problems.append(f"[{split_name}/{r['id']}] options与原始CSV不一致（疑似截断/改写）")
            if (src.get("Explanation") or "").strip() != r["explanation"]:
                problems.append(f"[{split_name}/{r['id']}] explanation与原始CSV不一致")

    train_hashes = {r["norm_hash"] for r in train}
    dev_hashes = {r["norm_hash"] for r in dev}
    overlap = train_hashes & dev_hashes
    if overlap:
        problems.append(f"train与dev存在重复题 {len(overlap)} 条")
    train_index = NearDuplicateIndex()
    for r in train:
        train_index.add(r["question"])
    for r in dev:
        if train_index.query(r["question"]):
            problems.append(f"[dev/{r['id']}] 与train近重复")
            break

    if len(review) != review_target:
        problems.append(f"human_review_100 数量为 {len(review)}，期望 {review_target}")
    review_ids = [r["id"] for r in review]
    if len(set(review_ids)) != len(review_ids):
        problems.append("human_review_100 存在重复记录")
    source_map = {r["id"]: r for r in train + dev}
    for r in review:
        if r["id"] not in source_map:
            problems.append(f"[review/{r['id']}] 不属于train/dev")
        elif source_map[r["id"]]["target_text"] != r["target_text"]:
            problems.append(f"[review/{r['id']}] 与来源记录不一致")
    return problems


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------


def build(
    out_dir: Path = OUTPUT_DIR,
    raw_train_path: Path | None = None,
    external_sources: list[tuple[str, Path, str]] | None = None,
    seed: int = SEED,
    train_target: int = TRAIN_TARGET,
    dev_target: int = DEV_TARGET,
    review_target: int = REVIEW_TARGET,
    do_validate: bool = True,
) -> dict:
    raw_train_path = raw_train_path or HERE / "data" / "raw" / "train.csv"
    sources = external_sources if external_sources is not None else default_external_sources()
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = load_raw_train(raw_train_path)

    # 加载外部排除数据（评测题、历史训练集、历史QA、raw val/test）
    by_source_texts, missing_sources, external_input_md5 = load_external_question_texts(sources)
    by_source_hashes = {
        label: {md5_text(norm_q(q)) for q in questions}
        for label, questions in by_source_texts.items()
    }
    external_all: set = set()
    for hashes in by_source_hashes.values():
        external_all |= hashes
    external_index = NearDuplicateIndex()
    for questions in by_source_texts.values():
        for q in questions:
            external_index.add(q)

    # 逐行筛选
    reasons: Counter = Counter()
    external_exact_by_source: Counter = Counter()
    rejected_examples: dict[str, list] = {}
    accepted: list[dict] = []
    internal_hashes: set = set()
    internal_index = NearDuplicateIndex()

    for row_idx, row in enumerate(rows, 1):
        reason, record, knowledge = screen_row(row, row_idx)
        if reason is None:
            question = record["question"]
            h = md5_text(norm_q(question))
            if h in external_all:
                reason = "external_exact_duplicate"
                for label in by_source_hashes:
                    if h in by_source_hashes[label]:
                        external_exact_by_source[label] += 1
                        break
            elif external_index.query(question):
                reason = "external_near_duplicate"
            elif h in internal_hashes:
                reason = "internal_exact_duplicate"
            elif internal_index.query(question):
                reason = "internal_near_duplicate"
        if reason is not None:
            reasons[reason] += 1
            bucket = rejected_examples.setdefault(reason, [])
            if len(bucket) < 10:
                bucket.append({
                    "source_row": row_idx,
                    "question": (row.get("Question") or "").strip()[:80],
                })
            continue
        record["_knowledge"] = knowledge
        internal_hashes.add(md5_text(norm_q(record["question"])))
        internal_index.add(record["question"])
        accepted.append(record)

    # 按答案字母分层切分（先dev后train），并分配确定序号
    train, dev = stratified_split(accepted, seed, train_target, dev_target)
    review = build_review(train, dev, seed, review_target)

    write_jsonl(out_dir / "train.jsonl", train)
    write_jsonl(out_dir / "dev.jsonl", dev)
    write_jsonl(out_dir / "human_review_100.jsonl", review)

    if do_validate:
        problems = validate_output_dir(out_dir, raw_train_path, by_source_hashes, external_index, review_target)
        if problems:
            for p in problems[:50]:
                print("验证失败:", p, file=sys.stderr)
            raise SystemExit(f"产物校验未通过，共 {len(problems)} 个问题；已中止（未写 manifest）")

    # rejected 统计
    rejected_stats = {
        "seed": seed,
        "raw_question_count": len(rows),
        "total_rejected": sum(reasons.values()),
        "reasons": {
            reason: {"count": reasons[reason], "examples": rejected_examples.get(reason, [])}
            for reason in sorted(reasons)
        },
    }
    with open(out_dir / "rejected_stats.json", "w", encoding="utf-8") as f:
        json.dump(rejected_stats, f, ensure_ascii=False, indent=2)

    # manifest（不包含时间戳，保证同种子完全可复现）
    output_names = ["train.jsonl", "dev.jsonl", "human_review_100.jsonl", "rejected_stats.json"]
    input_files = [raw_train_path] + [p for _, p, _ in sources if p.is_file()]
    letter_counts = {
        split: {L: sum(1 for r in recs if r["answer"] == L) for L in ANSWER_LETTERS}
        for split, recs in (("train", train), ("dev", dev), ("human_review_100", review))
    }
    manifest = {
        "experiment": "curated_knowledge_v1",
        "description": "医学选择题精简知识训练集（答案选项文字 + 单句正向知识句）",
        "seed": seed,
        "near_duplicate_jaccard_threshold": NEAR_DUP_JACCARD_THRESHOLD,
        "targets": {
            "train_max": train_target,
            "dev_target": dev_target,
            "human_review": review_target,
            "train_per_letter_cap": train_target // len(ANSWER_LETTERS),
            "dev_per_letter_cap": -(-dev_target // len(ANSWER_LETTERS)),
        },
        "raw_question_count": len(rows),
        "final_counts": {
            "train": len(train),
            "dev": len(dev),
            "human_review_100": len(review),
            "total": len(train) + len(dev),
        },
        "answer_letter_counts": letter_counts,
        "source_counts": {"CMExam_train": len(train) + len(dev)},
        "dedup_counts": {
            "external_exact_duplicate": reasons.get("external_exact_duplicate", 0),
            "external_exact_by_source": dict(external_exact_by_source),
            "external_near_duplicate": reasons.get("external_near_duplicate", 0),
            "internal_exact_duplicate": reasons.get("internal_exact_duplicate", 0),
            "internal_near_duplicate": reasons.get("internal_near_duplicate", 0),
        },
        "exclusion_reasons": {reason: reasons[reason] for reason in sorted(reasons)},
        "target_text_length_stats": {
            "train": length_stats(train, "target_text"),
            "dev": length_stats(dev, "target_text"),
        },
        "answer_text_length_stats": {
            "train": length_stats(train, "answer_text"),
            "dev": length_stats(dev, "answer_text"),
        },
        "knowledge_length_stats": {
            "train": length_stats(train, "concise_knowledge"),
            "dev": length_stats(dev, "concise_knowledge"),
        },
        "input_md5": {
            str(p): md5_file(p)
            for p in input_files
        }
        | external_input_md5,
        "output_md5": {name: md5_file(out_dir / name) for name in output_names},
        "missing_external_sources": missing_sources,
        "notes": {
            "source_row": "train.csv 中除去表头的数据行序号（1-based），可回溯原始题目",
            "subject_balance": "raw train.csv 无科目元数据列，无法按医学科目分层；已按答案字母分层均衡",
            "knowledge_rule": "知识句仅做标记清理（A对/B错/为本题正确答案等），不编造医学内容；无法可靠提取则整题排除",
            "split_rule": "先dev后train，按答案字母分层等额抽取；train/dev共享同一去重池，天然无重复",
        },
    }
    with open(out_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    return {
        "train": len(train),
        "dev": len(dev),
        "review": len(review),
        "raw": len(rows),
        "letter_counts": letter_counts,
        "reasons": dict(reasons),
        "manifest": manifest,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="构建医学选择题精简知识训练集 curated_knowledge_v1")
    parser.add_argument("--out_dir", default=str(OUTPUT_DIR))
    parser.add_argument("--raw_train", default=str(HERE / "data" / "raw" / "train.csv"))
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--train_target", type=int, default=TRAIN_TARGET)
    parser.add_argument("--dev_target", type=int, default=DEV_TARGET)
    args = parser.parse_args()

    summary = build(
        out_dir=Path(args.out_dir),
        raw_train_path=Path(args.raw_train),
        seed=args.seed,
        train_target=args.train_target,
        dev_target=args.dev_target,
    )
    print(json.dumps({
        "raw": summary["raw"],
        "train": summary["train"],
        "dev": summary["dev"],
        "human_review_100": summary["review"],
        "answer_letter_counts": summary["letter_counts"],
        "top_exclusion_reasons": dict(
            sorted(summary["reasons"].items(), key=lambda kv: -kv[1])[:10]
        ),
        "out_dir": str(args.out_dir),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
