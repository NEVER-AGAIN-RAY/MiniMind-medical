#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
prepare_curated_knowledge.py 的自动化测试。

覆盖：
1. 纯函数单元测试（否定题/组合选项/判题标记/知识句抽取/近重复索引/目标格式）；
2. 合成数据端到端构建测试：产物完整性、答案字母在options、answer_text一致、
   target_text无判题标记、原始字段未被截断改写、train/dev无重复、评测题不泄漏、
   相同随机种子结果完全一致（文件MD5相同）；
3. 真实产物回归校验（若 curated_knowledge_v1 已构建）。

运行: python -m unittest experiments.lora_medical_mcq_pilot.test_prepare_curated_knowledge -v
本地执行边界：合成数据全部写入临时目录，绝不触碰真实数据目录（只读校验除外）。
"""

import csv
import hashlib
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.lora_medical_mcq_pilot import prepare_curated_knowledge as prep


def md5_file(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


class TestNormalizationAndParsing(unittest.TestCase):
    def test_norm_q_matches_prepare_data_semantics(self):
        self.assertEqual(prep.norm_q("主诉：发热，咳嗽 3 天。"), "主诉发热咳嗽3天")

    def test_parse_options_strict(self):
        raw = "A 阿胶\nB 砂仁\nC 西红花\nD 雷丸\nE 车前子"
        opts = prep.parse_options_strict(raw)
        self.assertEqual(set(opts), {"A", "B", "C", "D", "E"})
        self.assertEqual(opts["B"], "砂仁")

    def test_parse_options_strict_rejects_missing_or_extra(self):
        self.assertIsNone(prep.parse_options_strict("A 甲\nB 乙\nC 丙\nD 丁"))
        self.assertIsNone(prep.parse_options_strict("A 甲\n乙\nC 丙\nD 丁\nE 戊"))
        self.assertIsNone(prep.parse_options_strict("A \nB 乙\nC 丙\nD 丁\nE 戊"))

    def test_parse_options_strict_accepts_dot_style(self):
        opts = prep.parse_options_strict("A. 阿胶\nB. 砂仁\nC. 西红花\nD. 雷丸\nE. 车前子")
        self.assertEqual(opts["A"], "阿胶")


class TestRuleDetection(unittest.TestCase):
    def test_negative_question_by_stem(self):
        for stem in ("下列哪项是错误的", "不属于典型表现的是", "下列除哪项外均符合", "哪种情况不能出现"):
            self.assertTrue(prep.is_negative_question(stem, "正常解析内容。"), stem)
        self.assertFalse(prep.is_negative_question("最常见的死亡原因是什么", "解析内容充分。"))

    def test_negative_question_by_explanation_pattern(self):
        self.assertTrue(prep.is_negative_question("以下哪项正确", "瞳孔开大肌（D错，为本题正确答案）参与。"))
        self.assertFalse(prep.is_negative_question("以下哪项正确", "瞳孔括约肌（D对）参与。"))

    def test_combination_options(self):
        self.assertTrue(prep.is_combination_option("以上都对"))
        self.assertTrue(prep.is_combination_option("以上都不是"))
        self.assertTrue(prep.is_combination_option("A和B"))
        self.assertFalse(prep.is_combination_option("对乙酰氨基酚"))
        self.assertFalse(prep.is_combination_option("急性脑血管病"))

    def test_judgment_marker_boundaries(self):
        self.assertTrue(prep.has_judgment_marker("（C对）"))
        self.assertTrue(prep.has_judgment_marker("，B错，其余选项均不符合"))
        self.assertFalse(prep.has_judgment_marker("维生素A对视觉发育有益"))
        self.assertFalse(prep.has_judgment_marker("甲状腺C细胞分泌降钙素"))

    def test_extract_judgment_letters(self):
        dui, cuo = prep.extract_judgment_letters("（B对），高血压危象（A错）、尿毒症（C错）。")
        self.assertEqual(dui, {"B"})
        self.assertEqual(cuo, {"A", "C"})

    def test_conflicted_explanation(self):
        self.assertFalse(prep.is_conflicted_explanation("砂仁需后下（B对）。", "B"))
        self.assertTrue(prep.is_conflicted_explanation("砂仁需后下（A对）。", "B"))
        self.assertTrue(prep.is_conflicted_explanation("答案为C。", "B"))

    def test_visual_and_garbage(self):
        self.assertTrue(prep.VISUAL_RE.search("如图所示，箭头所指结构是"))
        self.assertTrue(prep.VISUAL_RE.search("见下表"))
        self.assertFalse(prep.VISUAL_RE.search("胃下表面黏膜"))  # “下表面”不应误伤
        self.assertTrue(prep.GARBAGE_RE.search("题干<br>残留"))
        self.assertTrue(prep.GARBAGE_RE.search("乱码\ufffd出现"))


class TestKnowledgeExtraction(unittest.TestCase):
    OPTIONS = {
        "A": "高血压危象", "B": "急性脑血管病", "C": "尿毒症", "D": "心力衰竭", "E": "缺血性心脏病",
    }

    def test_clean_fragment_removes_markers(self):
        self.assertEqual(
            prep.clean_fragment("降钙素主要由甲状腺C细胞（C对）分泌。"),
            "降钙素主要由甲状腺C细胞分泌",
        )
        self.assertEqual(
            prep.clean_fragment("医生应避免诱导性提问（D错，为本题正确答案）。"),
            "医生应避免诱导性提问",
        )
        self.assertEqual(prep.clean_fragment("A对，B错，需后下的药是砂仁（B对）。"), "需后下的药是砂仁")

    def test_extract_knowledge_positive(self):
        exp = "在我国高血压病最常见的死亡原因为急性脑血管病（B对），高血压危象（A错）、尿毒症（C错）等。"
        knowledge, fail = prep.extract_knowledge(exp, "B", self.OPTIONS)
        self.assertEqual(fail, "")
        self.assertEqual(knowledge, "在我国高血压病最常见的死亡原因为急性脑血管病")
        target = prep.compose_target(self.OPTIONS["B"], knowledge)
        self.assertEqual(target, "急性脑血管病。知识点：在我国高血压病最常见的死亡原因为急性脑血管病。")

    def test_extract_knowledge_sentence_level(self):
        exp = "本题考查的是内分泌。降钙素主要由甲状腺C细胞（C对）分泌，作用是降低血钙。"
        knowledge, fail = prep.extract_knowledge(exp, "C", {"A": "甲状旁腺", "B": "胰岛A细胞", "C": "甲状腺C细胞", "D": "肾上腺", "E": "垂体"})
        self.assertEqual(fail, "")
        self.assertEqual(knowledge, "降钙素主要由甲状腺C细胞分泌，作用是降低血钙")

    def test_extract_knowledge_rejects_negative_statement(self):
        exp = "维生素K不是该病出血的病因（B对），临床需注意鉴别。"
        knowledge, fail = prep.extract_knowledge(exp, "B", {"A": "凝血因子", "B": "维生素K", "C": "肝素", "D": "华法林", "E": "阿司匹林"})
        self.assertIsNone(knowledge)
        self.assertEqual(fail, prep.FAIL_NEGATIVE)

    def test_extract_knowledge_cleans_combo_marker(self):
        # “（A对E错）”组合判题标记应被清理，保留正向知识句
        exp = "可以全面描述正态分布资料特征的两个指标是均数和标准差（A对E错）。"
        options = {"A": "均数和标准差", "B": "均数", "C": "标准差", "D": "变异系数", "E": "中位数"}
        knowledge, fail = prep.extract_knowledge(exp, "A", options)
        self.assertEqual(fail, "")
        self.assertEqual(knowledge, "可以全面描述正态分布资料特征的两个指标是均数和标准差")

    def test_extract_knowledge_cleans_mixed_letter_block(self):
        # CMExam 常见“（E对BCD错）/（ACDE错）”标注应整体清理
        self.assertEqual(
            prep.clean_fragment("容易闻及二尖瓣杂音的体位是左侧卧位（E对BCD错），因为二尖瓣杂音常向左腋下传导。"),
            "容易闻及二尖瓣杂音的体位是左侧卧位，因为二尖瓣杂音常向左腋下传导",
        )
        self.assertEqual(
            prep.clean_fragment("小儿心、肝虽同样未曾充盛，功能未健，不及肺、脾、肾突出（ACDE错）。"),
            "小儿心、肝虽同样未曾充盛，功能未健，不及肺、脾、肾突出",
        )
        # 含正文的括号不受影响
        self.assertEqual(
            prep.clean_fragment("可选对乙酰氨基酚（A型）进行治疗（B对）。"),
            "可选对乙酰氨基酚（A型）进行治疗",
        )

    def test_extract_knowledge_rejects_pure_marker_echo(self):
        # 清理后只剩答案回声（无医学陈述）→ 整题排除
        exp = "选择四七汤（A对E错）。"
        options = {"A": "四七汤", "B": "连理汤", "C": "芍药汤", "D": "白头翁汤", "E": "乌梅丸"}
        knowledge, fail = prep.extract_knowledge(exp, "A", options)
        self.assertIsNone(knowledge)
        self.assertEqual(fail, prep.FAIL_NO_CANDIDATE)

    def test_extract_knowledge_rejects_multi_wrong_options(self):
        exp = "与高血压危象（A错）、尿毒症（C错）不同，答案是急性脑血管病（B对）。"
        knowledge, fail = prep.extract_knowledge(exp, "B", self.OPTIONS)
        self.assertIsNone(knowledge)
        self.assertEqual(fail, prep.FAIL_WRONG_OPTIONS)

    def test_extract_knowledge_rejects_when_absent(self):
        exp = "本节内容请参考教材相应章节。"
        knowledge, fail = prep.extract_knowledge(exp, "B", self.OPTIONS)
        self.assertIsNone(knowledge)
        self.assertEqual(fail, prep.FAIL_NO_CANDIDATE)

    def test_compose_target(self):
        self.assertEqual(
            prep.compose_target("甲状腺C细胞", "降钙素主要由甲状腺C细胞分泌。"),
            "甲状腺C细胞。知识点：降钙素主要由甲状腺C细胞分泌。",
        )


class TestNearDuplicateIndex(unittest.TestCase):
    def test_exact_and_distinct(self):
        idx = prep.NearDuplicateIndex()
        idx.add("空腹状态下胰岛素分泌的主要调节方式是下列选项中的哪一种类型")
        self.assertTrue(idx.query("空腹状态下胰岛素分泌的主要调节方式是下列选项中的哪一种类型"))
        self.assertFalse(idx.query("心绞痛发作时首选的舌下含服药物是以下哪一项内容"))

    def test_minor_edit_is_near_duplicate(self):
        idx = prep.NearDuplicateIndex()
        idx.add("空腹状态下机体调节胰岛素分泌的最主要体液因素是下列选项中的哪一种类型")
        self.assertTrue(idx.query("空腹状态下机体调节胰岛素分泌的最主要体液因素是下列选项中的哪二种类型"))

    def test_distinct_medical_questions_not_flagged(self):
        idx = prep.NearDuplicateIndex()
        idx.add("高血压病最常见的死亡原因是哪一项")
        self.assertFalse(idx.query("低钾血症最典型的心电图改变是以下哪种表现"))


class TestScreenRow(unittest.TestCase):
    def test_good_row_passes(self):
        row = {
            "Question": "降钙素主要由下列哪种细胞分泌",
            "Options": "A 甲状旁腺\nB 胰岛A细胞\nC 甲状腺C细胞\nD 肾上腺\nE 垂体",
            "Answer": "C",
            "Explanation": "降钙素主要由甲状腺C细胞（C对）分泌，可降低血钙水平。",
        }
        reason, record, knowledge = prep.screen_row(row, 1)
        self.assertIsNone(reason)
        self.assertEqual(record["answer_text"], "甲状腺C细胞")
        self.assertEqual(knowledge, "降钙素主要由甲状腺C细胞分泌，可降低血钙水平")

    def test_negative_and_combo_rows_rejected(self):
        row_neg = {
            "Question": "下列不属于心包填塞表现的是",
            "Options": "A 心率加快\nB 心音遥远\nC 血压下降\nD 奇脉\nE 心影缩小",
            "Answer": "E",
            "Explanation": "心包填塞时心影多增大，心影缩小（E错，为本题正确答案）不符合表现。",
        }
        reason, _, _ = prep.screen_row(row_neg, 1)
        self.assertEqual(reason, "negative_question")

        row_combo = {
            "Question": "主诉书写的要求包括哪些内容",
            "Options": "A 症状\nB 时间\nC 部位\nD 症状+时间\nE 以上都对",
            "Answer": "E",
            "Explanation": "主诉包括主要症状及持续时间（E对），书写应简明扼要。",
        }
        reason, _, _ = prep.screen_row(row_combo, 2)
        self.assertEqual(reason, "combination_options")

    def test_long_answer_text_rejected(self):
        long_text = "弥漫性毒性甲状腺肿合并浸润性突眼胫前黏液性水肿以及甲亢性心脏病的患者"
        self.assertGreater(len(long_text), 30)
        row = {
            "Question": "该患者的最可能诊断是下列哪一项所述的疾病名称",
            "Options": f"A Graves病\nB 结节性甲状腺肿\nC {long_text}\nD 亚急性甲状腺炎\nE 桥本甲状腺炎",
            "Answer": "C",
            "Explanation": f"{long_text}（C对）符合该患者的全部临床特点，其余选项不符合。",
        }
        reason, _, _ = prep.screen_row(row, 1)
        self.assertEqual(reason, "answer_text_too_long_or_multi_sentence")


class TestSyntheticEndToEnd(unittest.TestCase):
    """端到端：合成CSV -> build -> 全量校验 + 确定性（同种子两次构建文件MD5一致）。"""

    # 10 条合格题：答案字母 A-E 各 2 条
    GOOD_ROWS = [
        ("空腹状态下机体调节胰岛素分泌的最主要体液因素是下列选项中的哪一种类型",
         "A 胰岛素\nB 胰高血糖素\nC 生长抑素\nD 皮质醇\nE 肾上腺素", "A",
         "血糖水平是调节胰岛素分泌最重要的因素（A对），血糖升高时胰岛素分泌增加。"),
        ("促进肝糖原分解和糖异生最主要的激素是下列选项中的哪一项",
         "A 胰岛素\nB 胰高血糖素\nC 生长抑素\nD 皮质醇\nE 肾上腺素", "B",
         "胰高血糖素（B对）促进肝糖原分解和糖异生，从而升高血糖水平。"),
        ("降钙素主要由下列哪种细胞分泌",
         "A 甲状旁腺\nB 胰岛A细胞\nC 甲状腺C细胞\nD 肾上腺皮质\nE 垂体前叶", "C",
         "降钙素主要由甲状腺C细胞（C对）分泌，其生理作用是降低血钙。"),
        ("慢性心力衰竭最基本的血流动力学改变是下列哪一项",
         "A 心率增快\nB 心肌肥厚\nC 心室扩大\nD 心排出量下降\nE 静脉压升高", "D",
         "心排出量下降（D对）是慢性心力衰竭最基本的血流动力学改变。"),
        ("需要先煎的药物是下列哪一种",
         "A 阿胶\nB 砂仁\nC 西红花\nD 车前子\nE 龙骨", "E",
         "龙骨（E对）为矿物类药材，质地坚硬，需要先煎以利有效成分煎出。"),
        ("参与瞳孔对光反射的结构是下列哪一项",
         "A 视神经\nB 瞳孔括约肌\nC 动眼神经\nD 瞳孔开大肌\nE 睫状神经节", "E",
         "睫状神经节（E对）参与瞳孔对光反射通路，其节后纤维支配瞳孔括约肌。"),
        ("弥漫性毒性甲状腺肿最常见的眼部体征是下列哪一项",
         "A 眼睑下垂\nB 眼球突出\nC 眼睑水肿\nD 复视\nE 视野缺损", "B",
         "眼球突出（B对）是弥漫性毒性甲状腺肿最常见的眼部体征。"),
        ("慢性肾小球肾炎最基本的临床表现是下列哪一项",
         "A 蛋白尿\nB 血尿\nC 水肿\nD 高血压\nE 肾功能减退", "A",
         "蛋白尿（A对）是慢性肾小球肾炎最基本的临床表现。"),
        ("糖尿病微血管病变的特异性改变是下列哪一项",
         "A 糖尿病肾病\nB 神经病变\nC 糖尿病视网膜病变\nD 糖尿病心肌病\nE 皮肤病变", "C",
         "糖尿病视网膜病变（C对）是糖尿病微血管病变的特异性改变之一。"),
        ("引起社区获得性肺炎最常见的病原体是下列哪一项",
         "A 金黄色葡萄球菌\nB 流感嗜血杆菌\nC 肺炎克雷伯菌\nD 肺炎链球菌\nE 铜绿假单胞菌", "D",
         "肺炎链球菌（D对）是社区获得性肺炎最常见的病原体。"),
    ]

    @staticmethod
    def _write_synthetic_dataset(tmp: Path):
        raw_dir = tmp / "raw"
        raw_dir.mkdir(parents=True)
        rows = []

        # 合格题（A-E 各 2 条）
        for q, opts, ans, exp in TestSyntheticEndToEnd.GOOD_ROWS:
            rows.append([q, opts, ans, exp])

        def add(q, opts, ans, exp):
            rows.append([q, opts, ans, exp])

        # 各类应被排除的题
        add("下列不属于心包填塞表现的是",
            "A 心率加快\nB 心音遥远\nC 血压下降\nD 奇脉\nE 心影缩小", "E",
            "心包填塞时心影多增大，心影缩小（E错，为本题正确答案）不符合典型表现。")
        add("主诉书写的要求包括哪些内容",
            "A 主要症状\nB 持续时间\nC 主要体征\nD 简明扼要\nE 以上都对", "E",
            "主诉一般包括主要症状（或体征）及持续时间（E对），书写要简明。")
        add("该患者的最可能诊断是下列哪一项所述的疾病名称",
            "A Graves病\nB 结节性甲状腺肿\nC 弥漫性毒性甲状腺肿合并浸润性突眼胫前黏液性水肿以及甲亢性心脏病的患者\nD 亚急性甲状腺炎\nE 桥本甲状腺炎",
            "C", "弥漫性毒性甲状腺肿合并浸润性突眼胫前黏液性水肿以及甲亢性心脏病的患者（C对）符合全部临床特点。")
        # 内部精确重复（与合格题第3条完全相同）
        add("降钙素主要由下列哪种细胞分泌",
            "A 甲状旁腺\nB 胰岛A细胞\nC 甲状腺C细胞\nD 肾上腺皮质\nE 垂体前叶", "C",
            "降钙素主要由甲状腺C细胞（C对）分泌，其生理作用是降低血钙。")
        # 与合格题第1条仅差一个字的近重复
        add("空腹状态下机体调节胰岛素分泌的最主要体液因素是下列选项中的哪二种类型",
            "A 胰岛素\nB 胰高血糖素\nC 生长抑素\nD 皮质醇\nE 肾上腺素", "A",
            "血糖水平是调节胰岛素分泌最重要的因素（A对），血糖升高时胰岛素分泌增加。")
        # 与外部评测集完全相同的问题（泄漏样例）
        add("流感病毒最容易发生变异的成分是哪一项",
            "A 包膜脂质\nB 血凝素和神经氨酸酶\nC 核蛋白\nD 基质蛋白\nE 多聚酶蛋白", "B",
            "血凝素和神经氨酸酶（B对）是流感病毒最容易发生变异的成分。")
        # 解析与答案冲突
        add("该病活动期最典型的实验室发现是哪一项",
            "A 血沉增快\nB 抗核抗体阳性\nC 补体下降\nD 类风湿因子阳性\nE C反应蛋白升高", "D",
            "类风湿因子（A对）在多数患者中为阳性表现。")
        # 知识句为负向陈述
        add("维生素K的生理作用特点是下列哪一项",
            "A 促进凝血因子合成\nB 抑制凝血\nC 扩张血管\nD 利尿\nE 镇痛",
            "A", "促进凝血因子合成（A对）不是其不良反应，临床用药需注意区分。")
        # 乱码/HTML 残留
        add("该患儿最可能的诊断是哪一项，<br>请结合检查",
            "A 佝偻病\nB 软骨发育不全\nC 黏多糖病\nD 成骨不全\nE 先天性甲减", "A",
            "佝偻病（A对）患儿可见方颅、肋骨串珠等典型骨骼改变。")
        # 题干疑似截断
        add("该患者目前最需要紧急处理的措施是哪一项，",
            "A 补液\nB 利尿\nC 抗感染\nD 输血\nE 手术", "A",
            "补液（A对）是休克患者最需要紧急处理的措施。")
        # 解析中缺少正确选项文字
        add("下列哪种药物需要监测血药浓度",
            "A 地高辛\nB 阿司匹林\nC 青霉素\nD 维生素C\nE 葡萄糖", "A",
            "该药治疗窗窄，用药期间应监测其血药浓度变化。")

        with open(raw_dir / "train.csv", "w", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            w.writerow(["Question", "Options", "Answer", "Explanation"])
            w.writerows(rows)

        # 外部评测源（合成）：与上面“泄漏样例”同题
        ext = tmp / "eval.jsonl"
        with open(ext, "w", encoding="utf-8") as f:
            f.write(json.dumps({"id": "mcq_001",
                                "question": "流感病毒最容易发生变异的成分是哪一项",
                                "options": {}, "answer": "B"}, ensure_ascii=False) + "\n")
        return raw_dir / "train.csv", [("synthetic_eval", ext, "jsonl_question")]

    def test_build_end_to_end_and_determinism(self):
        tmp = Path(tempfile.mkdtemp(prefix="curated_test_"))
        try:
            raw_train, ext_sources = self._write_synthetic_dataset(tmp)
            out1, out2 = tmp / "out1", tmp / "out2"
            summary = prep.build(
                out_dir=out1, raw_train_path=raw_train, external_sources=ext_sources,
                seed=42, train_target=5, dev_target=5, review_target=4, do_validate=True,
            )
            # 构建必须内部通过全部校验（do_validate=True 时失败会 SystemExit）
            self.assertEqual(summary["train"] + summary["dev"], 10)

            # 再次构建（同种子）→ 全部产物逐字节一致
            prep.build(out_dir=out2, raw_train_path=raw_train, external_sources=ext_sources,
                       seed=42, train_target=5, dev_target=5, review_target=4, do_validate=True)
            for name in ["train.jsonl", "dev.jsonl", "human_review_100.jsonl",
                         "manifest.json", "rejected_stats.json"]:
                self.assertEqual(md5_file(out1 / name), md5_file(out2 / name), name)

            # 逐条不变量
            train = [json.loads(l) for l in (out1 / "train.jsonl").read_text(encoding="utf-8").splitlines()]
            dev = [json.loads(l) for l in (out1 / "dev.jsonl").read_text(encoding="utf-8").splitlines()]
            train_hashes = {r["norm_hash"] for r in train}
            dev_hashes = {r["norm_hash"] for r in dev}
            self.assertFalse(train_hashes & dev_hashes)
            ext_hash = prep.md5_text(prep.norm_q("流感病毒最容易发生变异的成分是哪一项"))
            self.assertNotIn(ext_hash, train_hashes)
            self.assertNotIn(ext_hash, dev_hashes)
            for r in train + dev:
                self.assertEqual(set(r["options"]), {"A", "B", "C", "D", "E"})
                self.assertIn(r["answer"], r["options"])
                self.assertEqual(r["answer_text"], r["options"][r["answer"]])
                self.assertTrue(r["concise_knowledge"].strip())
                self.assertEqual(r["target_text"],
                                 prep.compose_target(r["answer_text"], r["concise_knowledge"]))
                self.assertFalse(prep.has_judgment_marker(r["target_text"]))
                self.assertFalse(prep.TARGET_FORBIDDEN_RE.search(r["target_text"]))

            # 排除原因齐全
            reasons = summary["reasons"]
            for expected in ["negative_question", "combination_options",
                             "answer_text_too_long_or_multi_sentence",
                             "internal_exact_duplicate", "internal_near_duplicate",
                             "external_exact_duplicate",
                             "explanation_answer_conflict", "knowledge_negative_statement",
                             "garbled_or_html", "question_truncated",
                             "answer_absent_from_explanation"]:
                self.assertIn(expected, reasons, expected)

            # 排除统计与示例
            rej = json.loads((out1 / "rejected_stats.json").read_text(encoding="utf-8"))
            self.assertEqual(rej["seed"], 42)
            self.assertIn("external_exact_duplicate", rej["reasons"])
            self.assertLessEqual(len(rej["reasons"]["external_exact_duplicate"]["examples"]), 10)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_validation_catches_broken_record(self):
        full_options = {"A": "甲", "B": "乙", "C": "丙", "D": "丁", "E": "戊"}
        broken = [{
            "id": "curated_train_00001", "question": "q", "options": full_options,
            "answer": "C", "answer_text": "乙", "concise_knowledge": "k",
            "target_text": "丙。知识点：k。", "source": "x", "source_row": 1,
            "explanation": "e", "norm_hash": "h",
        }]
        problems = prep.validate_records(broken, "train")
        self.assertTrue(any("answer_text与正确选项不一致" in p for p in problems))

        marked = [dict(broken[0], answer_text="丙", target_text="丙。知识点：B错，k。")]
        problems = prep.validate_records(marked, "train")
        self.assertTrue(any("判题标记" in p for p in problems))


REAL_OUT_DIR = prep.OUTPUT_DIR


@unittest.skipUnless(REAL_OUT_DIR.joinpath("train.jsonl").is_file(),
                     "真实产物尚未构建，跳过回归校验")
class TestRealOutputs(unittest.TestCase):
    def test_real_curated_outputs_pass_validation(self):
        sources = prep.default_external_sources()
        by_source_texts, _, _ = prep.load_external_question_texts(sources)
        by_source_hashes = {
            label: {prep.md5_text(prep.norm_q(q)) for q in qs}
            for label, qs in by_source_texts.items()
        }
        ext_index = prep.NearDuplicateIndex()
        for qs in by_source_texts.values():
            for q in qs:
                ext_index.add(q)
        problems = prep.validate_output_dir(
            REAL_OUT_DIR, HERE / "data" / "raw" / "train.csv",
            by_source_hashes, ext_index, prep.REVIEW_TARGET,
        )
        self.assertEqual(problems, [])


if __name__ == "__main__":
    unittest.main(verbosity=2)
