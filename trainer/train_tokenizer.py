# 注：不建议再重复训练tokenizer（“词典”），MiniMind已自带，此脚本仅供学习和参考。基于不同词典训练的模型将导致输出完全不统一，降低社区的模型复用性
# Note: It is not recommended to re-train the tokenizer. MiniMind already includes one. This script is for learning and reference only. Training models with different tokenizers will lead to inconsistent outputs and reduce model reusability in the community.
import json
import os

from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

DATA_PATH = "../dataset/sft_t2t_mini.jsonl"
TOKENIZER_DIR = "../model_learn_tokenizer/"
VOCAB_SIZE = 6400
SPECIAL_TOKENS_NUM = 36


def get_texts(data_path):
    with open(data_path, "r", encoding="utf-8", errors="ignore") as f:
        for i, line in enumerate(f):
            if i >= 10000:
                break  # 选10000行测试
            try:
                data = json.loads(line)
                contents = [
                    item.get("content")
                    for item in data.get("conversations", [])
                    if item.get("content")
                ]
                if contents:
                    yield "\n".join(contents)
            except json.JSONDecodeError:
                continue


def train_tokenizer(
    data_path, tokenizer_dir, vocab_size, special_tokens_num=SPECIAL_TOKENS_NUM
):
    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)

    special_tokens_list = [
        "<|endoftext|>",
        "<|im_start|>",
        "<|im_end|>",
        "<|object_ref_start|>",
        "<|object_ref_end|>",
        "<|box_start|>",
        "<|box_end|>",
        "<|quad_start|>",
        "<|quad_end|>",
        "<|vision_start|>",
        "<|vision_end|>",
        "<|vision_pad|>",
        "<|image_pad|>",
        "<|video_pad|>",
        "<|audio_start|>",
        "<|audio_end|>",
        "<|audio_pad|>",
        "<tts_pad>",
        "<tts_text_bos>",
        "<tts_text_eod>",
        "<tts_text_bos_single>",
    ]

    additional_tokens_list = [
        "<tool_call>",
        "</tool_call>",
        "<tool_response>",
        "</tool_response>",
        "<think>",
        "</think>",
    ]
    num_buffer = special_tokens_num - len(special_tokens_list + additional_tokens_list)
    buffer_tokens = [
        f"<|buffer{i}|>" for i in range(1, num_buffer + 1)
    ]  # 预留一定数量的token位置
    all_special_tokens = special_tokens_list + additional_tokens_list + buffer_tokens
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        show_progress=True,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        special_tokens=all_special_tokens,
    )
    texts = get_texts(data_path)
    tokenizer.train_from_iterator(texts, trainer=trainer)
    tokenizer.decoder = decoders.ByteLevel()
    tokenizer.add_special_tokens(special_tokens_list)

    os.makedirs(tokenizer_dir, exist_ok=True)  # 创建保存目录（已存在也不报错）
    tokenizer.save(os.path.join(tokenizer_dir, "tokenizer.json"))  # 保存完整分词器（词表+规则+配置）为单个json
    tokenizer.model.save(tokenizer_dir)  # 再存一份传统格式的词表（vocab.json + merges.txt），供老式工具使用
    tokenizer_json_path = os.path.join(tokenizer_dir, "tokenizer.json")  # 记下文件路径，下面要打开它修改
    # ===== 修正special标记：训练器把所有added token都标成special=True，需要手动区分 =====
    with open(tokenizer_json_path, "r", encoding="utf-8") as f:
        tokenizer_data = json.load(f)  # 把刚保存的json读成Python字典
    for token_info in tokenizer_data.get("added_tokens", []):  # 遍历每个额外添加的token
        if token_info["content"] not in special_tokens_list:  # 不在"真·控制符"名单里的（如<tool_call>、<think>）
            token_info["special"] = False  # 改成非special，这样decode时不会被隐藏，下游能解析到这些标签
    with open(tokenizer_json_path, "w", encoding="utf-8") as f:
        json.dump(tokenizer_data, f, ensure_ascii=False, indent=2)  # 改完写回文件（ensure_ascii=False让中文正常显示）

    # ===== 构建"特殊token档案"字典：id(字符串) -> token描述信息，稍后写进tokenizer_config.json =====
    added_tokens_decoder = {}
    for i, token in enumerate(all_special_tokens):  # 遍历全部特殊token（i是序号，实际没用到）
        idx = tokenizer.token_to_id(token)  # 用文本查出该token的数字id（档案要求以id为key）
        added_tokens_decoder[str(idx)] = {
            "content": token,  # token的文本内容
            "lstrip": False,  # 匹配时不吞掉左边的空格
            "normalized": False,  # 不做文本规范化，永远按原样精确匹配
            "rstrip": False,  # 匹配时不吞掉右边的空格
            "single_word": False,  # 不要求独立成词，出现在任何位置都匹配
            "special": True if token in special_tokens_list else False,  # 同上面的修正逻辑：真·控制符才是special
        }

    # ===== 汇总所有配置，准备写进 tokenizer_config.json（分词器的"使用说明书"） =====
    config = {
        # --- 编码行为开关：全设False，表示分词器不自动增删任何内容，输入什么就编码什么 ---
        "add_bos_token": False,  # 编码时不自动在开头加句首符
        "add_eos_token": False,  # 编码时不自动在结尾加句尾符
        "add_prefix_space": False,  # 编码前不自动加空格
        "clean_up_tokenization_spaces": False,  # 解码时不自动清理空格
        "spaces_between_special_tokens": False,  # 特殊token之间不自动插空格
        # --- 特殊token档案与名单 ---
        "added_tokens_decoder": added_tokens_decoder,  # 上面构建的档案：每个特殊token的行为设置
        "additional_special_tokens": [  # 特殊token名单
            t for t in special_tokens_list if t not in ["<|endoftext|>"]  # 排除<|endoftext|>，它已单独声明为pad/unk
        ],
        # --- 角色分配：告诉框架哪个token干什么 ---
        "bos_token": "<|im_start|>",  # 句首符 = 每轮对话的开始标记
        "eos_token": "<|im_end|>",  # 句尾符 = 话说完了，模型生成遇到它就停止
        "pad_token": "<|endoftext|>",  # 填充符：批量训练时把短序列补齐用
        "unk_token": "<|endoftext|>",  # 未知字符的兜底token
        # --- 多模态占位符声明（预留给图像/音频/视频能力） ---
        "image_token": "<|image_pad|>",  # 文本中图像的占位符
        "audio_token": "<|audio_pad|>",  # 文本中音频的占位符
        "video_token": "<|video_pad|>",  # 文本中视频的占位符
        "vision_bos_token": "<|vision_start|>",  # 视觉内容开始标记
        "vision_eos_token": "<|vision_end|>",  # 视觉内容结束标记
        "audio_bos_token": "<|audio_start|>",  # 音频内容开始标记
        "audio_eos_token": "<|audio_end|>",  # 音频内容结束标记
        # --- 其他元信息 ---
        "legacy": True,  # 兼容旧版行为
        "model_max_length": 131072,  # 最大上下文长度：128K
        "sp_model_kwargs": {},  # sentencepiece参数（本项目用不到，留空占位）
        # --- 对话模板（Jinja2）：把 messages 列表拼成训练格式的完整文本 ---
        # 处理规则：system/user/assistant包成 <|im_start|>角色\n内容<|im_end|>；
        # 有tools时插入工具说明书；assistant自动套<think>格式和<tool_call>格式；
        # add_generation_prompt=True时结尾补 <|im_start|>assistant\n 表示"该模型说了"
        "chat_template": "{%- if tools %}\n    {{- '<|im_start|>system\\n' }}\n    {%- if messages[0].role == 'system' %}\n        {{- messages[0].content + '\\n\\n' }}\n    {%- endif %}\n    {{- \"# Tools\\n\\nYou may call one or more functions to assist with the user query.\\n\\nYou are provided with function signatures within <tools></tools> XML tags:\\n<tools>\" }}\n    {%- for tool in tools %}\n        {{- \"\\n\" }}\n        {{- tool | tojson }}\n    {%- endfor %}\n    {{- \"\\n</tools>\\n\\nFor each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\\n<tool_call>\\n{\\\"name\\\": <function-name>, \\\"arguments\\\": <args-json-object>}\\n</tool_call><|im_end|>\\n\" }}\n{%- else %}\n    {%- if messages[0].role == 'system' %}\n        {{- '<|im_start|>system\\n' + messages[0].content + '<|im_end|>\\n' }}\n    {%- endif %}\n{%- endif %}\n{%- set ns = namespace(multi_step_tool=true, last_query_index=messages|length - 1) %}\n{%- for message in messages[::-1] %}\n    {%- set index = (messages|length - 1) - loop.index0 %}\n    {%- if ns.multi_step_tool and message.role == \"user\" and message.content is string and not(message.content.startswith('<tool_response>') and message.content.endswith('</tool_response>')) %}\n        {%- set ns.multi_step_tool = false %}\n        {%- set ns.last_query_index = index %}\n    {%- endif %}\n{%- endfor %}\n{%- for message in messages %}\n    {%- if message.content is string %}\n        {%- set content = message.content %}\n    {%- else %}\n        {%- set content = '' %}\n    {%- endif %}\n    {%- if (message.role == \"user\") or (message.role == \"system\" and not loop.first) %}\n        {{- '<|im_start|>' + message.role + '\\n' + content + '<|im_end|>' + '\\n' }}\n    {%- elif message.role == \"assistant\" %}\n        {%- set reasoning_content = '' %}\n        {%- if message.reasoning_content is string %}\n            {%- set reasoning_content = message.reasoning_content %}\n        {%- else %}\n            {%- if '</think>' in content %}\n                {%- set reasoning_content = content.split('</think>')[0].rstrip('\\n').split('<think>')[-1].lstrip('\\n') %}\n                {%- set content = content.split('</think>')[-1].lstrip('\\n') %}\n            {%- endif %}\n        {%- endif %}\n        {%- if true %}\n            {{- '<|im_start|>' + message.role + '\\n<think>\\n' + reasoning_content.strip('\\n') + '\\n</think>\\n\\n' + content.lstrip('\\n') }}\n        {%- endif %}\n        {%- if message.tool_calls %}\n            {%- for tool_call in message.tool_calls %}\n                {%- if (loop.first and content) or (not loop.first) %}\n                    {{- '\\n' }}\n                {%- endif %}\n                {%- if tool_call.function %}\n                    {%- set tool_call = tool_call.function %}\n                {%- endif %}\n                {{- '<tool_call>\\n{\"name\": \"' }}\n                {{- tool_call.name }}\n                {{- '\", \"arguments\": ' }}\n                {%- if tool_call.arguments is string %}\n                    {{- tool_call.arguments }}\n                {%- else %}\n                    {{- tool_call.arguments | tojson }}\n                {%- endif %}\n                {{- '}\\n</tool_call>' }}\n            {%- endfor %}\n        {%- endif %}\n        {{- '<|im_end|>\\n' }}\n    {%- elif message.role == \"tool\" %}\n        {%- if loop.first or (messages[loop.index0 - 1].role != \"tool\") %}\n            {{- '<|im_start|>user' }}\n        {%- endif %}\n        {{- '\\n<tool_response>\\n' }}\n        {{- content }}\n        {{- '\\n</tool_response>' }}\n        {%- if loop.last or (messages[loop.index0 + 1].role != \"tool\") %}\n            {{- '<|im_end|>\\n' }}\n        {%- endif %}\n    {%- endif %}\n{%- endfor %}\n{%- if add_generation_prompt %}\n    {{- '<|im_start|>assistant\\n' }}\n    {%- if open_thinking is defined and open_thinking is true %}\n        {{- '<think>\\n' }}\n    {%- else %}\n        {{- '<think>\\n\\n</think>\\n\\n' }}\n    {%- endif %}\n{%- endif %}",
        "tokenizer_class": "PreTrainedTokenizerFast",  # 告诉transformers用fast版分词器类来加载
    }

    # ===== 把config字典写进 tokenizer_config.json =====
    with open(
        os.path.join(tokenizer_dir, "tokenizer_config.json"), "w", encoding="utf-8"
    ) as f:
        json.dump(config, f, ensure_ascii=False, indent=4)  # ensure_ascii=False让中文原样保存，indent=4格式化缩进
    print("Tokenizer training completed.")


def eval_tokenizer(tokenizer_dir):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)
    messages = [
        {"role": "system", "content": "你是一个优秀的聊天机器人，总是给我正确的回应！"},
        {"role": "user", "content": "你来自哪里？"},
        {"role": "assistant", "content": "我来自月球"},
        {"role": "user", "content": "你到底来自哪里？"},
        {"role": "assistant", "content": "我来自地球"},
    ]
    new_prompt = tokenizer.apply_chat_template(messages, tokenize=False)
    print("-" * 100)
    print(new_prompt)
    print("-" * 100)
    print("tokenizer词表长度：", len(tokenizer))
    model_inputs = tokenizer(new_prompt)
    print("encoder长度：", len(model_inputs["input_ids"]))
    response = tokenizer.decode(model_inputs["input_ids"], skip_special_tokens=False)
    print("decoder一致性：", response == new_prompt, "\n")
    print("-" * 100)
    print("压缩率测试（Chars/Tokens）：")
    test_texts = [
        # 中文样本 (约200字)
        "人工智能是计算机科学的一个分支，它企图了解智能的实质，并生产出一种新的能以人类智能相似的方式做出反应的智能机器，该领域的研究包括机器人、语言识别、图像识别、自然语言处理和专家系统等。人工智能从诞生以来，理论和技术日益成熟，应用领域也不断扩大，可以设想，未来人工智能带来的科技产品，将会是人类智慧的“容器”。人工智能可以对人的意识、思维的信息过程的模拟。人工智能不是人的智能，但能像人那样思考、也可能超过人的智能。",
        "星际航行是指在星系内甚至星系间的空间中进行的航行。由于宇宙空间极其广阔，传统的化学火箭动力在恒星间航行时显得力不从心。科学家们提出了多种方案，包括离子推进器、核热火箭、甚至是利用反物质作为能源的设想。此外，曲率驱动和虫洞旅行等科幻概念也在理论物理研究中被反复探讨。尽管目前人类的足迹仅限于月球，但随着核聚变技术和材料科学的突破，前往火星乃至更遥远的太阳系边缘将成为可能。",
        # 英文样本 (约200词/字符)
        "Large language models (LLMs) are a type of artificial intelligence (AI) trained on vast amounts of text data to understand and generate human-like language. These models use deep learning techniques, specifically transformers, to process and predict the next word in a sequence. LLMs like GPT-4, Llama, and Claude have demonstrated remarkable capabilities in coding, translation, and creative writing. However, they also face challenges such as hallucinations, where the model generates factually incorrect information, and the need for significant computational resources.",
        "The development of sustainable energy is crucial for the future of our planet. As climate change continues to impact global weather patterns, transitioning from fossil fuels to renewable sources like solar, wind, and hydroelectric power has become an urgent priority. Innovations in battery storage technology and smart grid management are essential to ensure a reliable energy supply. International cooperation and policy frameworks are also necessary to drive the global shift towards a greener economy and reduce carbon emissions.",
        # 混合样本
        "Python 是一种高级编程语言，以其简洁的语法和强大的生态系统而闻名。It is widely used in data science, machine learning, and web development. 开发者可以利用 NumPy, Pandas, and PyTorch 等库快速构建复杂的应用。学习 Python 的过程非常愉快，因为它的代码读起来就像英语一样。Whether you are a beginner or an expert, Python offers something for everyone.",
    ]

    total_compression = 0
    for i, text in enumerate(test_texts):
        encoded = tokenizer.encode(text)
        token_count = len(encoded)
        char_count = len(text)
        compression_ratio = char_count / token_count
        total_compression += compression_ratio
        print(
            f"样本 {i+1} | 字符数: {char_count:4} | Tokens: {token_count:3} | 压缩率: {compression_ratio:.2f}"
        )

    print(f"平均压缩率: {total_compression / len(test_texts):.2f}")
    print("-" * 100)
    print("流式解码（字节缓冲）测试：")
    input_ids = model_inputs["input_ids"]
    token_cache = []
    for tid in input_ids:
        token_cache.append(tid)
        current_decode = tokenizer.decode(token_cache)
        if current_decode and "\ufffd" not in current_decode:
            display_ids = token_cache[0] if len(token_cache) == 1 else token_cache
            raw_tokens = [
                tokenizer.convert_ids_to_tokens(int(t))
                for t in (
                    token_cache if isinstance(token_cache, list) else [token_cache]
                )
            ]
            print(
                f"Token ID: {str(display_ids):15} -> Raw: {str(raw_tokens):20} -> Decode Str: {current_decode}"
            )
            token_cache = []


if __name__ == "__main__":
    train_tokenizer(DATA_PATH, TOKENIZER_DIR, VOCAB_SIZE)
    eval_tokenizer(TOKENIZER_DIR)
