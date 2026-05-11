KB_PATH = "/root/dataset/ACG_kb.db"
FTS_TABLE = "wiki_entries_fts"
DEFAULT_FALLBACK_STRATEGY = "fts_then_web"

MAX_CONTEXT_ENTRIES_CAP = 6
MIN_CONTEXT_CHARS = 120
MAX_CONTEXT_CHARS_CAP = 680
MAX_LOCAL_TOP_K = 12
MAX_FALLBACK_TOP_K = 6

PROMPT = (
    "你是一个精通 ACG 文化的影评分析专家。你的任务是分析用户的 Bangumi 评论及其情感倾向，并将其转化为符合以下定义的 JSON 格式。\n\n"
    "输出要求:\n"
    "1. 必须严格遵守 JSON 语法，输出原始、紧凑的 JSON，不要输出 markdown 代码块（```json）。\n"
    "2. JSON 必须包含以下字段:\n"
    " - \"glossary\": {\"术语名\": \"解释\"}。如果评论中包含 ACG 行话或术语（如 \"神回\"、\"暴死\"），请在此解释；否则为空对象。\n"
    " - \"step_by_step_analysis\": 字符串。请先分析评论中提到的黑话，再结合 \"虽然... 但是...\" 等逻辑，分析评论中表达出对于剧情、作画、音乐、角色等维度的优缺点评价。\n"
    " - \"aspects\": 对象。包含 \"story\", \"visual\", \"music\", \"character\" 四个键，值为 \"Positive\", \"Negative\", \"Neutral\" 或 \"Not Mentioned\"。\n"
    " - \"final_sentiment\": 字符串。整体评价，仅限 \"Positive\", \"Neutral\", \"Negative\"。\n"
    " - \"confidence\": 整数。1-10，表示你对该判断的置信度。\n\n"
    "注意：先进行分析（step_by_step_analysis），再给出结论。不要编造用户未提到的观点。\n"
)

PROMPT_COT = (
    "你是一个精通 ACG 文化的影评分析专家。你的任务是分析用户的 Bangumi 评论及其情感倾向，并将其转化为符合以下定义的 JSON 格式。\n\n"
    "输出要求:\n"
    "1. 必须严格遵守 JSON 语法，输出原始、紧凑的 JSON，不要包含任何空格或缩进，不要输出 markdown 代码块（```json），输出以{开头，字符串以英文双引号开头，且务必输出对应的双引号结尾。\n"
    "2. JSON 必须包含以下字段:\n"
    " - \"glossary\": {\"术语名\": \"解释\"}。如果评论中包含 ACG 行话或术语（如 \"神回\"、\"暴死\"），请在此解释；否则为空对象。\n"
    " - \"step_by_step_analysis\": 字符串。请先分析评论中提到的黑话，再结合 \"虽然... 但是...\" 等逻辑，分析评论中表达出对于剧情、作画、音乐、角色等维度的优缺点评价。\n"
    " - \"aspects\": 对象。包含 \"story\", \"visual\", \"music\", \"character\" 四个键，值为 \"Positive\", \"Negative\", \"Neutral\" 或 \"Not Mentioned\"。\n"
    " - \"final_sentiment\": 字符串。整体评价，仅限 \"Positive\", \"Neutral\", \"Negative\"。\n"
    " - \"confidence\": 整数。1-10，表示你对该判断的置信度。\n\n"
    "注意：先进行分析（step_by_step_analysis），再给出结论。不要编造用户未提到的观点。\n"
)

RAG_FINAL_PROMPT = (
    PROMPT_COT
    + "\n你还会收到一组来自本地知识库、FTS 补召回或网页搜索的候选知识。"
    "请优先利用这些知识解释评论中真正出现或明显指向的术语、别名、黑话和专有名词。"
    "不要复述作品剧情、人物生平或大段背景设定。"
    "glossary 只保留评论中真正需要解释的术语；如果没有就输出空对象。"
    "如果候选知识不足，也要基于评论本身完成分析。"
)

EXTRACT_TERMS_PROMPT = """你是一个资深的 ACG（二次元）实体提取专家。请从评论中严格提取与 ACG 文化紧密相关的专有名词，禁止提取日常词汇。只输出 JSON 格式。

【提取范围】必须且只能是以下四类：
1. 作品名或简称（如：进击的巨人、MyGO、轻音）
2. 角色名或其外号（如：艾伦、睦子米、黄毛）
3. 现实人物与机构（声优、监督、画师、动画公司等，如：羊宫妃娜、大河内、京阿尼、MAPPA）
4. ACG 专属黑话/梗/术语（如：神回、暴死、作画崩坏、NTR、胃疼、三集定律、cjb）

【严禁提取（负面清单）】
1. 严禁提取日常普通名词（如：剧情、故事、画面、时间、学校、朋友）
2. 严禁提取普通的情绪词、形容词、副词（如：爱、温柔、感动、搞笑、恶心）
3. 严禁提取通用指代词（如：男主、女主、别人、角色）

【输出格式】
{"acg_terms": ["词1", "词2", ...]}
如果没找到任何符合要求的 ACG 相关词汇，输出 {"acg_terms": []}。"""

SELECT_CANDIDATES_PROMPT = """你是 ACG 知识检索调度器。你会在候选词条中挑选最相关标题，并决定是否需要后续补召回。
候选可能来自本地知识库，也可能来自网页搜索摘要。
只输出 JSON，格式为:
{"selected_titles": ["title1"], "need_more_search": true/false, "next_terms": ["词"], "reason": "简短理由"}
规则:
1. selected_titles 必须来自候选列表，并优先选择能直接解释评论中专有名词的词条。
2. need_more_search=true 只在当前候选仍不足以解释评论中的关键专名时使用。
3. next_terms 只能填写更具体的作品名、角色名、机构名、别名或黑话，不要填写泛化评论词。
4. 尽量不要选择标题像“/剧情”“/设定”“/列表”这类子页面，除非没有更直接的词条。
【严禁提取（负面清单）】
- 严禁把“剧情、故事、设定、画面、画风、演出、分镜、音乐、台词、节奏、角色、人物、男主、女主、主角、反派、结局、世界观”放进 next_terms。
- 严禁把情绪词、评价词、普通名词放进 next_terms。
- 如果不需要补召回，next_terms 必须输出空列表。"""

GENERIC_NEXT_TERM_BLOCKLIST = {
    "剧情",
    "故事",
    "设定",
    "画面",
    "画风",
    "演出",
    "分镜",
    "音乐",
    "台词",
    "节奏",
    "角色",
    "人物",
    "男主",
    "女主",
    "主角",
    "反派",
    "结局",
    "世界观",
}

NOISY_RECALL_TERM_BLOCKLIST = {
    "剧情",
    "故事",
    "设定",
    "角色",
    "人物",
    "男主",
    "女主",
    "主角",
    "反派",
    "结局",
    "世界观",
}

LOW_VALUE_TITLE_PATTERNS = (
    "／剧情",
    "/剧情",
    "／设定",
    "/设定",
    "／台词",
    "/台词",
    "／音乐",
    "/音乐",
    "／列表",
    "/列表",
    "模板:",
    "分类:",
)

ASPECTS_CH = {
    "story": "剧情",
    "visual": "画面",
    "music": "音乐",
    "character": "角色",
    "final_sentiment": "总评价",
}
