import json
import re
import sqlite3
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Protocol, Tuple

import json_repair

from single_comment_config import (
    DEFAULT_FALLBACK_STRATEGY,
    EXTRACT_TERMS_PROMPT,
    FTS_TABLE,
    GENERIC_NEXT_TERM_BLOCKLIST,
    KB_PATH,
    LOW_VALUE_TITLE_PATTERNS,
    MAX_CONTEXT_CHARS_CAP,
    MAX_CONTEXT_ENTRIES_CAP,
    MAX_FALLBACK_TOP_K,
    MAX_LOCAL_TOP_K,
    MIN_CONTEXT_CHARS,
    NOISY_RECALL_TERM_BLOCKLIST,
    RAG_FINAL_PROMPT,
    SELECT_CANDIDATES_PROMPT,
)

try:
    from ddgs import DDGS
except ImportError:
    DDGS = None

try:
    from tavily import TavilyClient
except ImportError:
    TavilyClient = None


class JSONChatLLM(Protocol):
    def chat_json(
        self,
        system_prompt: str,
        user_prompt: str,
        fallback: dict,
        max_new_tokens: int = 512,
        temperature: float = 0.2,
    ) -> dict: ...


def is_low_value_title(title: str) -> bool:
    normalized = title.strip()
    return any(pattern in normalized for pattern in LOW_VALUE_TITLE_PATTERNS)


def sanitize_next_terms(terms: List[str]) -> List[str]:
    cleaned: List[str] = []
    seen = set()
    for term in terms:
        normalized = str(term).strip()
        if not normalized or len(normalized) <= 1:
            continue
        if normalized in GENERIC_NEXT_TERM_BLOCKLIST:
            continue
        if normalized in seen:
            continue
        seen.add(normalized)
        cleaned.append(normalized)
    return cleaned


def sanitize_recall_terms(terms: List[str]) -> List[str]:
    cleaned: List[str] = []
    seen = set()
    for term in terms:
        normalized = str(term).strip()
        if not normalized:
            continue
        if normalized in NOISY_RECALL_TERM_BLOCKLIST:
            continue
        if normalized in seen:
            continue
        seen.add(normalized)
        cleaned.append(normalized)
    return cleaned


def normalize_source_title(source_title: str) -> str:
    return str(source_title or "").strip()


def build_comment_context(comment: str, source_title: str, source_hint_only: bool = False) -> str:
    normalized_source = normalize_source_title(source_title)
    if not normalized_source:
        return comment
    prefix = "评论来源作品（仅作背景参考，不必强行提取）" if source_hint_only else "评论来源作品"
    return f"{prefix}: {normalized_source}\n用户评论: {comment}"


@dataclass
class RetrievalBudget:
    local_top_k: int
    fallback_top_k: int
    max_context_entries: int
    max_context_chars: int


def compute_retrieval_budget(query: str, recall_terms: List[str]) -> RetrievalBudget:
    query_len = len(query.strip())
    term_count = len(recall_terms)

    if term_count == 0:
        fallback_top_k = 2 if query_len <= 380 else 3
        return RetrievalBudget(
            local_top_k=0,
            fallback_top_k=fallback_top_k,
            max_context_entries=1,
            max_context_chars=MIN_CONTEXT_CHARS,
        )

    if term_count == 1:
        local_top_k = 4
        fallback_top_k = 3
        max_context_entries = 2
    elif term_count <= 3:
        local_top_k = 5
        fallback_top_k = 4
        max_context_entries = 3
    else:
        local_top_k = 10
        fallback_top_k = 5
        max_context_entries = 6

    if query_len > 180:
        local_top_k += 1
        fallback_top_k += 1

    if query_len > 380:
        local_top_k += 1

    local_top_k = min(local_top_k, MAX_LOCAL_TOP_K)
    fallback_top_k = min(fallback_top_k, MAX_FALLBACK_TOP_K)
    max_context_entries = min(max_context_entries, MAX_CONTEXT_ENTRIES_CAP)
    max_context_chars = MAX_CONTEXT_CHARS_CAP

    return RetrievalBudget(
        local_top_k=local_top_k,
        fallback_top_k=fallback_top_k,
        max_context_entries=max_context_entries,
        max_context_chars=max_context_chars,
    )


def shorten_context_text(text: str, max_chars: int) -> str:
    compact = re.sub(r"\s+", " ", text).strip()
    if len(compact) <= max_chars:
        return compact
    sentence_end = compact.rfind("。", 0, max_chars)
    if sentence_end >= max_chars // 2:
        return compact[: sentence_end + 1]
    return compact[:max_chars].rstrip() + "..."


def clean_model_output(text: str) -> str:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S)
    return text.strip()


def parse_json_response(raw_text: str) -> dict:
    cleaned = clean_model_output(raw_text)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        repaired = json_repair.repair_json(cleaned, return_objects=True)
        if isinstance(repaired, dict):
            return repaired
        if isinstance(repaired, str):
            return json.loads(repaired)
        raise ValueError("模型输出不是有效 JSON")


@dataclass
class KBItem:
    title: str
    text: str
    category: str = ""
    aliases: str = ""
    tags: str = ""
    source: str = "local"
    url: str = ""


class SQLiteKB:
    def __init__(self, db_path: str, fts_table: str = FTS_TABLE):
        self.db_path = db_path
        self.fts_table = fts_table
        if not Path(db_path).exists():
            raise FileNotFoundError(f"知识库文件不存在: {db_path}")

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.db_path)

    def has_fts_index(self) -> bool:
        with self._connect() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT 1
                FROM sqlite_master
                WHERE type = 'table' AND name = ? AND sql LIKE '%fts5%'
                LIMIT 1
                """,
                (self.fts_table,),
            )
            return cursor.fetchone() is not None

    def _row_to_item(self, row: tuple, source: str = "local") -> KBItem:
        return KBItem(
            title=row[0],
            text=row[1],
            category=row[2] or "",
            aliases=row[3] or "",
            tags=row[4] or "",
            source=source,
        )

    def fetch_by_titles(self, titles: List[str]) -> Dict[str, KBItem]:
        cleaned_titles = [t.strip() for t in titles if t and t.strip()]
        if not cleaned_titles:
            return {}

        placeholders = ",".join(["?"] * len(cleaned_titles))
        query = f"""
            SELECT title, content, category, aliases, tags
            FROM wiki_entries
            WHERE title IN ({placeholders})
        """
        with self._connect() as conn:
            cursor = conn.cursor()
            cursor.execute(query, cleaned_titles)
            rows = cursor.fetchall()

        return {row[0]: self._row_to_item(row, source="local") for row in rows}

    def retrieve_source_title(self, source_title: str, used_titles: set, top_k: int = 2) -> List[KBItem]:
        normalized_source = normalize_source_title(source_title)
        if not normalized_source:
            return []

        hits: List[KBItem] = []
        seen_titles = set(used_titles)
        exact_alias_pattern = f'%"{normalized_source}"%'

        with self._connect() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT title, content, category, aliases, tags
                FROM wiki_entries
                WHERE title = ? OR aliases LIKE ?
                ORDER BY
                    CASE
                        WHEN title = ? THEN 3
                        WHEN aliases LIKE ? THEN 2
                        ELSE 0
                    END DESC,
                    LENGTH(title) ASC
                LIMIT ?
                """,
                (
                    normalized_source,
                    exact_alias_pattern,
                    normalized_source,
                    exact_alias_pattern,
                    top_k,
                ),
            )
            for row in cursor.fetchall():
                title = row[0]
                if title in seen_titles or is_low_value_title(title):
                    continue
                hits.append(self._row_to_item(row, source="local"))
                seen_titles.add(title)
        return hits

    def retrieve_strict_by_terms(self, terms: List[str], used_titles: set, top_k: int = 8) -> List[KBItem]:
        hits: List[KBItem] = []
        seen_titles = set(used_titles)
        cleaned_terms = [t.strip() for t in terms if t and t.strip()]
        if not cleaned_terms:
            return hits

        with self._connect() as conn:
            cursor = conn.cursor()
            for term in cleaned_terms:
                if len(hits) >= top_k:
                    break
                exact_alias_pattern = f'%"{term}"%'
                exact_tag_pattern = f'%"{term}"%'
                cursor.execute(
                    """
                    SELECT title, content, category, aliases, tags
                    FROM wiki_entries
                    WHERE title = ? OR aliases LIKE ? OR tags LIKE ?
                    ORDER BY
                        CASE
                            WHEN title = ? THEN 4
                            WHEN aliases LIKE ? THEN 3
                            WHEN tags LIKE ? THEN 2
                            ELSE 0
                        END DESC,
                        LENGTH(title) ASC
                    LIMIT ?
                    """,
                    (
                        term,
                        exact_alias_pattern,
                        exact_tag_pattern,
                        term,
                        exact_alias_pattern,
                        exact_tag_pattern,
                        top_k,
                    ),
                )
                for row in cursor.fetchall():
                    title = row[0]
                    if title in seen_titles or is_low_value_title(title):
                        continue
                    hits.append(self._row_to_item(row, source="local"))
                    seen_titles.add(title)
                    if len(hits) >= top_k:
                        break
        return hits

    def retrieve_fts_by_terms(self, terms: List[str], used_titles: set, top_k: int = 6) -> List[KBItem]:
        if not self.has_fts_index():
            return []

        hits: List[KBItem] = []
        seen_titles = set(used_titles)
        cleaned_terms = [t.strip() for t in terms if t and t.strip()]
        if not cleaned_terms:
            return hits

        with self._connect() as conn:
            cursor = conn.cursor()
            for term in cleaned_terms:
                if len(hits) >= top_k:
                    break
                match_query = '"' + term.replace('"', '""') + '"'
                cursor.execute(
                    f"""
                    SELECT w.title, w.content, w.category, w.aliases, w.tags
                    FROM {self.fts_table}
                    JOIN wiki_entries AS w ON {self.fts_table}.rowid = w.id
                    WHERE {self.fts_table} MATCH ?
                    ORDER BY bm25({self.fts_table}, 10.0, 6.0, 4.0, 1.0)
                    LIMIT ?
                    """,
                    (match_query, top_k),
                )
                for row in cursor.fetchall():
                    title = row[0]
                    if title in seen_titles or is_low_value_title(title):
                        continue
                    hits.append(self._row_to_item(row, source="fts"))
                    seen_titles.add(title)
                    if len(hits) >= top_k:
                        break
        return hits


class WebSearcher:
    def __init__(self, tavily_api_key: str):
        self.tavily_api_key = tavily_api_key.strip()
        self.provider = "tavily" if self.tavily_api_key else "ddgs"
        self.cache: Dict[Tuple[str, int], List[KBItem]] = {}
        self.tavily_client = None
        if self.provider == "tavily" and TavilyClient is not None:
            self.tavily_client = TavilyClient(api_key=self.tavily_api_key)

    def _build_query(self, term: str, source_title: str = "") -> str:
        normalized_source = normalize_source_title(source_title)
        if normalized_source and normalized_source not in term:
            return f"{term} {normalized_source} 萌娘百科 site:zh.moegirl.org.cn"
        return f"{term} 萌娘百科 site:zh.moegirl.org.cn"

    def _search_tavily(self, term: str, top_k: int, source_title: str = "") -> List[KBItem]:
        if self.tavily_client is None:
            return []
        response = self.tavily_client.search(query=self._build_query(term, source_title), max_results=top_k)
        results = response.get("results", []) if isinstance(response, dict) else []
        items: List[KBItem] = []
        for result in results:
            title = str(result.get("title", "")).strip()
            text = str(result.get("content", "")).strip()
            url = str(result.get("url", "")).strip()
            if not title:
                title = url or term
            if title and not is_low_value_title(title) and (text or url):
                items.append(
                    KBItem(
                        title=title,
                        text=text or url,
                        category="网页结果",
                        tags=term,
                        source="web",
                        url=url,
                    )
                )
        return items

    def _search_ddgs(self, term: str, top_k: int, source_title: str = "") -> List[KBItem]:
        if DDGS is None:
            return []
        results = list(DDGS().text(self._build_query(term, source_title), max_results=top_k))
        items: List[KBItem] = []
        for result in results:
            title = str(result.get("title", "")).strip()
            text = str(result.get("body", "")).strip()
            url = str(result.get("href", "")).strip()
            if not title:
                title = url or term
            if title and not is_low_value_title(title) and (text or url):
                items.append(
                    KBItem(
                        title=title,
                        text=text or url,
                        category="网页结果",
                        tags=term,
                        source="web",
                        url=url,
                    )
                )
            if len(items) >= top_k:
                break
        return items

    def _search_one_term(self, term: str, top_k: int, source_title: str = "") -> List[KBItem]:
        cache_key = (f"{source_title}::{term}", top_k)
        if cache_key in self.cache:
            return self.cache[cache_key]

        items: List[KBItem] = []
        try:
            if self.provider == "tavily":
                items = self._search_tavily(term, top_k, source_title)
                if not items:
                    items = self._search_ddgs(term, top_k, source_title)
            else:
                items = self._search_ddgs(term, top_k, source_title)
        except Exception:
            items = self._search_ddgs(term, top_k, source_title)

        self.cache[cache_key] = items
        return items

    def search_terms(
        self,
        terms: List[str],
        used_titles: set,
        top_k: int = 6,
        source_title: str = "",
    ) -> List[KBItem]:
        hits: List[KBItem] = []
        seen_titles = set(used_titles)
        seen_urls = set()
        cleaned_terms = [t.strip() for t in terms if t and t.strip()]
        if not cleaned_terms:
            return hits

        for term in cleaned_terms:
            if len(hits) >= top_k:
                break
            for item in self._search_one_term(term, top_k, source_title):
                key = item.url or item.title
                if item.title in seen_titles or key in seen_urls:
                    continue
                hits.append(item)
                seen_titles.add(item.title)
                if item.url:
                    seen_urls.add(item.url)
                if len(hits) >= top_k:
                    break
        return hits


@lru_cache(maxsize=1)
def get_kb() -> SQLiteKB:
    return SQLiteKB(KB_PATH)


def format_candidates(candidates: List[KBItem], max_text_len: int = 160) -> str:
    lines = []
    for i, c in enumerate(candidates, start=1):
        snippet = shorten_context_text(c.text, max_text_len)
        meta_parts = [f"source: {c.source}"]
        if c.category:
            meta_parts.append(f"category: {c.category}")
        if c.tags:
            meta_parts.append(f"tags: {c.tags}")
        if c.url:
            meta_parts.append(f"url: {c.url}")
        meta_line = f"\n   {'; '.join(meta_parts)}" if meta_parts else ""
        lines.append(f"{i}. title: {c.title}{meta_line}\n   text_snippet: {snippet}")
    return "\n".join(lines)


def extract_terms(llm: JSONChatLLM, query: str, source_title: str = "") -> List[str]:
    user_prompt = build_comment_context(query, source_title, source_hint_only=True)
    res = llm.chat_json(
        EXTRACT_TERMS_PROMPT,
        user_prompt,
        fallback={"acg_terms": []},
        max_new_tokens=256,
        temperature=0.1,
    )
    acg_terms = res.get("acg_terms", [])
    if not isinstance(acg_terms, list):
        return []
    seen = set()
    cleaned = []
    for term in acg_terms:
        normalized = str(term).strip()
        if normalized and normalized not in seen:
            seen.add(normalized)
            cleaned.append(normalized)
    return cleaned


def select_candidates(
    llm: JSONChatLLM,
    query: str,
    stage_label: str,
    current_terms: List[str],
    candidates: List[KBItem],
    source_title: str = "",
) -> Tuple[List[str], bool, List[str]]:
    source_block = f"来源作品名: {normalize_source_title(source_title)}\n\n" if normalize_source_title(source_title) else ""
    user_prompt = (
        f"{source_block}用户评论:\n{query}\n\n"
        f"当前阶段: {stage_label}\n"
        f"当前检索词: {current_terms}\n\n"
        f"候选词条:\n{format_candidates(candidates)}\n\n"
        "请给出 JSON。"
    )
    fallback = {"selected_titles": [], "need_more_search": False, "next_terms": [], "reason": ""}
    res = llm.chat_json(SELECT_CANDIDATES_PROMPT, user_prompt, fallback=fallback, max_new_tokens=512, temperature=0.1)

    selected_titles = res.get("selected_titles", [])
    if not isinstance(selected_titles, list):
        selected_titles = []
    raw_next_terms = res.get("next_terms", [])
    if not isinstance(raw_next_terms, list):
        raw_next_terms = []
    next_terms = sanitize_next_terms(raw_next_terms)
    need_more_search = bool(res.get("need_more_search", False))

    return [str(x).strip() for x in selected_titles if str(x).strip()], need_more_search, next_terms


def merge_selected_candidates(
    selected_titles: List[str],
    candidates: List[KBItem],
    used_titles: set,
    selected_titles_all: List[str],
    selected_item_map: Dict[str, KBItem],
) -> None:
    candidate_map = {item.title: item for item in candidates}
    for title in selected_titles:
        item = candidate_map.get(title)
        if item is None or title in used_titles:
            continue
        selected_titles_all.append(title)
        used_titles.add(title)
        selected_item_map[title] = item


def get_fallback_steps(strategy: str) -> List[str]:
    if strategy == "只进行全文搜索":
        return ["fts"]
    if strategy == "只进行网页搜索":
        return ["web"]
    if strategy == "先进行网页搜索再进行全文搜索":
        return ["web", "fts"]
    return ["fts", "web"]


def build_context_text(selected_entries: List[KBItem], budget: RetrievalBudget) -> str:
    if not selected_entries:
        return "无可用词条。"

    blocks = []
    for entry in selected_entries[: budget.max_context_entries]:
        snippet = shorten_context_text(entry.text, budget.max_context_chars)
        if entry.source == "web":
            blocks.append(f"词条: {entry.title}\n来源: 网页搜索\nURL: {entry.url}\n摘要: {snippet}")
        else:
            blocks.append(f"词条: {entry.title}\n解释: {snippet}")
    return "\n\n".join(blocks)


def analyze_with_rag_context(
    llm: JSONChatLLM,
    comment: str,
    source_title: str,
    extracted_terms: List[str],
    selected_entries: List[KBItem],
    budget: RetrievalBudget,
) -> dict:
    context = build_context_text(selected_entries, budget)
    user_prompt = (
        f"{build_comment_context(comment, source_title)}\n\n"
        f"提取术语:\n{extracted_terms}\n\n"
        f"候选知识:\n{context}\n\n"
        "请输出最终 JSON。"
    )
    fallback = {
        "glossary": {},
        "step_by_step_analysis": "模型输出解析失败，未能生成详细分析。",
        "aspects": {
            "story": "Not Mentioned",
            "visual": "Not Mentioned",
            "music": "Not Mentioned",
            "character": "Not Mentioned",
        },
        "final_sentiment": "Neutral",
        "confidence": 5,
    }
    return llm.chat_json(RAG_FINAL_PROMPT, user_prompt, fallback=fallback, max_new_tokens=1200, temperature=0.2)


def run_agentic_rag_analysis(
    llm: JSONChatLLM,
    comment: str,
    source_title: str = "",
    fallback_strategy: str = DEFAULT_FALLBACK_STRATEGY,
    tavily_api_key: str = "",
) -> dict:
    normalized_source_title = normalize_source_title(source_title)
    kb = get_kb()
    web_searcher = WebSearcher(tavily_api_key=tavily_api_key)

    terms = extract_terms(llm, comment, normalized_source_title)
    recall_terms = sanitize_recall_terms(terms)
    budget = compute_retrieval_budget(comment, recall_terms)

    used_titles = set()
    selected_titles_all: List[str] = []
    selected_item_map: Dict[str, KBItem] = {}
    need_more_search = False
    next_terms: List[str] = []

    source_candidates = kb.retrieve_source_title(normalized_source_title, used_titles, top_k=2)
    if recall_terms and budget.local_top_k > 0:
        local_candidates = kb.retrieve_strict_by_terms(recall_terms, used_titles, top_k=budget.local_top_k)
    else:
        local_candidates = []

    if source_candidates:
        merged_candidates: List[KBItem] = []
        seen_candidate_titles = set()
        for item in source_candidates + local_candidates:
            if item.title in seen_candidate_titles:
                continue
            seen_candidate_titles.add(item.title)
            merged_candidates.append(item)
        local_candidates = merged_candidates

    if local_candidates:
        local_selected_titles, need_more_search, next_terms = select_candidates(
            llm=llm,
            query=comment,
            stage_label="本地精确召回",
            current_terms=recall_terms,
            candidates=local_candidates,
            source_title=normalized_source_title,
        )
        merge_selected_candidates(
            local_selected_titles,
            local_candidates,
            used_titles,
            selected_titles_all,
            selected_item_map,
        )
    else:
        need_more_search = True

    fallback_terms = next_terms if next_terms else recall_terms[:]
    if not fallback_terms and normalized_source_title:
        fallback_terms = [normalized_source_title]
    should_run_fallback = need_more_search or not selected_titles_all

    if should_run_fallback and fallback_terms:
        for stage in get_fallback_steps(fallback_strategy):
            if stage == "fts":
                fallback_candidates = kb.retrieve_fts_by_terms(
                    fallback_terms,
                    used_titles,
                    top_k=budget.fallback_top_k,
                )
                stage_label = "FTS补召回"
            else:
                fallback_candidates = web_searcher.search_terms(
                    fallback_terms,
                    used_titles,
                    top_k=budget.fallback_top_k,
                    source_title=normalized_source_title,
                )
                stage_label = f"Web补召回({web_searcher.provider})"

            if not fallback_candidates:
                continue

            selected_fallback_titles, _, _ = select_candidates(
                llm=llm,
                query=comment,
                stage_label=stage_label,
                current_terms=fallback_terms,
                candidates=fallback_candidates,
                source_title=normalized_source_title,
            )
            merge_selected_candidates(
                selected_fallback_titles,
                fallback_candidates,
                used_titles,
                selected_titles_all,
                selected_item_map,
            )
            if selected_fallback_titles:
                break

    title_index = kb.fetch_by_titles(selected_titles_all)
    for title, item in title_index.items():
        selected_item_map[title] = item
    selected_entries = [selected_item_map[t] for t in selected_titles_all if t in selected_item_map]

    return analyze_with_rag_context(
        llm=llm,
        comment=comment,
        source_title=normalized_source_title,
        extracted_terms=terms,
        selected_entries=selected_entries,
        budget=budget,
    )
