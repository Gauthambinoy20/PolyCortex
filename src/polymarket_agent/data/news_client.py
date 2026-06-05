"""Tavily news client with in-memory caching."""

import asyncio
import logging
import os
import re
import time

logger = logging.getLogger(__name__)

_FILLER_WORDS = re.compile(
    r"\b(Will|Does|Is|Are|Has|Have|Can|Could|Would|Should|Do)\b",
    re.IGNORECASE,
)


class NewsClient:
    """Fetches news articles via the Tavily API with time-based caching."""

    def __init__(self, cache_minutes: int = 30, timeout: float = 10.0, baseline_count: int = 3) -> None:
        self._cache_minutes = cache_minutes
        self._timeout = timeout
        self._baseline_count = baseline_count
        self._cache: dict[str, tuple[float, list[dict]]] = {}

    @staticmethod
    def _clean_query(query: str) -> str:
        cleaned = query.replace("?", "")
        cleaned = _FILLER_WORDS.sub("", cleaned)
        return re.sub(r"\s+", " ", cleaned).strip()

    async def search_news(self, query: str, max_results: int = 5) -> list[dict]:
        now = time.time()
        if query in self._cache:
            cached_time, cached_results = self._cache[query]
            if (now - cached_time) < self._cache_minutes * 60:
                logger.debug("News cache hit for query: %s", query)
                return cached_results

        api_key = os.environ.get("TAVILY_API_KEY")
        if not api_key:
            logger.warning("TAVILY_API_KEY not set, returning empty news results")
            return []

        cleaned = self._clean_query(query)

        try:
            from tavily import TavilyClient  # type: ignore[import-untyped]

            client = TavilyClient(api_key=api_key)
            raw_response: dict = await asyncio.to_thread(
                client.search,
                query=cleaned,
                max_results=max_results,
                search_depth="basic",
            )
        except Exception as exc:
            logger.warning("Tavily search failed for '%s': %s", cleaned, exc)
            return []

        results: list[dict] = []
        for item in raw_response.get("results", []):
            content = item.get("content", "")
            results.append(
                {
                    "title": item.get("title", ""),
                    "snippet": content[:500],
                    "url": item.get("url", ""),
                    "published_date": item.get("published_date"),
                }
            )

        self._cache[query] = (now, results)
        return results

    def get_cached(self, query: str) -> list[dict] | None:
        cached = self._cache.get(query)
        if cached is None:
            return None
        cached_time, cached_results = cached
        if (time.time() - cached_time) >= self._cache_minutes * 60:
            self._cache.pop(query, None)
            return None
        return cached_results

    def get_news_volume(self, results: list[dict]) -> tuple[int, float]:
        count = len(results)
        ratio = count / max(self._baseline_count, 1)
        return count, ratio
