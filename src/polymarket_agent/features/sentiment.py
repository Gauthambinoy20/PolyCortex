import asyncio
import json
import logging
import os
import re
import sqlite3
import time
from dataclasses import dataclass

logger = logging.getLogger(__name__)

SENTIMENT_PROMPT = """You are analyzing prediction markets. For each market below, estimate the TRUE probability it resolves YES.
Consider: recent news, historical base rates, expert consensus.
Return ONLY a JSON array, one object per market, in order:
[{{"score": float (-1 to +1), "confidence": float (0 to 1), "reasoning": "one sentence"}}]

Markets:
{markets_text}"""


@dataclass
class SentimentResult:
    score: float  # -1 to +1
    confidence: float  # 0 to 1
    reasoning: str


class SentimentAnalyzer:
    """LLM-powered sentiment analysis with SQLite caching and batched API calls."""

    _llm_semaphore = asyncio.Semaphore(3)  # Max 3 concurrent Anthropic calls

    def __init__(
        self,
        cache_db: str = "data/sentiment_cache.db",
        cache_minutes: int = 30,
        model: str = "claude-haiku-4-5-20251001",
        timeout: float = 30.0,
    ) -> None:
        self._cache_db: str = cache_db
        self._cache_ttl: float = cache_minutes * 60.0
        self._model: str = model
        self._timeout: float = timeout
        self._init_db()

    def _init_db(self) -> None:
        os.makedirs(os.path.dirname(self._cache_db) or ".", exist_ok=True)
        self._conn = sqlite3.connect(self._cache_db, check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA busy_timeout=5000")
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS cache ("
            "  market_id TEXT PRIMARY KEY,"
            "  timestamp REAL,"
            "  score REAL,"
            "  confidence REAL,"
            "  reasoning TEXT"
            ")"
        )
        self._conn.commit()

    def _get_cached(self, market_id: str) -> SentimentResult | None:
        row = self._conn.execute(
            "SELECT timestamp, score, confidence, reasoning FROM cache WHERE market_id = ?",
            (market_id,),
        ).fetchone()

        if row is None:
            return None

        ts, score, confidence, reasoning = row
        if (time.time() - ts) > self._cache_ttl:
            return None

        return SentimentResult(
            score=float(score),
            confidence=float(confidence),
            reasoning=str(reasoning),
        )

    def get_cached(self, market_id: str) -> SentimentResult | None:
        """Public API for retrieving cached sentiment results."""
        return self._get_cached(market_id)

    def _save_cache(self, market_id: str, result: SentimentResult) -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO cache (market_id, timestamp, score, confidence, reasoning) VALUES (?, ?, ?, ?, ?)",
            (market_id, time.time(), result.score, result.confidence, result.reasoning),
        )
        self._conn.commit()

    async def analyze_batch(self, markets: list[tuple[str, str, str]]) -> dict[str, SentimentResult]:
        """Analyze a batch of markets, returning sentiment for each.

        Args:
            markets: List of (market_id, question, description) tuples.

        Returns:
            Dict mapping market_id to SentimentResult.
        """
        if not markets:
            return {}

        results: dict[str, SentimentResult] = {}
        uncached: list[tuple[int, str, str, str]] = []  # (original_idx, id, question, desc)

        # Check cache first
        for idx, (market_id, question, description) in enumerate(markets):
            cached = self._get_cached(market_id)
            if cached is not None:
                results[market_id] = cached
                logger.debug("Cache hit for market %s", market_id)
            else:
                uncached.append((idx, market_id, question, description))

        if not uncached:
            return results

        # Check API key
        api_key: str | None = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            logger.warning("ANTHROPIC_API_KEY not set — returning neutral sentiment for all markets")
            for _, market_id, _, _ in uncached:
                neutral = SentimentResult(score=0.0, confidence=0.0, reasoning="no API key")
                results[market_id] = neutral
            return results

        # Build prompt
        lines: list[str] = []
        for i, (_, _, question, description) in enumerate(uncached, start=1):
            lines.append(f"{i}. {question} — {description}")
        markets_text: str = "\n".join(lines)
        prompt: str = SENTIMENT_PROMPT.format(markets_text=markets_text)

        # Call API
        parsed: list[dict] | None = await self._call_api(api_key, prompt)

        if parsed is None or len(parsed) != len(uncached):
            if parsed is not None:
                logger.warning("Response count mismatch: got %d, expected %d", len(parsed), len(uncached))
            for _, market_id, _, _ in uncached:
                error_result = SentimentResult(score=0.0, confidence=0.0, reasoning="error")
                results[market_id] = error_result
            return results

        # Map results back
        for (_, market_id, _, _), item in zip(uncached, parsed, strict=True):
            try:
                score = max(-1.0, min(1.0, float(item.get("score", 0.0))))
                confidence = max(0.0, min(1.0, float(item.get("confidence", 0.0))))
                reasoning = str(item.get("reasoning", ""))
                result = SentimentResult(score=score, confidence=confidence, reasoning=reasoning)
            except (TypeError, ValueError) as exc:
                logger.warning("Failed to parse result for %s: %s", market_id, exc)
                result = SentimentResult(score=0.0, confidence=0.0, reasoning="error")

            results[market_id] = result
            self._save_cache(market_id, result)

        return results

    async def analyze_single(self, market_id: str, question: str, description: str = "") -> SentimentResult:
        """Convenience wrapper for a single market."""
        batch = await self.analyze_batch([(market_id, question, description)])
        return batch.get(
            market_id,
            SentimentResult(score=0.0, confidence=0.0, reasoning="error"),
        )

    async def _call_api(self, api_key: str, prompt: str) -> list[dict] | None:
        """Send prompt to Claude and parse JSON response."""
        try:
            import anthropic
        except ImportError:
            logger.error("anthropic package not installed")
            return None

        client = anthropic.Anthropic(api_key=api_key)

        def _sync_call() -> str:
            response = client.messages.create(
                model=self._model,
                max_tokens=1024,
                timeout=self._timeout,
                messages=[{"role": "user", "content": prompt}],
            )
            return str(getattr(response.content[0], "text", ""))

        try:
            async with self._llm_semaphore:
                raw_text: str = await asyncio.to_thread(_sync_call)
        except Exception as exc:
            logger.error("Anthropic API call failed: %s", exc)
            return None

        return self._parse_json_response(raw_text)

    @staticmethod
    def _parse_json_response(text: str) -> list[dict] | None:
        """Extract and parse a JSON array from the API response text."""
        # Try direct parse first
        try:
            data = json.loads(text)
            if isinstance(data, list):
                return data
        except json.JSONDecodeError:
            pass

        # Try extracting from markdown fences
        fence_match = re.search(r"```(?:json)?\s*\n?(.*?)\n?\s*```", text, re.DOTALL)
        if fence_match:
            try:
                data = json.loads(fence_match.group(1))
                if isinstance(data, list):
                    return data
            except json.JSONDecodeError:
                pass

        # Try finding a JSON array anywhere in the text
        bracket_match = re.search(r"\[.*\]", text, re.DOTALL)
        if bracket_match:
            try:
                data = json.loads(bracket_match.group(0))
                if isinstance(data, list):
                    return data
            except json.JSONDecodeError:
                pass

        logger.warning("Could not parse JSON from response: %.200s", text)
        return None
