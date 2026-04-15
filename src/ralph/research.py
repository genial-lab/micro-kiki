from __future__ import annotations

import logging
import re
from datetime import datetime
from pathlib import Path

from src.search.base import SearchBackend

logger = logging.getLogger(__name__)


class StoryResearcher:
    """Pre-story research: searches web + papers before implementation."""

    def __init__(
        self,
        exa_backend: SearchBackend,
        scholar_backend: SearchBackend,
        output_dir: Path | str = ".ralph/research",
    ) -> None:
        self._exa = exa_backend
        self._scholar = scholar_backend
        self._output_dir = Path(output_dir)
        self._output_dir.mkdir(parents=True, exist_ok=True)

    def extract_keywords(self, story: dict) -> str:
        title = story.get("title", "")
        desc = story.get("description", "")
        combined = f"{title} {desc}"
        combined = re.sub(r"arxiv \d+\.\d+", "", combined)
        combined = re.sub(r"[^\w\s-]", " ", combined)
        return " ".join(combined.split()[:10])

    async def research_story(self, story: dict) -> Path:
        story_id = story.get("id", "unknown")
        keywords = self.extract_keywords(story)

        web_results = await self._exa.search(keywords, max_results=5)
        paper_results = await self._scholar.search(keywords, max_results=5)

        output_path = self._output_dir / f"{story_id}.md"
        lines = [
            f"# Research: {story.get('title', story_id)}",
            "",
            f"Date: {datetime.now().isoformat()[:10]}",
            f"Keywords: {keywords}",
            "",
            "## Web Results",
            "",
        ]
        for r in web_results:
            lines.append(f"- **{r.title}** — [{r.url}]({r.url})")
            lines.append(f"  {r.snippet[:200]}")
            lines.append("")

        lines.append("## Papers")
        lines.append("")
        for r in paper_results:
            meta = r.metadata
            year = meta.get("year", "?")
            lines.append(f"- **{r.title}** ({year}) — [{r.url}]({r.url})")
            lines.append(f"  {r.snippet[:200]}")
            lines.append("")

        output_path.write_text("\n".join(lines))
        logger.info("Research for %s saved to %s", story_id, output_path)
        return output_path
