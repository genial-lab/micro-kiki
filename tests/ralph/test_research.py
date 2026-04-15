from __future__ import annotations

import pytest
from unittest.mock import AsyncMock
from src.ralph.research import StoryResearcher
from src.search.base import SearchResult


@pytest.fixture
def researcher(tmp_path):
    mock_exa = AsyncMock()
    mock_exa.search.return_value = [
        SearchResult(title="MoE Guide", url="https://a.com", snippet="MoE tutorial", source="exa", metadata={}),
    ]
    mock_scholar = AsyncMock()
    mock_scholar.search.return_value = [
        SearchResult(title="MoLoRA Paper", url="https://arxiv.org/abs/2603.15965",
                     snippet="We propose...", source="scholar", metadata={"year": 2026}),
    ]
    return StoryResearcher(
        exa_backend=mock_exa, scholar_backend=mock_scholar,
        output_dir=tmp_path / "research",
    )


class TestStoryResearcher:
    @pytest.mark.asyncio
    async def test_research_produces_markdown(self, researcher, tmp_path):
        story = {"id": "story-9", "title": "MoE-LoRA stack trainer",
                 "description": "Implement trainer with OPLoRA support"}
        output_path = await researcher.research_story(story)
        assert output_path.exists()
        content = output_path.read_text()
        assert "MoE Guide" in content
        assert "MoLoRA Paper" in content

    def test_extracts_keywords(self, researcher):
        story = {"id": "story-15", "title": "Forgetting check framework",
                 "description": "gradient subspace overlap metric, arxiv 2603.02224"}
        keywords = researcher.extract_keywords(story)
        assert "forgetting" in keywords.lower() or "gradient" in keywords.lower()
