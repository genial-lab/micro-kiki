from __future__ import annotations

from src.ralph.self_review import CodeReview, ReviewResult


class TestCodeReview:
    def test_review_template_has_required_fields(self):
        review = CodeReview()
        template = review.get_template()
        assert "bugs" in template
        assert "edge_cases" in template
        assert "perf" in template
        assert "security" in template
        assert "style" in template

    def test_parse_clean_review(self):
        review = CodeReview()
        raw = '{"bugs": [], "edge_cases": [], "perf": [], "security": [], "style": [], "approved": true, "summary": "LGTM"}'
        result = review.parse_review(raw)
        assert isinstance(result, ReviewResult)
        assert result.approved is True
        assert result.total_issues == 0

    def test_parse_review_with_issues(self):
        review = CodeReview()
        raw = '{"bugs": ["off by one"], "edge_cases": ["empty input"], "perf": [], "security": [], "style": ["naming"], "approved": false, "summary": "Needs fixes"}'
        result = review.parse_review(raw)
        assert result.approved is False
        assert result.total_issues == 3

    def test_max_passes_limit(self):
        review = CodeReview(max_passes=3)
        assert review.max_passes == 3
