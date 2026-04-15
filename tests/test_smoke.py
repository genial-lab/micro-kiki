from __future__ import annotations


class TestSmokeImports:
    def test_search_package(self):
        from src.search.base import SearchResult, SearchBackend
        assert SearchResult is not None

    def test_critique_package(self):
        from src.critique.best_of_n import BestOfN
        assert BestOfN is not None

    def test_distill_package(self):
        from src.distill.teacher_client import TeacherClient
        assert TeacherClient is not None

    def test_base_package(self):
        from src.base.loader import BaseModelLoader
        assert BaseModelLoader is not None


class TestSmokeFixtures:
    def test_tmp_model_dir(self, tmp_model_dir):
        assert (tmp_model_dir / "config.json").exists()

    def test_mock_teacher(self, mock_teacher):
        assert mock_teacher.generate is not None

    def test_sample_prompts(self, sample_prompts):
        assert len(sample_prompts) == 5
