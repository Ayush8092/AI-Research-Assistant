"""
Unit Tests — 20 tests, no Ollama or network required.
Run: python -m pytest tests/ -v
"""

import sys
import unittest
import tempfile
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parent.parent))


# ------------------------------------------------------------------ #
#  State Schema                                                        #
# ------------------------------------------------------------------ #

class TestStateSchema(unittest.TestCase):

    def test_create_initial_state(self):
        from agents.state import create_initial_state
        s = create_initial_state("test query", "sess-123")
        self.assertEqual(s["query"], "test query")
        self.assertEqual(s["session_id"], "sess-123")
        self.assertIsInstance(s["subtopics"], list)
        self.assertIsInstance(s["errors"], list)

    def test_initial_state_filters(self):
        from agents.state import create_initial_state
        s = create_initial_state("q", "s", filters={"year_after": 2022})
        self.assertEqual(s["filters"]["year_after"], 2022)


# ------------------------------------------------------------------ #
#  Planner Agent                                                       #
# ------------------------------------------------------------------ #

class TestPlannerAgent(unittest.TestCase):

    @patch("agents.planner_agent.llm_generate")
    def test_generates_subtopics(self, mock_llm):
        mock_llm.return_value = (
            "1. Deep learning architectures\n"
            "2. Healthcare datasets\n"
            "3. Clinical decision support\n"
            "4. Evaluation metrics\n"
            "5. Safety concerns"
        )
        from agents.state import create_initial_state
        from agents.planner_agent import planner_agent
        state  = create_initial_state("agentic AI in healthcare", "sess-1")
        result = planner_agent(state)
        self.assertGreater(len(result["subtopics"]), 2)
        self.assertIn("search_agent", result["agent_plan"])

    @patch("agents.planner_agent.llm_generate")
    def test_fallback_subtopics(self, mock_llm):
        mock_llm.return_value = "[LLM unavailable]"
        from agents.state import create_initial_state
        from agents.planner_agent import planner_agent
        state  = create_initial_state("medical AI", "sess-2")
        result = planner_agent(state)
        self.assertGreater(len(result["subtopics"]), 0)

    def test_refine_year_filter(self):
        from agents.state import create_initial_state
        from agents.planner_agent import refine_plan
        state  = create_initial_state("AI", "s")
        result = refine_plan(state, "focus on papers after 2022")
        self.assertEqual(result["filters"].get("year_after"), 2022)

    def test_refine_exclude_surveys(self):
        from agents.state import create_initial_state
        from agents.planner_agent import refine_plan
        state  = create_initial_state("AI", "s")
        result = refine_plan(state, "exclude survey papers")
        self.assertTrue(result["filters"].get("exclude_surveys"))


# ------------------------------------------------------------------ #
#  Arxiv Client                                                        #
# ------------------------------------------------------------------ #

class TestArxivClient(unittest.TestCase):

    def test_domain_relevance_high(self):
        from utils.arxiv_client import ArxivClient, ArxivPaper
        client = ArxivClient()
        paper  = ArxivPaper(
            "id", "Clinical AI", "",
            "clinical patient hospital diagnosis",
            "2023", "2023", [], "", []
        )
        score = client._compute_domain_relevance(paper)
        self.assertGreater(score, 0.3)

    def test_domain_relevance_low(self):
        from utils.arxiv_client import ArxivClient, ArxivPaper
        client = ArxivClient()
        paper  = ArxivPaper(
            "id", "Quantum Computing", "",
            "quantum circuits entanglement",
            "2023", "2023", [], "", []
        )
        score = client._compute_domain_relevance(paper)
        self.assertLess(score, 0.3)

    def test_enrich_query(self):
        from utils.arxiv_client import ArxivClient
        client   = ArxivClient()
        enriched = client._enrich_query(
            "AI in healthcare", ["deep learning", "clinical trials"]
        )
        self.assertIn("AI in healthcare", enriched)


# ------------------------------------------------------------------ #
#  FAISS Store                                                         #
# ------------------------------------------------------------------ #

class TestFAISSStore(unittest.TestCase):

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def _store(self, suffix="1"):
        from vectorstore.faiss_store import FAISSStore
        return FAISSStore(
            index_path=f"{self.tmpdir}/idx{suffix}.bin",
            meta_path=f"{self.tmpdir}/meta{suffix}.json",
            max_docs=50
        )

    def test_add_and_search(self):
        store  = self._store("a")
        papers = [
            {"paper_id": "p1", "title": "Deep Learning X-Ray",
             "abstract": "CNN for chest X-ray diagnosis"},
            {"paper_id": "p2", "title": "NLP for EHR",
             "abstract": "BERT for clinical notes"}
        ]
        self.assertEqual(store.add_papers(papers), 2)
        results = store.search("neural network chest imaging", top_k=2)
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0]["paper_id"], "p1")

    def test_no_duplicate_adds(self):
        store = self._store("b")
        paper = [{"paper_id": "p1", "title": "Test", "abstract": "Test abstract"}]
        store.add_papers(paper)
        self.assertEqual(store.add_papers(paper), 0)

    def test_stats(self):
        store = self._store("c")
        stats = store.stats()
        self.assertIn("total_documents", stats)
        self.assertIn("max_documents", stats)


# ------------------------------------------------------------------ #
#  Memory Store (SQLite)                                               #
# ------------------------------------------------------------------ #

class TestMemoryStore(unittest.TestCase):

    def setUp(self):
        tmpdir = tempfile.mkdtemp()
        from database.memory_store import MemoryStore
        self.memory = MemoryStore(db_path=f"{tmpdir}/test.db")

    def test_create_and_get_session(self):
        self.memory.create_session("s1", "AI healthcare", ["methods"])
        session = self.memory.get_session("s1")
        self.assertIsNotNone(session)
        self.assertEqual(session["query"], "AI healthcare")

    def test_save_and_get_papers(self):
        self.memory.create_session("s2", "test")
        papers = [{
            "paper_id": "p1", "title": "Test", "authors": ["A"],
            "abstract": "", "published": "2023", "final_score": 0.75,
            "citation_count": 10, "venue": "NeurIPS", "score_breakdown": {}
        }]
        self.memory.save_papers("s2", papers)
        retrieved = self.memory.get_papers("s2")
        self.assertEqual(len(retrieved), 1)
        self.assertAlmostEqual(retrieved[0]["final_score"], 0.75)

    def test_metrics(self):
        self.memory.create_session("s3", "test")
        self.memory.save_metrics("s3", {
            "query_time_sec": 5.0, "papers_retrieved": 20,
            "papers_selected": 18, "score_mean": 0.6, "score_std": 0.1
        })
        m = self.memory.get_metrics("s3")
        self.assertEqual(len(m), 1)
        self.assertAlmostEqual(m[0]["query_time_sec"], 5.0)

    def test_conversation(self):
        self.memory.create_session("s4", "test")
        self.memory.add_message("s4", "user", "Hello")
        self.memory.add_message("s4", "assistant", "Hi!")
        history = self.memory.get_conversation("s4")
        self.assertEqual(len(history), 2)
        self.assertEqual(history[0]["role"], "user")


# ------------------------------------------------------------------ #
#  Critic Agent Scoring                                                #
# ------------------------------------------------------------------ #

class TestCriticScoring(unittest.TestCase):

    def test_citation_scores(self):
        from agents.critic_agent import _citation_score
        from utils.openalex_client import OpenAlexMetadata
        high = OpenAlexMetadata("", "", "", 1000, "", "high", 2023, True)
        low  = OpenAlexMetadata("", "", "", 1,    "", "low",  2020, False)
        self.assertAlmostEqual(_citation_score(high, 1000), 1.0, places=2)
        self.assertLess(_citation_score(low, 1000), _citation_score(high, 1000))
        self.assertGreater(_citation_score(None, 1000), 0)

    def test_recency_scores(self):
        from agents.critic_agent import _recency_score
        import datetime
        cy = datetime.datetime.now().year
        self.assertGreater(
            _recency_score(f"{cy}-01-01"),
            _recency_score("2015-01-01")
        )

    def test_venue_scores(self):
        from agents.critic_agent import _venue_score
        from utils.openalex_client import OpenAlexMetadata
        high = OpenAlexMetadata("", "", "", 0, "NeurIPS", "high", 2023, True)
        low  = OpenAlexMetadata("", "", "", 0, "Unknown", "low",  2023, False)
        self.assertGreater(_venue_score(high), _venue_score(low))


# ------------------------------------------------------------------ #
#  BibTeX Generator                                                    #
# ------------------------------------------------------------------ #

class TestBibTeX(unittest.TestCase):

    def test_entry_generation(self):
        from artifacts.bibtex_generator import _make_bibtex_entry, _make_cite_key
        p = {
            "paper_id":  "2301.12345",
            "title":     "Deep Learning for Segmentation",
            "authors":   ["John Smith", "Jane Doe"],
            "published": "2023-01-15",
            "arxiv_url": "https://arxiv.org/abs/2301.12345",
            "doi":       "",
            "venue":     ""
        }
        key   = _make_cite_key(p)
        self.assertIn("2023", key)
        self.assertIn("Smith", key)
        entry = _make_bibtex_entry(p)
        self.assertIn("@misc", entry)
        self.assertIn("John Smith", entry)


# ------------------------------------------------------------------ #
#  Knowledge Graph                                                     #
# ------------------------------------------------------------------ #

class TestKnowledgeGraph(unittest.TestCase):

    def test_entity_extraction(self):
        from knowledge_graph.graph_builder import _extract_entities, METHOD_KEYWORDS
        text     = "we use transformer and bert with federated learning"
        entities = _extract_entities(text, METHOD_KEYWORDS)
        self.assertIn("transformer", entities)
        self.assertIn("bert", entities)

    def test_graph_stats(self):
        from knowledge_graph.graph_builder import get_graph_stats
        entities = [
            {"type": "paper"},
            {"type": "method"},
            {"type": "dataset"}
        ]
        edges = [
            {"relation": "uses"},
            {"relation": "trained_on"}
        ]
        stats = get_graph_stats(entities, edges)
        self.assertEqual(stats["total_entities"], 3)
        self.assertEqual(stats["total_edges"], 2)

    def test_report_list_parsing(self):
        from artifacts.report_generator import _parse_list
        text  = "1. Deep learning\n2. Transformers\n• BERT\n- Random forest"
        items = _parse_list(text)
        self.assertEqual(len(items), 4)
        self.assertIn("Deep learning", items)


if __name__ == "__main__":
    unittest.main(verbosity=2)