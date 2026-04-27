"""Tests for the FastAPI backend API."""

import pytest
from fastapi.testclient import TestClient

from gale_shapley_algorithm._api.app import app


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)


class TestHealth:
    """Tests for the health endpoint."""

    def test_health(self, client: TestClient) -> None:
        response = client.get("/api/health")
        assert response.status_code == 200
        assert response.json() == {"status": "ok"}


class TestMatching:
    """Tests for the POST /api/matching endpoint."""

    def test_basic_2x2(self, client: TestClient) -> None:
        response = client.post(
            "/api/matching",
            json={
                "proposer_preferences": {"alice": ["bob", "charlie"], "dave": ["charlie", "bob"]},
                "responder_preferences": {"bob": ["alice", "dave"], "charlie": ["dave", "alice"]},
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert data["rounds"] >= 1
        assert data["all_matched"] is True
        assert len(data["matches"]) == 2

    def test_response_includes_stability_info(self, client: TestClient) -> None:
        response = client.post(
            "/api/matching",
            json={
                "proposer_preferences": {"m1": ["w1"], "m2": ["w1"]},
                "responder_preferences": {"w1": ["m1", "m2"]},
            },
        )
        data = response.json()
        assert "is_stable" in data
        assert "is_individually_rational" in data
        assert "blocking_pairs" in data

    def test_asymmetric_preferences_self_matches(self, client: TestClient) -> None:
        response = client.post(
            "/api/matching",
            json={
                "proposer_preferences": {"m1": ["w1"], "m2": ["w1"], "m3": ["w1"]},
                "responder_preferences": {"w1": ["m1", "m2", "m3"]},
            },
        )
        data = response.json()
        assert len(data["matches"]) == 1
        assert len(data["self_matches"]) > 0
        assert data["all_matched"] is False

    def test_empty_preferences(self, client: TestClient) -> None:
        response = client.post(
            "/api/matching",
            json={
                "proposer_preferences": {"m1": []},
                "responder_preferences": {"w1": []},
            },
        )
        data = response.json()
        assert "m1" in data["self_matches"]
        assert "w1" in data["self_matches"]
        assert data["all_matched"] is False

    def test_minimal_1x1_matching(self, client: TestClient) -> None:
        """Minimal edge case: single proposer and single responder."""
        response = client.post(
            "/api/matching",
            json={
                "proposer_preferences": {"a": ["b"]},
                "responder_preferences": {"b": ["a"]},
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert data["matches"] == {"a": "b"}
        assert data["all_matched"] is True
        assert data["is_stable"] is True
        assert data["self_matches"] == []
        assert data["unmatched"] == []

    def test_rejects_unknown_responder_name(self, client: TestClient) -> None:
        """A typo in a proposer's preference list must not silently self-match."""
        response = client.post(
            "/api/matching",
            json={
                "proposer_preferences": {"alice": ["bob_typo"]},
                "responder_preferences": {"bob": ["alice"]},
            },
        )
        assert response.status_code == 422
        assert "bob_typo" in response.text

    def test_rejects_unknown_proposer_name(self, client: TestClient) -> None:
        """A typo in a responder's preference list must not silently self-match."""
        response = client.post(
            "/api/matching",
            json={
                "proposer_preferences": {"alice": ["bob"]},
                "responder_preferences": {"bob": ["alice_typo"]},
            },
        )
        assert response.status_code == 422
        assert "alice_typo" in response.text


class TestMatchingSteps:
    """Tests for the POST /api/matching/steps endpoint."""

    def test_steps_structure(self, client: TestClient) -> None:
        response = client.post(
            "/api/matching/steps",
            json={
                "proposer_preferences": {"alice": ["bob", "charlie"], "dave": ["charlie", "bob"]},
                "responder_preferences": {"bob": ["alice", "dave"], "charlie": ["dave", "alice"]},
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert "steps" in data
        assert "final_result" in data
        assert isinstance(data["steps"], list)
        assert len(data["steps"]) >= 1

    def test_step_round_fields(self, client: TestClient) -> None:
        response = client.post(
            "/api/matching/steps",
            json={
                "proposer_preferences": {"A": ["X"], "B": ["X"]},
                "responder_preferences": {"X": ["A", "B"]},
            },
        )
        data = response.json()
        step = data["steps"][0]
        assert "round" in step
        assert "proposals" in step
        assert "rejections" in step
        assert "tentative_matches" in step
        assert "self_matches" in step
        assert step["round"] == 1

    def test_multi_round_rejections(self, client: TestClient) -> None:
        """Both proposers prefer X, X prefers A: B gets rejected round 1."""
        response = client.post(
            "/api/matching/steps",
            json={
                "proposer_preferences": {"A": ["X", "Y"], "B": ["X", "Y"]},
                "responder_preferences": {"X": ["A", "B"], "Y": ["B", "A"]},
            },
        )
        data = response.json()
        assert len(data["steps"]) == 2

        # Round 1: both propose to X, B rejected
        step1 = data["steps"][0]
        assert len(step1["proposals"]) == 2
        assert len(step1["rejections"]) == 1
        assert step1["rejections"][0]["proposer"] == "B"
        assert step1["rejections"][0]["responder"] == "X"

        # Round 2: B proposes to Y
        step2 = data["steps"][1]
        assert len(step2["proposals"]) == 1
        assert step2["proposals"][0]["proposer"] == "B"

    def test_final_result_matches_matching_endpoint(self, client: TestClient) -> None:
        prefs = {
            "proposer_preferences": {"alice": ["bob", "charlie"], "dave": ["charlie", "bob"]},
            "responder_preferences": {"bob": ["alice", "dave"], "charlie": ["dave", "alice"]},
        }
        matching_response = client.post("/api/matching", json=prefs).json()
        steps_response = client.post("/api/matching/steps", json=prefs).json()
        assert steps_response["final_result"]["matches"] == matching_response["matches"]
        assert steps_response["final_result"]["rounds"] == matching_response["rounds"]

    def test_steps_self_matches(self, client: TestClient) -> None:
        response = client.post(
            "/api/matching/steps",
            json={
                "proposer_preferences": {"A": ["X"], "B": []},
                "responder_preferences": {"X": ["A"], "Y": ["B"]},
            },
        )
        data = response.json()
        # B has no acceptable responders, should self-match
        all_step_self_matches = []
        for step in data["steps"]:
            all_step_self_matches.extend(step["self_matches"])
        assert "B" in all_step_self_matches
