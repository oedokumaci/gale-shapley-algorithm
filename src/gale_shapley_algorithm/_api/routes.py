"""API route handlers."""

from fastapi import APIRouter

from gale_shapley_algorithm._api.models import MatchingRequest, MatchingResponse, StepsResponse
from gale_shapley_algorithm._api.step_through import _build_matching_response, run_step_through
from gale_shapley_algorithm.matching import _build_algorithm
from gale_shapley_algorithm.stability import check_stability

router = APIRouter(prefix="/api")


@router.get("/health")
def health() -> dict[str, str]:
    """Health check endpoint."""
    return {"status": "ok"}


@router.post("/matching")
def run_matching(req: MatchingRequest) -> MatchingResponse:
    """Run the Gale-Shapley algorithm and return results with stability info."""
    algorithm = _build_algorithm(req.proposer_preferences, req.responder_preferences)
    result = algorithm.execute()
    stability = check_stability(algorithm)
    return _build_matching_response(result, stability)


@router.post("/matching/steps")
def run_matching_steps(req: MatchingRequest) -> StepsResponse:
    """Run the algorithm step by step, returning per-round snapshots."""
    return run_step_through(req.proposer_preferences, req.responder_preferences)
