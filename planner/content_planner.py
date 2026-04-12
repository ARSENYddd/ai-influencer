"""
Assigns top trends to personas for content scheduling.
"""
import logging

logger = logging.getLogger(__name__)

def assign_trends_to_personas(personas: list[dict], trends: list[dict]) -> list[dict]:
    """
    Returns list of (persona, trend) assignments.
    Each persona gets a trend; cycles through trends if fewer than personas.
    """
    if not trends:
        logger.warning("No trends available for assignment")
        return []

    assignments = []
    for i, persona in enumerate(personas):
        # Prefer trend that matches best_persona, fallback to round-robin
        matched = next(
            (t for t in trends if t.get("best_persona") == persona["id"]),
            trends[i % len(trends)]
        )
        assignments.append({"persona": persona, "trend": matched})
        logger.info(f"Assigned trend '{matched.get('dance_type')}' to {persona['id']}")

    return assignments
