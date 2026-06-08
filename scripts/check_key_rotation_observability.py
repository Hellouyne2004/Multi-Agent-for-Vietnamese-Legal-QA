"""Print safe Gemini key-rotation routing without calling an LLM provider."""

from __future__ import annotations

import hashlib
import os


PURPOSES = [
    "router",
    "retriever",
    "grader",
    "web_searcher",
    "generator",
    "hallucination_grader",
]


def configured_key_names() -> list[str]:
    names: list[str] = []
    if os.getenv("GEMINI_API_KEY"):
        names.append("GEMINI_API_KEY")
    for index in range(1, 11):
        name = f"GEMINI_API_KEY_{index}"
        if os.getenv(name):
            names.append(name)
    return names


def key_order_for_purpose(purpose: str, total_keys: int) -> list[int]:
    if total_keys <= 0:
        return []
    if purpose != "default" and total_keys > 1:
        offset = int(hashlib.md5(purpose.encode()).hexdigest(), 16) % total_keys
        return list(range(offset, total_keys)) + list(range(0, offset))
    return list(range(total_keys))


def main() -> None:
    names = configured_key_names()
    simulated = False
    if not names:
        names = ["GEMINI_API_KEY", "GEMINI_API_KEY_1", "GEMINI_API_KEY_2", "GEMINI_API_KEY_3"]
        simulated = True

    print("Raw API keys are never printed.")
    if simulated:
        print("No Gemini keys found in the environment; showing a 4-key simulated routing plan.")
    else:
        print(f"Configured Gemini key slots: {len(names)}")

    for purpose in PURPOSES:
        order = key_order_for_purpose(purpose, len(names))
        primary = order[0] if order else None
        fallback = order[1:]
        print(
            f"[LLM] agent={purpose} primary_key_index={primary} "
            f"fallback_key_indices={fallback} json_mode={purpose in {'router', 'grader', 'generator', 'hallucination_grader'}}"
        )


if __name__ == "__main__":
    main()
