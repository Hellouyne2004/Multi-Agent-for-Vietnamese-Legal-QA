"""Lazy public exports for the workflow agents."""

from importlib import import_module

__all__ = [
    "retriever_node",
    "router_node",
    "grader_node",
    "web_searcher_node",
    "generator_node",
    "hallucination_grader_node",
]


_NODE_MODULES = {
    "retriever_node": ".retriever",
    "router_node": ".router",
    "grader_node": ".grader",
    "web_searcher_node": ".web_searcher",
    "generator_node": ".generator",
    "hallucination_grader_node": ".hallucination_grader",
}


def __getattr__(name: str):
    module_name = _NODE_MODULES.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    return getattr(import_module(module_name, __name__), name)
