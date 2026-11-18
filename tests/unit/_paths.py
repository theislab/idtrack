"""The engine's raw steps of an explained path, to compare a conversion's path with the steps the search took."""

from idtrack.results import ExplainedPath


def raw_steps(path: ExplainedPath, indices: tuple[int, ...]) -> list[list]:
    """Return steps as the engine records them (lists).

    Args:
        path: The explanation.
        indices: Step indices.

    Returns:
        list[list]: ``[source, target, key]``, ``[source, target, 0, release]`` or ``[None, gene, None]``.
    """
    out: list[list] = []
    for index in indices:
        step = path.steps[index]
        if step.kind == "identity":
            out.append([None, step.target, None])
        elif step.kind in ("history", "self_loop"):
            out.append([step.source, step.target, step.key])
        else:
            out.append([step.source, step.target, 0, step.release])
    return out
