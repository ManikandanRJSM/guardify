from guardify.guardrails.logistic_regression import LogisticRegressionGuardrail

_default_guardrail: LogisticRegressionGuardrail | None = None


def _get_default() -> LogisticRegressionGuardrail:
    global _default_guardrail
    if _default_guardrail is None:
        _default_guardrail = LogisticRegressionGuardrail()
    return _default_guardrail


def detect(text: str) -> dict:
    """
    Detect whether a prompt is a jailbreak / injection attempt.

    Returns:
        {"label": "safe" | "injection", "score": float}
    """
    return _get_default().predict(text)


__all__ = ["detect", "LogisticRegressionGuardrail"]
