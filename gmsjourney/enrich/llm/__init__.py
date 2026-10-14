"""The LLM enrichment arm: a model on the Microsoft gateway writes the customer turns; every turn is
gated and proved separately from the template arm (see validate.py for the full test)."""
from .client import CachedLLM, LLMSettings, LLMUnavailable, make_chat_model
from .extractor import LLMFrameExtractor
from .gates import LLMOutputGates
from .realizer import LLMRealizer
from .validate import LLMInvarianceValidator

__all__ = ["CachedLLM", "LLMSettings", "LLMUnavailable", "make_chat_model", "LLMFrameExtractor",
           "LLMOutputGates", "LLMRealizer", "LLMInvarianceValidator", "build_llm_arm"]


def build_llm_arm(store, oracle, reader, template_validator, settings: LLMSettings | None = None,
                  dual_read: str = "gate", adversarial: bool = True, max_attempts: int = 3,
                  asides: str = "verbatim", turn_fallback: str = "template", template_acts: tuple = ()):
    """Wire the arm: one cached client per role so the roles never share completions."""
    s = settings or LLMSettings()
    rephraser = CachedLLM(s, tag="rephrase")
    extractor = CachedLLM(LLMSettings(**{**s.__dict__, "temperature": 0.0}), tag="extract") if dual_read != "off" else None
    adversary = CachedLLM(LLMSettings(**{**s.__dict__, "temperature": 0.7}), tag="adversary") if adversarial else None
    realizer = LLMRealizer(rephraser, store, reader, template_validator.realizer, asides=asides)
    validator = LLMInvarianceValidator(store, oracle, reader, realizer, template_validator, dual_read=dual_read,
                                       extractor_llm=extractor, adversary_llm=adversary, max_attempts=max_attempts,
                                       turn_fallback=turn_fallback, template_acts=tuple(template_acts))
    return realizer, validator, {"rephrase": rephraser, "extract": extractor, "adversary": adversary}
