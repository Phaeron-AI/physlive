from __future__ import annotations

import inspect
import logging
from typing import Any, Callable, Dict

logger = logging.getLogger(__name__)

__all__ = ["call_filtered"]


def call_filtered(target: Callable[..., Any], params: Dict[str, Any]) -> Any:
  try:
    sig = inspect.signature(target)
  except (TypeError, ValueError):
    return target(**params)

  has_var_kw = any(
    p.kind is inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()
  )
  if has_var_kw:
    return target(**params)

  accepted = set(sig.parameters)
  used = {k: v for k, v in params.items() if k in accepted}
  dropped = set(params) - set(used)
  if dropped:
    logger.debug("dropping unused config keys for %s: %s", target, sorted(dropped))
  return target(**used)