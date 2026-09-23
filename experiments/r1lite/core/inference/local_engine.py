"""In-process inference engine: runs the G05 policy directly, no WebSocket hop.

Unlike WebSocketClientEngine, this engine pulls torch and the full g05
package into the robot-side client process. Only use it on a machine that:
  - has a CUDA GPU meeting the repo's Inference requirements (see the main
    README's GPU Requirements table), and
  - runs this client from the repo-root G05 .venv (the one created by
    `uv sync`), not a separate/minimal client-only environment.

Reuses scripts/serve_policy.py's `setup()` (model/processor loading) and
`ChunkedPolicyWrapper` (recompute-vs-cache chunk serving) so behavior stays
identical to the networked server path.
"""

from __future__ import annotations

import asyncio
import logging
import sys
from pathlib import Path
from typing import Any, Dict

from typing_extensions import override

from core.inference.inference_engine import InferenceEngine

logger = logging.getLogger(__name__)

_SCRIPTS_DIR = Path(__file__).resolve().parents[4] / "scripts"


class LocalInferenceEngine(InferenceEngine):
    """Loads the G05 policy in-process and serves actions with chunk caching."""

    def __init__(self, config: Dict[str, Any]):
        local_cfg = config["local"]
        self._ckpt_path = local_cfg["ckpt_path"]
        self._device = local_cfg.get("device", "cuda")
        self._action_steps = int(
            local_cfg.get("action_steps", config["basic"].get("action_steps", 16))
        )
        self._overrides = local_cfg.get("overrides", [])
        self._policy_wrapper = None

    @override
    def connect(self) -> None:
        if str(_SCRIPTS_DIR) not in sys.path:
            sys.path.insert(0, str(_SCRIPTS_DIR))
        import serve_policy  # repo script, not an installed package

        from g05.models.g05.inferencer import PolicyInferencer

        run_dir = serve_policy.find_run_dir(self._ckpt_path)
        cfg = serve_policy.load_config_from_run_dir(run_dir, self._ckpt_path, self._overrides)

        eval_embodiment = cfg.get("eval_embodiment", None)
        if eval_embodiment and "embodiment_datasets" in cfg.data:
            from g05.utils.eval.eval_utils import filter_embodiment

            filter_embodiment(cfg, eval_embodiment)

        policy, processor = serve_policy.setup(cfg, device=self._device)
        inferencer = PolicyInferencer(policy, processor, device=self._device)
        self._policy_wrapper = serve_policy.ChunkedPolicyWrapper(
            inferencer, processor, action_steps=self._action_steps
        )
        logger.info(
            "Local G05 policy loaded on %s (action_steps=%d).",
            self._device,
            self._action_steps,
        )

    @override
    def predict_action(self, obs: Dict) -> Dict:
        action, cot_text = asyncio.run(self._policy_wrapper.get_action(obs))
        result = {"action": action, "need_obs": self._policy_wrapper.need_obs}
        if cot_text is not None:
            result["cot_text"] = cot_text
        return result
