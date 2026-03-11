import json
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

from botocore.exceptions import ClientError


QUALLM_JSON_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "rationale": {"type": "string"},
        "option": {"type": "string", "enum": ["yes", "no"]},
    },
    "required": ["rationale", "option"],
    "additionalProperties": False,
}


def extract_json(txt: str) -> Optional[Dict[str, Any]]:
    s = (txt or "").strip()
    i = s.find("{")
    j = s.rfind("}")
    if i >= 0 and j > i:
        try:
            return json.loads(s[i : j + 1])
        except Exception:
            return None
    return None


def _response_format_block(schema_name: str = "quallm_json") -> Dict[str, Any]:
    return {
        "type": "json_schema",
        "json_schema": {
            "name": schema_name,
            "schema": QUALLM_JSON_SCHEMA,
        },
    }


def _looks_like_llama(model_id: str) -> bool:
    return model_id.startswith("meta.llama") or model_id.startswith("meta.llama3") or model_id.startswith("meta.llama3-") or model_id.startswith("meta.llama3-1")


def _looks_like_deepseek(model_id: str) -> bool:
    return "deepseek" in model_id.lower()


def _is_bedrock_structured_supported_model(model_id: str) -> bool:
    return model_id == "deepseek.v3-v1:0"


def _parse_raw_text(model_id: str, data: Dict[str, Any]) -> str:
    if "generation" in data and isinstance(data.get("generation"), str):
        return data.get("generation") or ""
    if "choices" in data and isinstance(data.get("choices"), list) and data["choices"]:
        first = data["choices"][0]
        if isinstance(first, dict):
            return first.get("text", "") or ""
    out = data.get("output")
    if isinstance(out, dict):
        msg = out.get("message")
        if isinstance(msg, dict):
            content = msg.get("content")
            if isinstance(content, list) and content:
                c0 = content[0]
                if isinstance(c0, dict) and "text" in c0:
                    return c0.get("text", "") or ""
                if isinstance(c0, str):
                    return c0
            if isinstance(content, str):
                return content
    if "content" in data and isinstance(data["content"], str):
        return data["content"]
    return json.dumps(data)


@dataclass
class BedrockLLMRouter:
    bedrock_client: Any
    model_id: str
    temperature: float = 0.3
    top_p: float = 0.9
    max_tokens: int = 512
    max_gen_len: int = 512

    structured_mode: str = "auto"

    _STRUCTURED_CACHE: Dict[str, bool] = None

    def __post_init__(self):
        if self._STRUCTURED_CACHE is None:
            if not hasattr(BedrockLLMRouter, "_STRUCTURED_CACHE_SHARED"):
                BedrockLLMRouter._STRUCTURED_CACHE_SHARED = {}
            self._STRUCTURED_CACHE = BedrockLLMRouter._STRUCTURED_CACHE_SHARED

        self.structured_mode = (self.structured_mode or "auto").lower().strip()
        if self.structured_mode not in ("off", "on", "auto"):
            self.structured_mode = "auto"

    def _wrap_prompt(self, prompt_text: str) -> str:
        if _looks_like_deepseek(self.model_id):
            return f"<｜begin▁of▁sentence｜><｜User｜>{prompt_text}<｜Assistant｜><think>\n"
        return prompt_text

    def _invoke_once(self, prompt_text: str, use_structured: bool) -> Tuple[str, bool]:
        wrapped = self._wrap_prompt(prompt_text)

        body: Dict[str, Any] = {
            "prompt": wrapped,
            "temperature": float(self.temperature),
            "top_p": float(self.top_p),
        }

        if _looks_like_llama(self.model_id):
            body["max_gen_len"] = int(min(self.max_gen_len, 2048))
        else:
            body["max_tokens"] = int(self.max_tokens)

        if use_structured:
            body["response_format"] = _response_format_block()

        resp = self.bedrock_client.invoke_model(
            modelId=self.model_id,
            body=json.dumps(body),
            accept="application/json",
            contentType="application/json",
        )
        data = json.loads(resp["body"].read())
        raw = _parse_raw_text(self.model_id, data)
        return raw, use_structured

    def _should_try_structured(self) -> bool:
        if self.structured_mode == "off":
            return False
        if self.structured_mode == "on":
            return True
        if self.model_id in self._STRUCTURED_CACHE:
            return self._STRUCTURED_CACHE[self.model_id]
        return True

    def _mark_structured(self, ok: bool):
        if self.structured_mode == "auto":
            self._STRUCTURED_CACHE[self.model_id] = bool(ok)

    def generate_json(self, prompt_text: str) -> Tuple[Optional[Dict[str, Any]], bool, Optional[str]]:
        """
        Returns (obj, used_structured, raw_text_if_failed)
        """
        if self._should_try_structured():
            try:
                raw, _ = self._invoke_once(prompt_text, use_structured=True)
                obj = None
                try:
                    obj = json.loads((raw or "").strip())
                except Exception:
                    obj = extract_json(raw)
                if isinstance(obj, dict) and "option" in obj and "rationale" in obj:
                    self._mark_structured(True)
                    return obj, True, None

                self._mark_structured(False)
            except ClientError:
                self._mark_structured(False)
            except Exception:
                self._mark_structured(False)

        try:
            raw, _ = self._invoke_once(prompt_text, use_structured=False)
            obj = extract_json(raw)
            if isinstance(obj, dict) and "option" in obj and "rationale" in obj:
                return obj, False, None
            return None, False, raw
        except Exception as e:
            return None, False, str(e)
        
def generate_raw(self, prompt_text: str) -> Tuple[str, bool]:
    """
    Returns (raw_text, used_structured)
    """
    if self._should_try_structured():
        try:
            raw, _ = self._invoke_once(prompt_text, use_structured=True)
            self._mark_structured(True)
            return raw, True
        except ClientError:
            self._mark_structured(False)
        except Exception:
            self._mark_structured(False)

    raw, _ = self._invoke_once(prompt_text, use_structured=False)
    return raw, False