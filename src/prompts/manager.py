from __future__ import annotations

from typing import Any

from src.config.settings import CONFIG


class PromptManager:
    def get_prompt(
        self,
        prompt_name: str,
        variables: dict[str, Any] | None = None,
    ) -> tuple[str, str]:
        prompt_text = self._fetch_from_langfuse(prompt_name, variables=variables)
        if not prompt_text:
            raise RuntimeError(f"Prompt not found in Langfuse: {prompt_name}")
        return prompt_text, "langfuse"

    def _fetch_from_langfuse(
        self,
        prompt_name: str,
        variables: dict[str, Any] | None = None,
    ) -> str | None:
        if not CONFIG.prompts.langfuse_public_key or not CONFIG.prompts.langfuse_secret_key:
            return None
        try:
            from langfuse import Langfuse
        except Exception:  # noqa: BLE001
            return None

        try:
            client = Langfuse(
                public_key=CONFIG.prompts.langfuse_public_key,
                secret_key=CONFIG.prompts.langfuse_secret_key,
                host=CONFIG.prompts.langfuse_host,
            )
            prompt = client.get_prompt(prompt_name, label=CONFIG.prompts.langfuse_label)
            if isinstance(prompt, str):
                return self._render(prompt, variables)
            if hasattr(prompt, "prompt"):
                normalized = self._normalize_prompt_content(prompt.prompt)
                if normalized:
                    return self._render(normalized, variables)
            if hasattr(prompt, "compile") and variables:
                normalized_compiled = self._normalize_prompt_content(prompt.compile(**variables))
                if normalized_compiled:
                    return normalized_compiled
        except Exception:  # noqa: BLE001
            return None
        return None

    @staticmethod
    def _render(template: str, variables: dict[str, Any] | None = None) -> str:
        if not variables:
            return template
        rendered = template
        for key, value in variables.items():
            rendered = rendered.replace("{{" + key + "}}", str(value))
            rendered = rendered.replace("{" + key + "}", str(value))
        return rendered

    @staticmethod
    def _normalize_prompt_content(content: Any) -> str | None:
        if isinstance(content, str):
            return content
        if isinstance(content, dict):
            if "prompt" in content and isinstance(content["prompt"], str):
                return content["prompt"]
            if "messages" in content and isinstance(content["messages"], list):
                return PromptManager._chat_messages_to_text(content["messages"])
            return None
        if isinstance(content, list):
            return PromptManager._chat_messages_to_text(content)
        return None

    @staticmethod
    def _chat_messages_to_text(messages: list[Any]) -> str:
        lines: list[str] = []
        for item in messages:
            if not isinstance(item, dict):
                continue
            role = str(item.get("role", "user")).strip().capitalize()
            text = str(item.get("content", "")).strip()
            if text:
                lines.append(f"{role}:\n{text}")
        return "\n\n".join(lines)
