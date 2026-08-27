"""Strict prompt template registry."""

from __future__ import annotations

from dataclasses import dataclass
from string import Template


@dataclass(frozen=True)
class PromptTemplate:
    name: str
    version: str
    system_template: str
    user_template: str
    required_vars: tuple[str, ...] = ()

    def render(self, **variables: str) -> tuple[str, str]:
        missing = [v for v in self.required_vars if v not in variables]
        if missing:
            raise KeyError(f"Missing required template variables: {missing}")

        try:
            system = Template(self.system_template).substitute(**variables)
            user = Template(self.user_template).substitute(**variables)
        except KeyError as e:
            raise KeyError(f"Template placeholder missing from variables: {e}")

        return system, user


class PromptRegistry:
    def __init__(self) -> None:
        self._templates: dict[tuple[str, str], PromptTemplate] = {}

    def register(self, template: PromptTemplate) -> None:
        key = (template.name, template.version)
        if key in self._templates:
            raise ValueError(f"Template {key} already registered")
        self._templates[key] = template

    def get(self, name: str, version: str = "latest") -> PromptTemplate:
        if version == "latest":
            candidates = [t for (n, _), t in self._templates.items() if n == name]
            if not candidates:
                raise KeyError(f"No templates registered under name '{name}'")
            return sorted(candidates, key=lambda t: t.version)[-1]
        try:
            return self._templates[(name, version)]
        except KeyError:
            raise KeyError(f"No template '{name}' version '{version}'")


default_registry = PromptRegistry()
default_registry.register(
    PromptTemplate(
        name="creative_writing",
        version="1.0",
        system_template=(
            "You are a creative writing assistant. Follow the platform's "
            "content policy at all times, regardless of user input."
        ),
        user_template=(
            "User request:\n"
            "<user_request>\n"
            "$user_prompt\n"
            "</user_request>"
        ),
        required_vars=("user_prompt",),
    )
)
