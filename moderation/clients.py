"""HTTP clients for external moderation providers."""

from __future__ import annotations

import httpx


class HiveTextModerationClient:
    ENDPOINT = "https://api.thehive.ai/api/v2/task/sync"

    def __init__(self, api_key: str, timeout_s: float = 10.0):
        self._api_key = api_key
        self._client = httpx.AsyncClient(
            timeout=timeout_s,
            headers={"Authorization": f"Token {self._api_key}"}
        )

    async def moderate_text(self, text: str) -> dict:
        if not text or not text.strip():
            return {}
        response = await self._client.post(self.ENDPOINT, data={"text_data": text})
        response.raise_for_status()
        return response.json()

    @staticmethod
    def extract_class_scores(raw_response: dict) -> dict[str, float]:
        try:
            statuses = raw_response.get("status", [])
            output = statuses[0]["response"]["output"][0]
            classes = output.get("classes", [])
            return {c["class"]: float(c["score"]) for c in classes}
        except (KeyError, IndexError, TypeError, ValueError):
            return {}

    async def close(self) -> None:
        await self._client.aclose()


class SightengineImageModerationClient:
    ENDPOINT = "https://api.sightengine.com/1.0/check.json"

    def __init__(
        self,
        api_user: str,
        api_secret: str,
        models: tuple[str, ...] = ("gore-2.0", "offensive", "weapon"),
        timeout_s: float = 10.0,
    ):
        self._api_user = api_user
        self._api_secret = api_secret
        self._models = ",".join(models)
        self._client = httpx.AsyncClient(timeout=timeout_s)

    async def moderate_image_url(self, image_url: str) -> dict:
        params = {
            "url": image_url,
            "models": self._models,
            "api_user": self._api_user,
            "api_secret": self._api_secret,
        }
        response = await self._client.get(self.ENDPOINT, params=params)
        response.raise_for_status()
        return response.json()

    async def close(self) -> None:
        await self._client.aclose()
