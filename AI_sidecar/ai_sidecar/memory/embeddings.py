from __future__ import annotations

import hashlib
import logging
import math
import re
from collections import Counter
from typing import Callable, Protocol

logger = logging.getLogger(__name__)

_TOKEN_RE = re.compile(r"[a-z0-9_]{2,}")


class SemanticEmbedder(Protocol):
    @property
    def dimensions(self) -> int: ...

    def tokenize(self, text: str) -> list[str]: ...

    def lexical_signature(self, text: str) -> str: ...

    def embed(self, text: str) -> tuple[list[float], float, str]: ...

    def cosine(self, lhs: list[float], rhs: list[float], *, lhs_norm: float | None = None, rhs_norm: float | None = None) -> float: ...


class LocalSemanticEmbedder:
    """
    Deterministic local embedding generator.

    This intentionally avoids external model downloads while still providing
    functional semantic-ish retrieval via lexical hashing vectors.
    """

    def __init__(self, dimensions: int) -> None:
        self._dimensions = dimensions

    @property
    def dimensions(self) -> int:
        return self._dimensions

    def tokenize(self, text: str) -> list[str]:
        return [token.lower() for token in _TOKEN_RE.findall(text.lower())]

    def lexical_signature(self, text: str) -> str:
        tokens = sorted(set(self.tokenize(text)))
        if not tokens:
            return " "
        return " " + " ".join(tokens) + " "

    def embed(self, text: str) -> tuple[list[float], float, str]:
        tokens = self.tokenize(text)
        counter = Counter(tokens)
        vector = [0.0] * self._dimensions
        if not counter:
            return vector, 0.0, " "

        for token, freq in counter.items():
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            idx = int.from_bytes(digest[:4], "little") % self._dimensions
            sign = 1.0 if (digest[4] % 2 == 0) else -1.0
            magnitude = 1.0 + math.log1p(freq)
            vector[idx] += sign * magnitude

            second_idx = int.from_bytes(digest[5:9], "little") % self._dimensions
            if second_idx != idx:
                vector[second_idx] += sign * 0.5 * magnitude

        norm = math.sqrt(sum(component * component for component in vector))
        return vector, norm, self.lexical_signature(text)

    def cosine(self, lhs: list[float], rhs: list[float], *, lhs_norm: float | None = None, rhs_norm: float | None = None) -> float:
        if len(lhs) != len(rhs):
            return 0.0
        dot = 0.0
        for a, b in zip(lhs, rhs, strict=False):
            dot += a * b
        if lhs_norm is not None and rhs_norm is not None and lhs_norm > 0 and rhs_norm > 0:
            return max(-1.0, min(1.0, dot / (lhs_norm * rhs_norm)))
        return max(-1.0, min(1.0, dot))


class ProviderSemanticEmbedder:
    """Embedding adapter backed by provider API with local deterministic fallback."""

    def __init__(
        self,
        *,
        dimensions: int,
        embed_texts: Callable[[list[str]], list[list[float]]],
        fallback: LocalSemanticEmbedder,
    ) -> None:
        self._dimensions = dimensions
        self._embed_texts = embed_texts
        self._fallback = fallback

    @property
    def dimensions(self) -> int:
        return self._dimensions

    def tokenize(self, text: str) -> list[str]:
        return self._fallback.tokenize(text)

    def lexical_signature(self, text: str) -> str:
        return self._fallback.lexical_signature(text)

    def embed(self, text: str) -> tuple[list[float], float, str]:
        lexical = self.lexical_signature(text)
        try:
            vectors = self._embed_texts([text])
            if vectors and isinstance(vectors[0], list):
                vector = [float(value) for value in vectors[0]]
                if vector:
                    self._dimensions = len(vector)
                    norm = math.sqrt(sum(component * component for component in vector))
                    return vector, norm, lexical
        except Exception:
            logger.exception("provider_embedding_failed")
        return self._fallback.embed(text)

    def cosine(self, lhs: list[float], rhs: list[float], *, lhs_norm: float | None = None, rhs_norm: float | None = None) -> float:
        return self._fallback.cosine(lhs, rhs, lhs_norm=lhs_norm, rhs_norm=rhs_norm)
