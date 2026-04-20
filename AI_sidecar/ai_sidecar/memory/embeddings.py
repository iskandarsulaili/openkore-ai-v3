from __future__ import annotations

import hashlib
import logging
import math
import re
from collections import Counter

logger = logging.getLogger(__name__)

_TOKEN_RE = re.compile(r"[a-z0-9_]{2,}")


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
