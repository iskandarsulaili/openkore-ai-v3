from __future__ import annotations

import logging
import math
from collections import defaultdict
from dataclasses import dataclass, field
from threading import RLock
from typing import Any

from ai_sidecar.contracts.ml_subconscious import ModelFamily
from ai_sidecar.ml_subconscious.labeling_pipeline import LabelingPipeline
from ai_sidecar.ml_subconscious.model_registry import ModelRegistry, vectorize_state_features

logger = logging.getLogger(__name__)

from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.linear_model import SGDClassifier, SGDRegressor
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import NearestNeighbors


def _split_train_eval(rows: list[dict[str, object]], ratio: float = 0.8) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    if not rows:
        return [], []
    cut = int(max(1, min(len(rows), round(len(rows) * ratio))))
    train_rows = rows[:cut]
    eval_rows = rows[cut:] if cut < len(rows) else rows
    return train_rows, eval_rows


def _softmax(values: list[float]) -> list[float]:
    if not values:
        return []
    mx = max(values)
    exp = [math.exp(v - mx) for v in values]
    s = sum(exp)
    if s <= 0:
        return [0.0 for _ in exp]
    return [v / s for v in exp]


def _cosine(lhs: list[float], rhs: list[float]) -> float:
    dot = sum(a * b for a, b in zip(lhs, rhs, strict=False))
    ln = math.sqrt(sum(a * a for a in lhs))
    rn = math.sqrt(sum(a * a for a in rhs))
    if ln <= 0.0 or rn <= 0.0:
        return 0.0
    return max(-1.0, min(1.0, dot / (ln * rn)))


def _label_confidence(probabilities: list[float], index: int) -> float:
    if not probabilities:
        return 0.0
    if index < 0 or index >= len(probabilities):
        return max(probabilities)
    return float(probabilities[index])


def _is_normal(row: dict[str, object]) -> bool:
    label = str(row.get("label") or "").lower()
    target = float(row.get("target") or 0.0)
    return label == "normal" or target <= 0.5


def _family_vector(row: dict[str, object]) -> list[float]:
    return [float(item) for item in list(row.get("vector") or [])]


@dataclass(slots=True)
class TrainingHarness:
    labels: LabelingPipeline
    registry: ModelRegistry
    _lock: RLock = field(default_factory=RLock)
    _metrics: dict[str, dict[str, float]] = field(default_factory=dict)

    def _ensure_rows(self, *, family: ModelFamily, bot_id: str | None, max_samples: int) -> list[dict[str, object]]:
        rows = self.labels.replay_buffer(family=family, bot_id=bot_id, limit=max_samples)
        return rows

    def _train_encounter_classifier(self, rows: list[dict[str, object]], *, incremental: bool) -> dict[str, object]:
        del incremental
        classes = sorted({str(row.get("label") or "balanced") for row in rows})
        class_to_idx = {label: idx for idx, label in enumerate(classes)}
        x_train = [_family_vector(row) for row in rows]
        y_train = [class_to_idx[str(row.get("label") or "balanced")] for row in rows]

        model = SGDClassifier(loss="log_loss", random_state=42, max_iter=500, tol=1e-3)
        model.partial_fit(x_train, y_train, classes=list(range(len(classes))))
        return {
            "type": "encounter_classifier_sgd",
            "model": model,
            "classes": classes,
            "class_to_idx": class_to_idx,
        }

    def _train_loot_ranker(self, rows: list[dict[str, object]], *, incremental: bool) -> dict[str, object]:
        del incremental
        regressor = SGDRegressor(random_state=42, max_iter=600, tol=1e-3, loss="huber")
        x_train = [_family_vector(row) for row in rows]
        y_train = [float(row.get("target") or 0.0) for row in rows]
        regressor.partial_fit(x_train, y_train)

        grouped_vectors: dict[str, list[list[float]]] = defaultdict(list)
        grouped_target: dict[str, list[float]] = defaultdict(list)
        for row in rows:
            label = str(row.get("label") or "unknown")
            grouped_vectors[label].append(_family_vector(row))
            grouped_target[label].append(float(row.get("target") or 0.0))

        centroids: dict[str, list[float]] = {}
        baselines: dict[str, float] = {}
        for label, vectors in grouped_vectors.items():
            width = len(vectors[0]) if vectors else 0
            centroid = [0.0] * width
            for vec in vectors:
                for idx, value in enumerate(vec):
                    centroid[idx] += value
            count = float(len(vectors)) if vectors else 1.0
            centroids[label] = [item / count for item in centroid]
            baselines[label] = float(sum(grouped_target[label]) / max(1.0, float(len(grouped_target[label]))))

        return {
            "type": "loot_ranker_sgd",
            "regressor": regressor,
            "centroids": centroids,
            "baselines": baselines,
        }

    def _train_route_classifier(self, rows: list[dict[str, object]], *, incremental: bool) -> dict[str, object]:
        del incremental
        classes = sorted({str(row.get("label") or "repath") for row in rows})
        class_to_idx = {label: idx for idx, label in enumerate(classes)}
        x_train = [_family_vector(row) for row in rows]
        y_train = [class_to_idx[str(row.get("label") or "repath")] for row in rows]
        model = RandomForestClassifier(n_estimators=64, max_depth=8, random_state=42, n_jobs=1)
        model.fit(x_train, y_train)
        return {
            "type": "route_recovery_rf",
            "model": model,
            "classes": classes,
            "class_to_idx": class_to_idx,
        }

    def _train_npc_dialogue_predictor(self, rows: list[dict[str, object]], *, incremental: bool) -> dict[str, object]:
        del incremental
        classes = sorted({str(row.get("label") or "default") for row in rows})
        class_to_idx = {label: idx for idx, label in enumerate(classes)}
        x_train = [[abs(value) for value in _family_vector(row)] for row in rows]
        y_train = [class_to_idx[str(row.get("label") or "default")] for row in rows]
        model = MultinomialNB(alpha=0.2)
        model.fit(x_train, y_train)
        return {
            "type": "npc_dialogue_multinomial_nb",
            "model": model,
            "classes": classes,
            "class_to_idx": class_to_idx,
        }

    def _train_risk_anomaly_detector(self, rows: list[dict[str, object]], *, incremental: bool) -> dict[str, object]:
        del incremental
        normal_rows = [row for row in rows if _is_normal(row)]
        if not normal_rows:
            normal_rows = rows
        x_train = [_family_vector(row) for row in normal_rows]
        model = IsolationForest(n_estimators=100, contamination="auto", random_state=42)
        model.fit(x_train)
        return {
            "type": "risk_isolation_forest",
            "model": model,
            "normal_support": len(normal_rows),
        }

    def _train_memory_ranker(self, rows: list[dict[str, object]], *, incremental: bool) -> dict[str, object]:
        del incremental
        x_train = [_family_vector(row) for row in rows]
        labels = [str(row.get("label") or "none") for row in rows]
        scores = [float(row.get("target") or 0.0) for row in rows]
        knn = NearestNeighbors(metric="cosine", algorithm="brute", n_neighbors=min(8, max(1, len(x_train))))
        knn.fit(x_train)
        return {
            "type": "memory_knn_ranker",
            "model": knn,
            "vectors": x_train,
            "labels": labels,
            "scores": scores,
        }

    def _train_family(self, *, family: ModelFamily, rows: list[dict[str, object]], incremental: bool) -> dict[str, object]:
        if family == ModelFamily.encounter_classifier:
            return self._train_encounter_classifier(rows, incremental=incremental)
        if family == ModelFamily.loot_ranker:
            return self._train_loot_ranker(rows, incremental=incremental)
        if family == ModelFamily.route_recovery_classifier:
            return self._train_route_classifier(rows, incremental=incremental)
        if family == ModelFamily.npc_dialogue_predictor:
            return self._train_npc_dialogue_predictor(rows, incremental=incremental)
        if family == ModelFamily.risk_anomaly_detector:
            return self._train_risk_anomaly_detector(rows, incremental=incremental)
        return self._train_memory_ranker(rows, incremental=incremental)

    def _predict_with_package(self, *, family: ModelFamily, package: dict[str, object], vector: list[float]) -> tuple[dict[str, object], float]:
        package_type = str(package.get("type") or "")

        if package_type in {"encounter_classifier_sgd", "route_recovery_rf", "npc_dialogue_multinomial_nb"}:
            model = package.get("model")
            classes = [str(item) for item in list(package.get("classes") or [])]
            if model is None or not classes:
                return {"label": "unknown", "scores": []}, 0.0

            proba = model.predict_proba([vector])[0] if hasattr(model, "predict_proba") else []
            pred_idx = int(model.predict([vector])[0])
            pred_label = classes[pred_idx] if 0 <= pred_idx < len(classes) else "unknown"
            scored = []
            for idx, label in enumerate(classes):
                p = float(proba[idx]) if idx < len(proba) else 0.0
                scored.append({"label": label, "score": p})
            confidence = _label_confidence([float(item.get("score") or 0.0) for item in scored], pred_idx)
            return {"label": pred_label, "scores": scored}, confidence

        if package_type == "loot_ranker_sgd":
            regressor = package.get("regressor")
            centroids = dict(package.get("centroids") or {})
            baselines = {str(k): float(v) for k, v in dict(package.get("baselines") or {}).items()}
            if regressor is None or not centroids:
                return {"label": "unknown", "scores": []}, 0.0
            base = float(regressor.predict([vector])[0])
            scores: list[tuple[str, float]] = []
            for label, centroid in centroids.items():
                similarity = _cosine(vector, [float(item) for item in list(centroid)])
                baseline = baselines.get(label, 0.0)
                total = 0.45 * ((similarity + 1.0) / 2.0) + 0.45 * baseline + 0.10 * max(0.0, min(1.0, base))
                scores.append((label, float(total)))
            scores.sort(key=lambda item: item[1], reverse=True)
            probs = _softmax([item[1] for item in scores])
            ranked = [{"label": label, "score": probs[idx]} for idx, (label, _) in enumerate(scores[:8])]
            top = ranked[0]["label"] if ranked else "unknown"
            confidence = float(ranked[0]["score"] if ranked else 0.0)
            return {"label": top, "scores": ranked}, confidence

        if package_type == "risk_isolation_forest":
            model = package.get("model")
            if model is None:
                return {"label": "normal", "anomaly_score": 0.0}, 0.0
            pred = int(model.predict([vector])[0])
            score = float(model.decision_function([vector])[0])
            anomaly_score = max(0.0, min(1.0, 0.5 - score))
            label = "anomaly" if pred == -1 else "normal"
            confidence = max(0.0, min(1.0, abs(score)))
            return {
                "label": label,
                "anomaly_score": anomaly_score,
                "decision_score": score,
            }, confidence

        if package_type == "memory_knn_ranker":
            model = package.get("model")
            vectors = [[float(v) for v in list(row)] for row in list(package.get("vectors") or [])]
            labels = [str(item) for item in list(package.get("labels") or [])]
            targets = [float(item) for item in list(package.get("scores") or [])]
            if model is None or not vectors:
                return {"label": "none", "scores": []}, 0.0

            distances, indices = model.kneighbors([vector], n_neighbors=min(8, len(vectors)), return_distance=True)
            ranked = []
            for dist, idx in zip(list(distances[0]), list(indices[0]), strict=False):
                similarity = max(0.0, min(1.0, 1.0 - float(dist)))
                label = labels[idx] if idx < len(labels) else "none"
                target_score = targets[idx] if idx < len(targets) else 0.0
                score = 0.7 * similarity + 0.3 * target_score
                ranked.append({"label": label, "score": score, "distance": float(dist)})
            ranked.sort(key=lambda item: float(item.get("score") or 0.0), reverse=True)
            top = str(ranked[0]["label"] if ranked else "none")
            confidence = float(ranked[0]["score"] if ranked else 0.0)
            return {"label": top, "scores": ranked[:8]}, confidence

        return {"label": "unknown", "scores": []}, 0.0

    def _evaluate(self, *, family: ModelFamily, package: dict[str, object], eval_rows: list[dict[str, object]]) -> dict[str, float]:
        if not eval_rows:
            return {"accuracy": 0.0, "f1": 0.0, "mae": 0.0, "confidence_mean": 0.0}

        y_true_label: list[str] = []
        y_pred_label: list[str] = []
        y_true_reg: list[float] = []
        y_pred_reg: list[float] = []
        confidences: list[float] = []

        for row in eval_rows:
            vector = _family_vector(row)
            pred, confidence = self._predict_with_package(family=family, package=package, vector=vector)
            confidences.append(float(confidence))

            if family in {ModelFamily.loot_ranker, ModelFamily.memory_retrieval_ranker}:
                y_true_reg.append(float(row.get("target") or 0.0))
                scores = list(pred.get("scores") or [])
                y_pred_reg.append(float(scores[0].get("score") if scores else 0.0))
            elif family == ModelFamily.risk_anomaly_detector:
                y_true_label.append("anomaly" if not _is_normal(row) else "normal")
                y_pred_label.append(str(pred.get("label") or "normal"))
            else:
                y_true_label.append(str(row.get("label") or ""))
                y_pred_label.append(str(pred.get("label") or ""))

        accuracy = 0.0
        f1 = 0.0
        mae = 0.0
        if y_true_label:
            accuracy = float(accuracy_score(y_true_label, y_pred_label))
            try:
                f1 = float(f1_score(y_true_label, y_pred_label, average="weighted"))
            except Exception:
                f1 = 0.0
        if y_true_reg:
            mae = float(mean_absolute_error(y_true_reg, y_pred_reg))

        confidence_mean = float(sum(confidences) / max(1.0, float(len(confidences))))
        return {
            "accuracy": accuracy,
            "f1": f1,
            "mae": mae,
            "confidence_mean": confidence_mean,
        }

    def train(
        self,
        *,
        family: ModelFamily,
        bot_id: str | None,
        incremental: bool,
        max_samples: int,
    ) -> tuple[str, int, dict[str, float], dict[str, object]]:
        rows = self._ensure_rows(family=family, bot_id=bot_id, max_samples=max_samples)
        if len(rows) < 16:
            return "", len(rows), {"accuracy": 0.0, "f1": 0.0, "mae": 0.0, "confidence_mean": 0.0}, self.registry.ab_state(family=family)

        train_rows, eval_rows = _split_train_eval(rows)
        package = self._train_family(family=family, rows=train_rows, incremental=incremental)
        package["family"] = family.value
        package["trained_samples"] = len(train_rows)
        metrics = self._evaluate(family=family, package=package, eval_rows=eval_rows)

        activate = self.registry.active_version(family=family) is None
        version = self.registry.save_model(family=family, package=package, metrics=metrics, activate=activate)

        with self._lock:
            self._metrics[family.value] = dict(metrics)

        logger.info(
            "ml_model_trained",
            extra={
                "event": "ml_model_trained",
                "family": family.value,
                "version": version,
                "samples": len(train_rows),
                "metrics": metrics,
            },
        )

        return version, len(rows), metrics, self.registry.ab_state(family=family)

    def predict(
        self,
        *,
        family: ModelFamily,
        state_features: dict[str, object],
        context: dict[str, object],
        version: str | None = None,
    ) -> tuple[str, dict[str, object], float]:
        del context
        resolved_version = version or self.registry.active_version(family=family) or ""
        package = self.registry.load_model(family=family, version=resolved_version if resolved_version else None)
        if package is None:
            return resolved_version, {"label": "no_model", "scores": []}, 0.0

        vector, _ = vectorize_state_features(state_features, dims=self.registry.vector_dims)
        pred, confidence = self._predict_with_package(family=family, package=package, vector=vector)

        if family == ModelFamily.encounter_classifier:
            recommendation = {"combat_profile": str(pred.get("label") or "balanced"), "scores": list(pred.get("scores") or [])}
        elif family == ModelFamily.loot_ranker:
            recommendation = {"top_loot": str(pred.get("label") or ""), "ranked": list(pred.get("scores") or [])}
        elif family == ModelFamily.route_recovery_classifier:
            recommendation = {"stuck_strategy": str(pred.get("label") or "repath"), "scores": list(pred.get("scores") or [])}
        elif family == ModelFamily.npc_dialogue_predictor:
            recommendation = {"next_branch": str(pred.get("label") or "default"), "scores": list(pred.get("scores") or [])}
        elif family == ModelFamily.risk_anomaly_detector:
            recommendation = {
                "risk_label": str(pred.get("label") or "normal"),
                "anomaly_score": float(pred.get("anomaly_score") or 0.0),
                "decision_score": float(pred.get("decision_score") or 0.0),
            }
        else:
            recommendation = {"top_memory": str(pred.get("label") or "none"), "ranked": list(pred.get("scores") or [])}

        return resolved_version, recommendation, float(max(0.0, min(1.0, confidence)))

    def metrics(self) -> dict[str, dict[str, float]]:
        with self._lock:
            return {key: dict(value) for key, value in self._metrics.items()}

