from typing import Optional, Iterable
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import joblib


class BuildupInitiatingPassesCluster:
    def __init__(
        self,
        pitch_x: float = 105.0,
        pitch_y: float = 68.0,
        positions: Iterable[str] = ("GK", "DC", "DL", "DR", "DMC"),
        def_third_x: float = 33.0,
        dx_eps: float = 0,
        default_k: int = 10,
    ):
        self.pitch_x = pitch_x
        self.pitch_y = pitch_y
        self.positions = {p.upper() for p in positions}
        self.def_third_x = def_third_x
        self.dx_eps = dx_eps
        self.default_k = default_k
        self.model: Optional[KMeans] = None

    def _detect_passes(self, df: pd.DataFrame) -> pd.DataFrame:
        is_pass = df["type_name"].str.lower() == "pass"
        in_def_third = df["start_x"] <= self.def_third_x
        dx = df["end_x"] - df["start_x"]
        dx_ok = dx >= -self.dx_eps

        pos = df["former_position"].fillna(df.get("starting_position"))
        pos_ok = pos.astype(str).str.upper().isin(self.positions)

        body_ok = df["bodypart_name"].astype(str).str.lower().str.contains("foot")

        keep = is_pass & in_def_third & dx_ok & pos_ok & body_ok
        passes = df[keep].copy()

        passes["x"] = passes["start_x"]
        passes["y"] = passes["start_y"]
        return passes

    def _normalize(self, df: pd.DataFrame) -> np.ndarray:
        X = df[["x", "y", "end_x", "end_y"]].to_numpy(dtype=float)
        X[:, [0, 2]] /= self.pitch_x
        X[:, [1, 3]] /= self.pitch_y
        return X

    def fit(
        self,
        df: pd.DataFrame,
        n_clusters: Optional[int] = None,
        k_min: int = 2,
        k_max: int = 20,
        show_plot: bool = False,
        random_state: int = 42,
        n_init: int = 50,
        algorithm: str = "elkan",
        ax: Optional[plt.Axes] = None,
    ) -> pd.DataFrame:
        passes = self._detect_passes(df)
        if passes.empty:
            raise ValueError("No buildup initiating passes found.")

        X = self._normalize(passes)

        if n_clusters is None:
            k_max = min(k_max, len(passes) - 1)
            ks = np.arange(k_min, k_max + 1)
            models, inertias = zip(*[
                (
                    (km := KMeans(k, n_init=n_init, random_state=random_state,
                                  algorithm=algorithm)).fit(X).inertia_,
                    km
                ) for k in ks
            ])

            x1, y1, x2, y2 = ks[0], inertias[0], ks[-1], inertias[-1]
            denom = np.hypot(x2 - x1, y2 - y1) or 1
            dists = np.abs((y2 - y1) * ks - (x2 - x1) * inertias +
                           x2 * y1 - y2 * x1) / denom
            best_idx = int(np.argmax(dists))
            self.model = models[best_idx]

            if show_plot:
                if ax is None:
                    _, ax = plt.subplots()
                ax.plot(ks, inertias, marker="o")
                ax.axvline(ks[best_idx], linestyle="--", label=f"k={ks[best_idx]}")
                ax.set(title="Elbow method", xlabel="k", ylabel="Inertia")
                ax.legend()
                plt.tight_layout()
        else:
            self.model = KMeans(
                n_clusters=n_clusters,
                n_init=n_init,
                random_state=random_state,
                algorithm=algorithm,
            ).fit(X)

        passes["buildup_cluster"] = self.model.predict(X)
        return df.merge(
            passes[["game_id", "action_id", "buildup_cluster"]],
            on=["game_id", "action_id"],
            how="left"
        )

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.model is None:
            raise ValueError("Model not fitted.")

        passes = self._detect_passes(df)
        if passes.empty:
            return df.copy().assign(buildup_cluster=np.nan)

        X = self._normalize(passes)
        passes["buildup_cluster"] = self.model.predict(X)

        return df.merge(
            passes[["game_id", "action_id", "buildup_cluster"]],
            on=["game_id", "action_id"],
            how="left"
        )

    def save(self, path: str = "models/BuildupInitiatingPasses.joblib") -> None:
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: str = "models/BuildupInitiatingPasses.joblib"):
        return joblib.load(path)
