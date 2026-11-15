"""Per-frame emission models: P(state | features), turned into a scaled likelihood for the decoder.

`FrameClassifier` works on a chosen feature set and only speaks on frames where its `required` column equals 1
(for the ball-player features that is `ball_ok`); elsewhere it contributes a zero log-ratio.
`RegimeEmission` combines two classifiers: one for frames where the ball was seen (ball-player and team-shape
features) and one for frames without a ball (team-shape features only, scaled by `beta`).
"""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from possession.features import team_shape

BALL_FEATURES = ("d_H", "d_A", "d_diff", "n_H_5", "n_A_5", "n_H_10", "n_A_10", "ball_speed", "line_dist")
# Team-shape features that carry over between camera views: how fast the players move and how many stand still.
SHAPE_FEATURES = ("sp_H", "sp_A", "still05", "still15")
# Position-based shape features. In the broadcast-derived training data the visible players cluster around the camera,
# i.e. around the ball, so their mean position acts as a proxy for the ball position; in a full-view tracking that
# relation does not exist and a model using these columns merges possessions (found by testing on a full-view tracking set).
VIEW_DEPENDENT_SHAPE = tuple(c for c in team_shape.COLUMNS if c not in ("sh_ok",) + SHAPE_FEATURES)
STATES = ("H", "A", "D")


class FrameClassifier:
    def __init__(self, features=BALL_FEATURES, required="ball_ok", subsample=5, C=1.0):
        self.features, self.required, self.subsample, self.C = tuple(features), required, subsample, C

    def _matrix(self, df):
        X = df[list(self.features)].to_numpy(dtype=float)
        return np.where(np.isnan(X), self.medians, X)

    def fit(self, tables, labels, weights):
        """`tables`, `labels`, `weights`: one entry per training match (25 Hz arrays)."""
        parts, y, w = [], [], []
        for df, lab, wt in zip(tables, labels, weights):
            keep = (df[self.required].to_numpy() == 1) & np.isin(lab, STATES) & (wt > 0)
            idx = np.flatnonzero(keep)[::self.subsample]
            parts.append(df.iloc[idx][list(self.features)].to_numpy(dtype=float)); y.append(lab[idx]); w.append(wt[idx])
        X, y, w = np.vstack(parts), np.concatenate(y), np.concatenate(w)
        self.medians = np.nanmedian(X, axis=0)
        X = np.where(np.isnan(X), self.medians, X)
        self.model = make_pipeline(StandardScaler(), LogisticRegression(C=self.C, max_iter=500))
        self.model.fit(X, y, logisticregression__sample_weight=w)
        self._order = [list(self.model.classes_).index(s) for s in STATES]
        prior = np.array([w[y == s].sum() for s in STATES])
        self.log_prior = np.log(prior / prior.sum())
        return self

    def log_ratio(self, df):
        """(n, 3) log P(state | x) - log P(state); zeros where the required column is not 1."""
        out = np.zeros((len(df), 3))
        ok = df[self.required].to_numpy() == 1
        if ok.any():
            proba = self.model.predict_proba(self._matrix(df[ok]))[:, self._order]
            out[ok] = np.log(np.clip(proba, 1e-6, None)) - self.log_prior
        return out


class RegimeEmission:
    """Ball present: ball-player + team-shape classifier. Ball missing: team-shape classifier, scaled by `beta`."""

    def __init__(self, ball_clf, free_clf, beta=1.0):
        self.ball_clf, self.free_clf, self.beta = ball_clf, free_clf, beta

    @classmethod
    def fit(cls, tables, labels, weights, beta=1.0):
        ball = FrameClassifier(BALL_FEATURES + SHAPE_FEATURES, "ball_ok").fit(tables, labels, weights)
        free = FrameClassifier(SHAPE_FEATURES, "sh_ok").fit(tables, labels, weights)
        return cls(ball, free, beta)

    def log_ratio(self, df):
        out = self.ball_clf.log_ratio(df)
        missing = df["ball_ok"].to_numpy() != 1
        if missing.any():
            out[missing] = self.beta * self.free_clf.log_ratio(df)[missing]
        return out


class BinaryClassifier:
    """P(positive | features) on the frames where `required` equals 1; used for the dead / live and the home / away splits."""

    def __init__(self, features, required, positive, subset, subsample=5, C=1.0):
        self.features, self.required, self.positive, self.subset = tuple(features), required, tuple(positive), tuple(subset)
        self.subsample, self.C = subsample, C

    def fit(self, tables, labels, weights):
        parts, y, w = [], [], []
        for df, lab, wt in zip(tables, labels, weights):
            keep = (df[self.required].to_numpy() == 1) & np.isin(lab, self.subset) & (wt > 0)
            idx = np.flatnonzero(keep)[::self.subsample]
            parts.append(df.iloc[idx][list(self.features)].to_numpy(dtype=float))
            y.append(np.isin(lab[idx], self.positive)); w.append(wt[idx])
        X, y, w = np.vstack(parts), np.concatenate(y), np.concatenate(w)
        self.medians = np.nanmedian(X, axis=0)
        X = np.where(np.isnan(X), self.medians, X)
        self.model = make_pipeline(StandardScaler(), LogisticRegression(C=self.C, max_iter=500))
        self.model.fit(X, y, logisticregression__sample_weight=w)
        self.base_rate = float(w[y].sum() / w.sum())
        return self

    def proba(self, df):
        """Probability of the positive class; 0.5 where the required column is not 1."""
        out = np.full(len(df), 0.5)
        ok = df[self.required].to_numpy() == 1
        if ok.any():
            X = df.loc[ok, list(self.features)].to_numpy(dtype=float)
            out[ok] = self.model.predict_proba(np.where(np.isnan(X), self.medians, X))[:, 1]
        return np.clip(out, 1e-4, 1 - 1e-4)


class FactoredEmission:
    """Emission built from a dead / live classifier and a home / away classifier.

    P(D | x) comes from the dead / live model, P(H | x) = (1 - P(D | x)) * P(home | live, x), likewise for A. The home / away
    model is trained on possession labels from one source, the dead / live model may be trained on others. Ball regime: both
    classifiers use the ball-player and speed features; without a ball only the speed features, scaled by `beta`.
    """

    def __init__(self, dead_ball, dead_free, ha_ball, ha_free, dead_prior, beta=1.0):
        self.dead_ball, self.dead_free, self.ha_ball, self.ha_free = dead_ball, dead_free, ha_ball, ha_free
        self.log_prior = np.log(np.array([(1 - dead_prior) / 2, (1 - dead_prior) / 2, dead_prior]))
        self.beta = beta

    @classmethod
    def fit(cls, dead_tables, dead_labels, dead_weights, ha_tables, ha_labels, ha_weights, beta=1.0,
            ball_extra=(), shape_extra=()):
        """`ball_extra` / `shape_extra`: additional columns (e.g. temporal context) for the ball regime / for both regimes."""
        feats = BALL_FEATURES + SHAPE_FEATURES + tuple(ball_extra) + tuple(shape_extra)
        free = SHAPE_FEATURES + tuple(shape_extra)
        dead_ball = BinaryClassifier(feats, "ball_ok", ("D",), STATES).fit(dead_tables, dead_labels, dead_weights)
        dead_free = BinaryClassifier(free, "sh_ok", ("D",), STATES).fit(dead_tables, dead_labels, dead_weights)
        ha_ball = BinaryClassifier(feats, "ball_ok", ("H",), ("H", "A")).fit(ha_tables, ha_labels, ha_weights)
        ha_free = BinaryClassifier(free, "sh_ok", ("H",), ("H", "A")).fit(ha_tables, ha_labels, ha_weights)
        return cls(dead_ball, dead_free, ha_ball, ha_free, dead_ball.base_rate, beta)

    def _ratio(self, dead, ha, df):
        pd_ = dead.proba(df)
        ph = ha.proba(df)
        logp = np.c_[np.log(1 - pd_) + np.log(ph), np.log(1 - pd_) + np.log(1 - ph), np.log(pd_)]
        return logp - self.log_prior

    def log_ratio(self, df):
        out = self._ratio(self.dead_ball, self.ha_ball, df)
        ball = df["ball_ok"].to_numpy() == 1
        free = self.beta * self._ratio(self.dead_free, self.ha_free, df)
        has_shape = df["sh_ok"].to_numpy() == 1
        out[~ball] = np.where(has_shape[~ball, None], free[~ball], 0.0)
        return out
