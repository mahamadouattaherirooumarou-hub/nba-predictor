import html
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss
from sklearn.calibration import calibration_curve

K = 20           # vitesse d'adaptation Elo
HOME_ADV = 65    # avantage du terrain en points Elo
BASE = 1500      # note initiale
WINDOW = 10      # matchs pris en compte pour la forme récente
FEATURES = ["elo_diff", "form_diff"]

HISTORY = Path("data/games.csv")
UPCOMING = Path("data/upcoming.csv")
PREDICTIONS = Path("predictions.csv")
PAGE = Path("docs/index.html")


def elo_expected(r_a, r_b):
    return 1 / (1 + 10 ** ((r_b - r_a) / 400))


class TeamState:
    """Mémoire des notes Elo et des écarts de score récents."""

    def __init__(self):
        self.elo = {}
        self.margins = {}

    def rating(self, team):
        return self.elo.get(team, BASE)

    def form(self, team):
        recent = self.margins.get(team, [])[-WINDOW:]
        return float(np.mean(recent)) if recent else 0.0

    def features(self, home, away):
        rh, ra = self.rating(home), self.rating(away)
        return {
            "elo_diff": rh - ra,
            "form_diff": self.form(home) - self.form(away),
            "p_elo": elo_expected(rh + HOME_ADV, ra),
        }

    def update(self, home, away, home_score, away_score):
        rh, ra = self.rating(home), self.rating(away)
        p_home = elo_expected(rh + HOME_ADV, ra)
        home_won = 1.0 if home_score > away_score else 0.0
        self.elo[home] = rh + K * (home_won - p_home)
        self.elo[away] = ra + K * ((1 - home_won) - (1 - p_home))
        margin = home_score - away_score
        self.margins.setdefault(home, []).append(margin)
        self.margins.setdefault(away, []).append(-margin)


def build_history(games: pd.DataFrame):
    """Enregistre les variables de chaque match AVANT de mettre à jour les notes."""
    state = TeamState()
    rows = []
    for g in games.itertuples(index=False):
        f = state.features(g.home_team, g.away_team)
        rows.append({
            "season": g.season,
            **f,
            "home_win": int(g.home_score > g.away_score),
        })
        state.update(g.home_team, g.away_team, g.home_score, g.away_score)
    return pd.DataFrame(rows), state


def evaluate(feat: pd.DataFrame):
    """Entraîne sur les saisons anciennes, teste sur la dernière saison."""
    seasons = sorted(feat["season"].unique())
    if len(seasons) < 2:
        return None
    test_season = seasons[-1]
    train = feat[feat["season"] < test_season]
    test = feat[feat["season"] == test_season]
    y = test["home_win"]

    model = LogisticRegression().fit(train[FEATURES], train["home_win"])
    p_model = model.predict_proba(test[FEATURES])[:, 1]

    results = {}
    for name, p in [("Elo seul", test["p_elo"].values),
                    ("Régression logistique", p_model)]:
        results[name] = {
            "accuracy": accuracy_score(y, p > 0.5),
            "brier": brier_score_loss(y, p),
            "logloss": log_loss(y, p),
        }

    frac_pos, mean_pred = calibration_curve(y, p_model, n_bins=10)
    calibration = list(zip(mean_pred, frac_pos))
    return test_season, results, calibration


def predict_upcoming(state: TeamState, model, upcoming: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for g in upcoming.itertuples(index=False):
        f = state.features(g.home_team, g.away_team)
        X = pd.DataFrame([[f["elo_diff"], f["form_diff"]]], columns=FEATURES)
        rows.append({
            "date": pd.Timestamp(g.date).strftime("%Y-%m-%d"),
            "home_team": g.home_team,
            "away_team": g.away_team,
            "p_home_model": round(float(model.predict_proba(X)[0, 1]), 3),
            "p_home_elo": round(f["p_elo"], 3),
        })
    return pd.DataFrame(rows, columns=["date", "home_team", "away_team",
                                       "p_home_model", "p_home_elo"])


def write_page(preds: pd.DataFrame, evaluation) -> None:
    PAGE.parent.mkdir(exist_ok=True)

    if preds.empty:
        pred_rows = '<tr><td colspan="5">Aucun match à venir.</td></tr>'
    else:
        pred_rows = "\n".join(
            f"<tr><td>{html.escape(r.date)}</td>"
            f"<td>{html.escape(r.home_team)}</td>"
            f"<td>{html.escape(r.away_team)}</td>"
            f"<td>{r.p_home_model:.0%}</td>"
            f"<td>{r.p_home_elo:.0%}</td></tr>"
            for r in preds.itertuples(index=False)
        )

    if evaluation is None:
        metric_rows = '<tr><td colspan="4">Pas assez de saisons pour évaluer.</td></tr>'
        season_label = "—"
        calib_rows = ""
    else:
        season_label, results, calibration = evaluation
        metric_rows = "\n".join(
            f"<tr><td>{html.escape(name)}</td><td>{m['accuracy']:.3f}</td>"
            f"<td>{m['brier']:.4f}</td><td>{m['logloss']:.4f}</td></tr>"
            for name, m in results.items()
        )
        calib_rows = "\n".join(
            f"<tr><td>{pred:.0%}</td><td>{real:.0%}</td></tr>"
            for pred, real in calibration
        )

    page = f"""<!doctype html>
<html lang="fr">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Prédictions NBA</title>
  <style>
    body {{ font-family: system-ui, sans-serif; margin: 2rem auto; max-width: 900px; padding: 0 1rem; color: #222; }}
    table {{ border-collapse: collapse; width: 100%; margin-bottom: 2rem; }}
    th, td {{ border-bottom: 1px solid #ddd; padding: 0.5rem; text-align: left; }}
    th {{ background: #f4f4f4; }}
    small {{ color: #666; }}
  </style>
</head>
<body>
  <h1>Prédictions NBA</h1>
  <p><small>Probabilités que l'équipe à domicile gagne. Mise à jour automatique.</small></p>

  <h2>Matchs à venir</h2>
  <table>
    <tr><th>Date</th><th>Domicile</th><th>Extérieur</th><th>Modèle</th><th>Elo seul</th></tr>
    {pred_rows}
  </table>

  <h2>Qualité du modèle (saison test : {html.escape(str(season_label))})</h2>
  <table>
    <tr><th>Méthode</th><th>Précision</th><th>Brier</th><th>Log loss</th></tr>
    {metric_rows}
  </table>

  <h2>Calibration</h2>
  <p><small>Un modèle bien calibré a une probabilité prédite proche de la fréquence observée.</small></p>
  <table>
    <tr><th>Probabilité prédite</th><th>Fréquence observée</th></tr>
    {calib_rows}
  </table>
</body>
</html>
"""
    PAGE.write_text(page, encoding="utf-8")


def main():
    games = (pd.read_csv(HISTORY, parse_dates=["date"])
             .sort_values("date")
             .reset_index(drop=True))

    feat, state = build_history(games)
    evaluation = evaluate(feat)

    model = LogisticRegression().fit(feat[FEATURES], feat["home_win"])

    if UPCOMING.exists():
        upcoming = (pd.read_csv(UPCOMING, parse_dates=["date"])
                    .sort_values("date")
                    .reset_index(drop=True))
        preds = predict_upcoming(state, model, upcoming)
    else:
        preds = pd.DataFrame(columns=["date", "home_team", "away_team",
                                      "p_home_model", "p_home_elo"])

    preds.to_csv(PREDICTIONS, index=False)
    write_page(preds, evaluation)

    print(f"{len(games)} matchs analysés, {len(preds)} prédictions générées.")
    if evaluation:
        season, results, _ = evaluation
        print(f"Évaluation sur la saison {season} :")
        for name, m in results.items():
            print(f"  {name:25s} précision={m['accuracy']:.3f} "
                  f"Brier={m['brier']:.4f} logloss={m['logloss']:.4f}")


if __name__ == "__main__":
    main()
