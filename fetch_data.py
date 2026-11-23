import time
from pathlib import Path

import pandas as pd
from nba_api.stats.endpoints import leaguegamefinder

# Année de début de saison : 2025 correspond à la saison 2025-26
SEASONS = [2021, 2022, 2023, 2024, 2025, 2026]
OUTPUT = Path("data/games.csv")


def fetch_season(start_year: int) -> pd.DataFrame:
    gf = leaguegamefinder.LeagueGameFinder(
        season_nb=start_year,
        season_type_nb=2,                  # saison régulière
        player_or_team_abbreviation="T",   # une ligne par équipe et par match
        timeout=60,
    )
    time.sleep(1)  # politesse envers l'API
    rows = gf.get_data_frames()[0]

    home = rows[rows["MATCHUP"].str.contains("vs.", regex=False)]
    away = rows[rows["MATCHUP"].str.contains("@", regex=False)]

    games = home[["GAME_ID", "GAME_DATE", "TEAM_NAME", "PTS"]].merge(
        away[["GAME_ID", "TEAM_NAME", "PTS"]],
        on="GAME_ID",
        suffixes=("_home", "_away"),
    )
    games = games.rename(columns={
        "GAME_DATE": "date",
        "TEAM_NAME_home": "home_team",
        "TEAM_NAME_away": "away_team",
        "PTS_home": "home_score",
        "PTS_away": "away_score",
    })
    games["season"] = f"{start_year}-{str(start_year + 1)[-2:]}"
    return games[["date", "season", "home_team", "away_team", "home_score", "away_score"]]


def main():
    frames = []
    for year in SEASONS:
        try:
            frame = fetch_season(year)
            print(f"Saison {year}: {len(frame)} matchs")
            frames.append(frame)
        except Exception as exc:  # une saison en erreur ne bloque pas les autres
            print(f"Saison {year} ignorée : {exc}")

    if not frames:
        raise SystemExit("Aucune donnée récupérée.")

    games = pd.concat(frames, ignore_index=True).sort_values("date")
    OUTPUT.parent.mkdir(exist_ok=True)
    games.to_csv(OUTPUT, index=False)
    print(f"{len(games)} matchs enregistrés dans {OUTPUT}")


if __name__ == "__main__":
    main()
