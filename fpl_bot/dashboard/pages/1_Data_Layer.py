import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import streamlit as st

from fpl_bot.dashboard.utils import go_grouped_bar, go_grouped_box, load_players_with_names, load_table, page_header

st.set_page_config(page_title="Data Layer", page_icon="🗄️", layout="wide")
page_header("Data Layer", "Everything every model is built on -- the raw ingested data.")

tab_players, tab_teams, tab_fixtures, tab_hist = st.tabs(["Players", "Teams", "Fixtures", "Historical Training Data"])

with tab_players:
    df = load_players_with_names()
    c1, c2 = st.columns([1, 3])
    with c1:
        position_filter = st.multiselect("Position", sorted(df["position_name"].unique()), default=[])
        team_filter = st.multiselect("Team", sorted(df["team_name"].unique()), default=[])
        search = st.text_input("Search name")
    filtered = df
    if position_filter:
        filtered = filtered[filtered["position_name"].isin(position_filter)]
    if team_filter:
        filtered = filtered[filtered["team_name"].isin(team_filter)]
    if search:
        filtered = filtered[filtered["web_name"].str.contains(search, case=False, na=False)]

    with c2:
        st.plotly_chart(
            go_grouped_box(df, "position_name", "total_points", categories=["GKP", "DEF", "MID", "FWD"],
                           title="Last season's total points by position (whole pool)"),
            use_container_width=True,
        )

    st.caption(f"{len(filtered)} of {len(df)} players")
    st.dataframe(
        filtered[["web_name", "team_name", "position_name", "now_cost", "total_points", "points_per_game",
                  "selected_by_percent", "status", "news"]]
        .rename(columns={"web_name": "Name", "team_name": "Team", "position_name": "Pos", "now_cost": "Price (0.1m)",
                          "total_points": "Pts (last szn)", "points_per_game": "PPG", "selected_by_percent": "Selected %"})
        .sort_values("Pts (last szn)", ascending=False),
        use_container_width=True, height=500,
    )

with tab_teams:
    teams_df = load_table("teams").sort_values("strength_attack_home", ascending=False)
    st.dataframe(teams_df, use_container_width=True, height=400)
    st.plotly_chart(
        go_grouped_bar(
            teams_df["short_name"].tolist(),
            {"Home attack": teams_df["strength_attack_home"].tolist(), "Home defence": teams_df["strength_defence_home"].tolist()},
            title="Home attack/defence strength rating by team",
        ),
        use_container_width=True,
    )

with tab_fixtures:
    fixtures_df = load_table("fixtures")
    teams_df = load_table("teams")[["id", "short_name"]].set_index("id")["short_name"]
    fixtures_df["home"] = fixtures_df["team_h"].map(teams_df)
    fixtures_df["away"] = fixtures_df["team_a"].map(teams_df)
    gw_filter = st.slider("Gameweek", int(fixtures_df["event"].min()), int(fixtures_df["event"].max()),
                           int(fixtures_df["event"].min()))
    st.dataframe(
        fixtures_df[fixtures_df["event"] == gw_filter][
            ["home", "away", "team_h_difficulty", "team_a_difficulty", "kickoff_time"]
        ],
        use_container_width=True,
    )

with tab_hist:
    hist_df = load_table("historical_gw")
    by_season = hist_df.groupby("season").agg(rows=("element_id", "count"), players=("element_id", "nunique")).reset_index()
    fig = go_grouped_bar(by_season["season"].tolist(), {"Player-gameweek rows": by_season["rows"].tolist()},
                          title="Historical training rows per season")
    fig.update_traces(text=by_season["players"].tolist(), texttemplate="%{text} players")
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Bar labels show distinct players per season. This is Model 1 and Model 2's entire training set.")
    st.markdown(
        """
**Data quality note:** players are tracked by the dataset's own numeric `element_id`, not name --
names collide often enough (two different players sharing a name, or literal duplicate source rows,
~1,070 found originally) to corrupt training if used as the identity key. Genuine double-gameweeks
(same player, same round, two fixtures) are summed into one row per round, matching how FPL actually
scores them, rather than being left as duplicate rows.
"""
    )
    st.dataframe(hist_df.sample(min(200, len(hist_df))).sort_values(["season", "round"]), use_container_width=True, height=300)
