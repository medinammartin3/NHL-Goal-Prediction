import streamlit as st
import pandas as pd
import os
import base64
import plotly.express as px
import plotly.graph_objects as go
from ift6758.client.serving_client import ServingClient
from ift6758.client.game_client import GameClient

# --- Configuration for Default Model ---
DEFAULT_WORKSPACE = "IFT6758--2025-A03"
DEFAULT_MODEL = "LogReg_Model_with_distance_and_angle"
DEFAULT_VERSION = "v6"
DEFAULT_PROJECT = "IFT6758.2025-A03"

st.title("NHL Goal prediction")

# --- Initialize Clients ---

# Maintain a single ServingClient for the entire session
if "serving_client" not in st.session_state:
    serving_ip = os.environ.get("SERVING_IP", "127.0.0.1")
    serving_port = 5000

    # Remove protocol if present (just to be clean), but the client handles the rest
    serving_ip = serving_ip.replace("https://", "").replace("http://", "").strip("/")

    # We can pass any port (e.g. 80), the client ignores it if "run.app" is in the IP
    st.session_state["serving_client"] = ServingClient(ip=serving_ip, port=80)

serving_client = st.session_state["serving_client"]

# Maintain a single GameClient for the entire session
if "game_client" not in st.session_state:
    st.session_state["game_client"] = GameClient(serving_client=serving_client)
game_client = st.session_state["game_client"]

# Persistent buffer: stores events by game_id to avoid reloading them on every refresh
if "events_buffers" not in st.session_state:
    st.session_state["events_buffers"] = {}

# --- Auto-Load Default Model ---
# This replaces the manual sidebar download
if serving_client.model is None:
    with st.spinner(f"Initializing: Downloading default model ({DEFAULT_MODEL})..."):
        try:
            result = serving_client.download_registry_model(
                entity=DEFAULT_WORKSPACE,
                artifact_name=DEFAULT_MODEL,
                project=DEFAULT_PROJECT,
                version=DEFAULT_VERSION
            )

            # Update features to use if provided
            if 'features' in result and result['features'] is not None:
                serving_client.features = result['features']

            st.success(f"Model loaded successfully")
        except Exception as e:
            st.error(f"Failed to auto-load model: {e}")

# --- Instructions ---
# Updated to remove the "Load Model" step
with st.expander("ℹ️ How to use"):
    st.markdown("""
    1. **Fetch Data:** Enter a **Game ID** (e.g., `2023020001`) and click **Ping Game** (each click loads 5 game events).
    2. **Analyze:** View the **Scoreboard**, **Shot Maps**, and **xG Charts** below.
    """)

st.divider()

# --- Main Application Logic ---

with st.container():
    # Game ID Input
    game_id = st.text_input("Game ID", value="2023020001")

    if 'previous_game_id' not in st.session_state:
        st.session_state['previous_game_id'] = game_id

    # If user changes game_id, reset buffer and teams
    if game_id != st.session_state['previous_game_id']:
        if game_id not in st.session_state["events_buffers"]:
            st.session_state["events_buffers"][game_id] = pd.DataFrame()
        st.session_state["home_team"] = None
        st.session_state["away_team"] = None
        st.session_state["new_events_df"] = pd.DataFrame()
        st.session_state['previous_game_id'] = game_id

    # Disable button if auto-load failed (model is None)
    ping_disabled = (serving_client.model is None)

    if ping_disabled:
        st.error("Model failed to load. Please check your connection or configuration.")

    ping_game = st.button("Get Events", disabled=ping_disabled)

    if ping_game:
        with st.spinner("Fetching game data..."):

            # Process and predict
            new_events_df, home_team, away_team, current_score_home, current_score_away = game_client.process_and_predict(
                game_id)

            # Load teams and new chunk into state
            st.session_state["home_team"] = home_team
            st.session_state["away_team"] = away_team

            if not new_events_df.empty:
                st.session_state["new_events_df"] = new_events_df

                # Update events buffer
                if game_id not in st.session_state["events_buffers"]:
                    st.session_state["events_buffers"][game_id] = pd.DataFrame()
                # Append new events to existing buffer
                st.session_state["events_buffers"][game_id] = pd.concat(
                    [st.session_state["events_buffers"][game_id], new_events_df],
                    ignore_index=True)
            else:
                st.info("No new events — showing last known state.")

# --- Dashboard & Scoreboard ---
with st.container():
    if "events_buffers" in st.session_state and game_id in st.session_state["events_buffers"]:

        buffer = st.session_state["events_buffers"][game_id]

        if buffer.empty:
            st.stop()

        home_team = st.session_state["home_team"]
        away_team = st.session_state["away_team"]

        # Last game state
        last = buffer.iloc[-1]

        goals = buffer[buffer["Event Type"] == "Goal"]
        # Current scores
        score_home = goals[goals["Team"].astype(str).str.strip().str.lower() ==
                           goals["Home Team"].astype(str).str.strip().str.lower()].shape[0]

        score_away = goals[goals["Team"].astype(str).str.strip().str.lower() !=
                           goals["Home Team"].astype(str).str.strip().str.lower()].shape[0]

        # Last period
        period = last["Period"]

        # Time remaining
        seconds_elapsed = last['Game Seconds']
        seconds_remaining = (20 * 60 * period) - seconds_elapsed
        minutes = seconds_remaining // 60
        seconds = seconds_remaining % 60
        clock_display = f"{int(minutes):02}:{int(seconds):02}"

        # Calculate total xG per team
        home_xg = buffer[buffer["Team"].astype(str).str.strip().str.lower()
                         == buffer["Home Team"].astype(str).str.strip().str.lower()]["prediction"].sum()

        away_xg = buffer[buffer["Team"].astype(str).str.strip().str.lower() !=
                         buffer["Home Team"].astype(str).str.strip().str.lower()]["prediction"].sum()

        # Difference between actual score and xG
        diff_home = score_home - home_xg
        diff_away = score_away - away_xg

        # Display dashboard
        st.markdown(f"## Game {game_id}: **{home_team} vs {away_team}**")
        st.write(f"**Period {period} — {clock_display} left**")

        col1, col2 = st.columns(2)
        with col1:
            st.metric(
                label=f"{home_team} xG (actual)",
                value=f"{home_xg:.2f} ({score_home})",
                delta=f"{diff_home:.2f}"
            )
        with col2:
            st.metric(
                label=f"{away_team} xG (actual)",
                value=f"{away_xg:.2f} ({score_away})",
                delta=f"{diff_away:.2f}"
            )

# --- Visualizations ---
with st.container():
    st.subheader("Visualizations")

    if "events_buffers" in st.session_state and game_id in st.session_state["events_buffers"]:
        df = st.session_state["events_buffers"][game_id]

        if not df.empty and "prediction" in df.columns:
            # Team names
            home_team = st.session_state.get("home_team", "Home")
            away_team = st.session_state.get("away_team", "Away")

            tab1, tab2 = st.tabs(["Shot Map", "Expected Goals (xG) Evolution"])

            # --- Shot Map ---
            with tab1:
                st.info(
                    """
                    **Shot Map:** Displays shot locations on the ice, colored by team, with size proportional to goal probability (shot danger).
                    Hovering over a shot displays a list of details about the event.
                    """
                )
                x_col = "X"
                y_col = "Y"

                if x_col in df.columns and y_col in df.columns:

                    df_rounded = df.copy()
                    df_rounded['prediction'] = df['prediction'].round(4)

                    fig_shot = px.scatter(
                        df_rounded,
                        x=x_col,
                        y=y_col,
                        color="Team",
                        size="prediction",  # Size proportional to goal probability
                        size_max=15,
                        hover_data=["Event Type", "Shooter", "prediction", "Period", "Time"],
                        title=f"Shot Map - {away_team} vs {home_team}",
                        color_discrete_sequence=["blue", "red"],
                        opacity=0.8
                    )

                    rink_image_path = "nhl_rink.png"

                    # Encoding
                    try:
                        with open(rink_image_path, "rb") as image_file:
                            encoded_image = base64.b64encode(image_file.read()).decode()

                        # Add background image
                        fig_shot.add_layout_image(
                            dict(
                                source='data:image/png;base64,' + encoded_image,
                                xref="x",
                                yref="y",
                                x=-100,
                                y=42.5,
                                sizex=200,
                                sizey=85,
                                sizing="contain",
                                opacity=1,
                                layer="below"
                            )
                        )
                    except FileNotFoundError:
                        st.warning("Rink image not found. Displaying map without background.")

                    # Fix axes to match rink and remove grid
                    fig_shot.update_layout(
                        xaxis=dict(
                            showgrid=False,
                            zeroline=False,
                            visible=True,
                            range=[-100, 100]
                        ),
                        yaxis=dict(
                            showgrid=False,
                            zeroline=False,
                            visible=True,
                            range=[-42.5, 42.5],
                            scaleanchor="x",
                            scaleratio=1  # Prevent image distortion
                        ),
                        plot_bgcolor='rgba(0,0,0,0)',
                        # Ensure points don't appear outside image area
                        modebar_remove=['zoom', 'pan', 'select', 'lasso', 'autoscale', 'resetview']
                    )

                    st.plotly_chart(fig_shot, use_container_width=True)
                else:
                    st.warning(f"Unable to display shot map.")

            # --- xG Evolution ---
            with tab2:
                st.info(
                    """
                    **Expected Goals (xG) Evolution:** Chart showing the cumulative expected goal probabilities over time.
                    Indicates which team is dominating the match as the game progresses.
                    """
                )
                if "Game Seconds" in df.columns:
                    # Separate data by team
                    df_home = df[df['Team'].str.strip() == df['Home Team'].str.strip()].copy()
                    df_away = df[df['Team'].str.strip() != df['Home Team'].str.strip()].copy()

                    # Calculate cumulative sum of xG
                    df_home['Cumulative xG'] = df_home['prediction'].cumsum()
                    df_away['Cumulative xG'] = df_away['prediction'].cumsum()

                    fig = go.Figure()

                    hover_template = '<b>Team:</b> %{data.name}<br>' + \
                                     '<b>Game Seconds:</b> %{x:.2f}<br>' + \
                                     '<b>Cumulative xG:</b> %{y:.2f}<br>' + \
                                     '<extra></extra>'

                    # Line for Home Team
                    fig.add_trace(go.Scatter(
                        x=df_home['Game Seconds'],
                        y=df_home['Cumulative xG'],
                        mode='lines+markers',
                        name=f"{home_team}",
                        line=dict(width=3, color='blue'),
                        hovertemplate=hover_template
                    ))

                    # Line for Away Team
                    fig.add_trace(go.Scatter(
                        x=df_away['Game Seconds'],
                        y=df_away['Cumulative xG'],
                        mode='lines+markers',
                        name=f"{away_team}",
                        line=dict(width=3, color='red'),
                        hovertemplate=hover_template
                    ))

                    fig.update_layout(
                        title="Expected Goals (xG) Evolution over the game",
                        xaxis_title="Game seconds",
                        yaxis_title="Cumulative xG",
                        showlegend=True,
                    )

                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("Unable to display Expected Goals (xG) evolution over time.")


# --- Data Table ---
with st.container():
    st.subheader("Data used for predictions")
    cols_to_show = ["Distance", "Angle", "Goal", "Empty Net", "prediction"]
    if "events_buffers" in st.session_state and game_id in st.session_state["events_buffers"]:
        buffer = st.session_state["events_buffers"][game_id]

    if "new_events_df" in st.session_state:
        # If new chunk is empty, show last chunk from buffer
        if st.session_state["new_events_df"].empty:
            # Show last chunk from buffer
            if "events_buffers" in st.session_state and game_id in st.session_state["events_buffers"]:
                st.write(buffer)
            else:
                st.write("No data available at the moment.")
        else:
            # Show new chunk
            st.write(buffer)