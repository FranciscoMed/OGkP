import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from mplsoccer.pitch import VerticalPitch
from statsbombpy import sb

# ----------------- New Data Retrieval Functions -----------------
@st.cache_data
def get_competition_data():
    """
    Fetches competition data from the StatsBomb API.
    Returns a DataFrame with competition details.
    """
    comps = sb.competitions()
    return comps

@st.cache_data
def get_team_data(competition_id, season_id):
    """
    Fetches match data for a specific competition and season,
    then extracts unique team names from both home and away teams.
    """
    matches = sb.matches(competition_id=competition_id, season_id=season_id)
    teams = list(pd.unique(matches['home_team'].tolist() + matches['away_team'].tolist()))
    return teams

@st.cache_data
def get_team_gk(competition_id, season_id, team):
    """
    Fetches available goalkeepers from each lineup for the given team
    in a specific competition and season.

    Based on the StatsBomb Open Lineups Specification v2.0.0,
    each lineup JSON contains "team_id", "team_name", and "lineup" (an array of player objects).

    If a player object includes a "position" field, we check if that position is "Goalkeeper".
    If not, we fall back on the common convention that the goalkeeper wears jersey number 1.

    Returns a list of unique goalkeeper names.
    """
    matches = sb.matches(competition_id=competition_id, season_id=season_id)
    team_matches = matches[(matches['home_team'] == team) | (matches['away_team'] == team)]
    gk_set = set()
    for fixture in team_matches['match_id']:
        try:
            lineups = sb.lineups(match_id=fixture)
        except Exception as e:
            continue  # Skip fixtures without lineup data
        # lineups is a dictionary with keys like "home" and "away"
        if lineups[f'{team}'] is not None:
            for idx,player in lineups[f'{team}'].iterrows():
                if player['positions']:   
                    if player['positions'][0]['position'] == 'Goalkeeper':
                        gk_set.add(player['player_name'])
    return list(gk_set)


# ----------------- Optimal GK Position Model Functions -----------------
B = np.array([120, 44])  # Left goalpost
C = np.array([120, 36])  # Right goalpost

def distance_to_GK(GK, shot):
    return np.linalg.norm(np.array(shot) - np.array(GK))

def calculate_optimal_gk_position(A):
    """
    Calculates the optimal goalkeeper position using the incenter of the triangle
    formed by the shooter (A) and the goalposts (B and C).
    """
    AB = np.linalg.norm(B - A)
    AC = np.linalg.norm(C - A)
    BC = np.linalg.norm(C - B)
    P = AB + AC + BC
    incenter = (AB * C + AC * B + BC * A) / P
    return incenter

def calculate_dive_radius(A, GK, vball=30, GKSpeed=1.1, GKReaction=0.2, d_min=0.5, d_max=35, r_min=1.5, r_max=3):
    distance = distance_to_GK(GK, A)
    if distance <= d_min:
        return r_min
    elif distance >= d_max:
        return r_max
    else:
        return r_min + (r_max - r_min) * GKSpeed * ((distance / vball) - GKReaction)

# ----------------- Plotting Function -----------------
def plot_goalkeeper_position(A, optimal_gk_position, left_post=B, right_post=C, similarity=None, actual_position=None):
    """
    Returns a matplotlib figure showing:
      - The shot cone from the shooter (A) to the goalposts
      - The optimal goalkeeper position (red)
      - The actual goalkeeper position (yellow, if provided)
      - Their respective dive circles
      - The calculated Goalkeeper Positional Efficiency (GPE) score
    """
    pitch = VerticalPitch(pitch_type='statsbomb', half=True, line_color='black')
    fig, ax = pitch.draw()
    
    # Plot shot cone
    pitch.lines(A[0], A[1], left_post[0], left_post[1], color='blue', lw=1, linestyle='-', ax=ax)
    pitch.lines(A[0], A[1], right_post[0], right_post[1], color='blue', lw=1, linestyle='-', ax=ax)
    pitch.lines(optimal_gk_position[0], optimal_gk_position[1], A[0], A[1],
                color='green', lw=1, linestyle='--', ax=ax, label='Bisector')
    
    # Plot optimal GK position
    pitch.scatter(optimal_gk_position[0], optimal_gk_position[1],
                  color='red', edgecolor='red', s=80, ax=ax, label='Optimal GK')
    
    if actual_position is not None:
        # Plot actual GK position
        pitch.scatter(actual_position[0], actual_position[1],
                      color='yellow', edgecolor='yellow', s=80, ax=ax, label='Actual GK')
    
    # Plot dive circles
    dive_radius_opt = calculate_dive_radius(A, optimal_gk_position)
    dive_circle_opt = Ellipse((optimal_gk_position[0], optimal_gk_position[1]),
                              width=2*dive_radius_opt, height=2*dive_radius_opt,
                              edgecolor='r', facecolor='none', linestyle='--', linewidth=2)
    ax.add_patch(dive_circle_opt)
    
    if actual_position is not None:
        dive_radius_act = calculate_dive_radius(A, actual_position)
        dive_circle_act = Ellipse((actual_position[0], actual_position[1]),
                                  width=2*dive_radius_act, height=2*dive_radius_act,
                                  edgecolor='y', facecolor='none', linestyle='--', linewidth=2)
        ax.add_patch(dive_circle_act)
    
    # Plot shooter position
    pitch.scatter(A[0], A[1], color='blue', edgecolor='blue', s=100, ax=ax, label='Shooter')
    
    # Add GPE text if provided
    if similarity is not None:
        ax.text(20, 90, f"OGkP: {similarity:.2f}", fontsize=10, color='black', ha='center', va='center')

    
    ax.set_title('Optimal Goalkeeper Positioning (OGkP)', fontsize=12)
    ax.legend(loc='upper left', fontsize=8, frameon=True)
    return fig

# ----------------- Similarity Metric Functions -----------------
def euclidean_distance(actual_position, optimal_position): 
    return np.linalg.norm(np.array(actual_position) - np.array(optimal_position))

def angular_difference(actual_position, optimal_position, reference_point):
    vec_actual = np.array(actual_position) - np.array(reference_point)
    vec_optimal = np.array(optimal_position) - np.array(reference_point)
    cos_theta = np.dot(vec_actual, vec_optimal) / (np.linalg.norm(vec_actual) * np.linalg.norm(vec_optimal))
    return np.degrees(np.arccos(np.clip(cos_theta, -1.0, 1.0)))

def dive_radius_overlap(actual_position, optimal_position, actual_radius, optimal_radius):
    center_distance = np.linalg.norm(np.array(actual_position) - np.array(optimal_position))
    if center_distance >= actual_radius + optimal_radius:
        return 0.0
    elif center_distance <= abs(actual_radius - optimal_radius):
        return 1.0
    else:
        r1, r2 = actual_radius, optimal_radius
        d = center_distance
        part1 = r1**2 * np.arccos((d**2 + r1**2 - r2**2) / (2 * d * r1))
        part2 = r2**2 * np.arccos((d**2 + r2**2 - r1**2) / (2 * d * r2))
        part3 = 0.5 * np.sqrt((-d + r1 + r2) * (d + r1 - r2) * (d - r1 + r2) * (d + r1 + r2))
        intersection_area = part1 + part2 - part3
        max_area = np.pi * min(r1, r2)**2
        return intersection_area / max_area

def distance_similarity(actual_position, optimal_position, shot_position, max_distance = 3, max_distance_ball = 3):
    gk_to_ogk = euclidean_distance(actual_position, optimal_position)
    gk_to_ball = euclidean_distance(actual_position, shot_position)
    ogk_to_ball = euclidean_distance(optimal_position,shot_position)
    distance_similarity = max(0, 1 - (gk_to_ogk / max_distance))
    diff = abs(gk_to_ball - ogk_to_ball)
    similarity_to_ball = max(0, 1 - (diff / max_distance_ball))
    
    return distance_similarity *0.9 + similarity_to_ball *0.1

def calculate_shot_angle(shooter, left_post, right_post):
    vec_left = np.array(left_post) - np.array(shooter)
    vec_right = np.array(right_post) - np.array(shooter)
    dot_product = np.dot(vec_left, vec_right)
    mag_left = np.linalg.norm(vec_left)
    mag_right = np.linalg.norm(vec_right)
    angle = np.arccos(np.clip(dot_product / (mag_left * mag_right), -1.0, 1.0))
    return angle

def calculate_angle_radians(GK, Shot):
    delta_x = Shot[0] - GK[0]
    delta_y = GK[1] - Shot[1]
    return np.arctan2(delta_y, delta_x)

def calculate_angular_difference(actual_orientation, optimal_orientation, shot, left_post, right_post):
    angular_diff = abs(actual_orientation - optimal_orientation)
    shot_angle = calculate_shot_angle(shot, left_post, right_post)
    if angular_diff > shot_angle:
        angular_diff = shot_angle
    return angular_diff / shot_angle

def angular_similarity(actual_position, optimal_position, shot_position, threshold=3, alpha=0.2):
    actual_angle = calculate_angle_radians(actual_position, shot_position)
    optimal_angle = calculate_angle_radians(optimal_position, shot_position)
    base_error = calculate_angular_difference(abs(actual_angle), abs(optimal_angle), shot_position, B, C)

    keeper_distance = min(euclidean_distance(actual_position, optimal_position), euclidean_distance(actual_position,shot_position))
    if keeper_distance < threshold:
        factor = 1 - alpha * (1 - keeper_distance / threshold)
    else:
        factor = 1
    final_error = base_error * factor

    return final_error

# ----------------- StatsBomb API Data Retrieval -----------------
@st.cache_data
def get_shot_data(competition_id, season_id, team, GK_name):
    """
    Fetches shot data from StatsBomb for the given competition, season, and team.
    Returns:
      - gk_pos: detected goalkeeper positions
      - shot_loc: corresponding shot locations
    """
    matches = sb.matches(competition_id=competition_id, season_id=season_id)
    gk_pos = []
    shot_loc = []
    matches = matches[(matches['away_team'] == team) | (matches['home_team'] == team)]
    for fixture in matches['match_id']:
        event_data = sb.events(match_id=fixture, split=True, flatten_attrs=False)
        shots_data = event_data['shots']
        opp_shots = shots_data[
            (shots_data['possession_team'] != team) &
            (~shots_data['play_pattern'].isin(["From Free Kick", "From Corner", "Other"]))
        ]
        for idx, shot in opp_shots.iterrows():
            shot_info = shot['shot']
            if shot_info['outcome']['name'] == 'Blocked':
                continue
            if shot_info['body_part']['name'] == 'Head':
                continue
            
            for player in shot_info['freeze_frame']:
                if player['position']['name'] == 'Goalkeeper' and GK_name in player['player']['name']:
                    shot_loc.append(shot['location'])
                    gk_pos.append(player['location'])
    return gk_pos, shot_loc

# ----------------- Optimal Positioning Metric Calculation -----------------
def calculate_OPM(gk_pos, shot_loc):
    positioning_metric = []
    figures = []
    for i in range(len(gk_pos)):
        shot_position = np.array(shot_loc[i])
        actual_position = np.array(gk_pos[i])
        optimal_position = calculate_optimal_gk_position(shot_position)
        similarity_score = ((1 - angular_similarity(actual_position, optimal_position, shot_position))) * 100
        fig = plot_goalkeeper_position(shot_position, optimal_position, B, C, similarity=similarity_score, actual_position=actual_position)
        positioning_metric.append(similarity_score)
        figures.append(fig)
    season_average = np.mean(positioning_metric) if positioning_metric else 0
    return season_average, positioning_metric, figures

# ----------------- Streamlit App Interface -----------------
st.title("Optimal Goalkeeper Positioning %")
st.write("This app calculates the Optimal Goalkeeper Positioning %  (OGkP%) using StatsBomb data.")

# Sidebar: Competition Selection
st.sidebar.header("Select Competition and Team")

comps = get_competition_data()
if comps is not None and not comps.empty:
    # Create a combined column for display
    comps["comp_season"] = comps["competition_name"] + " - " + comps["season_name"]
    selected_comp = st.sidebar.selectbox("Select Competition", comps["comp_season"].unique())
    comp_row = comps[comps["comp_season"] == selected_comp].iloc[0]
    competition_id = comp_row["competition_id"]
    season_id = comp_row["season_id"]
else:
    st.sidebar.error("No competition data available.")
    competition_id = 55
    season_id = 282

# Sidebar: Team Selection
teams = get_team_data(competition_id, season_id)
if teams:
    selected_team = st.sidebar.selectbox("Select Team", teams)
else:
    st.sidebar.error("No team data available.")
    selected_team = "Portugal"

# Sidebar: Goalkeeper Selection from Lineup
gk_list = get_team_gk(competition_id, season_id, selected_team)
if gk_list:
    selected_gk = st.sidebar.selectbox("Select Goalkeeper", options=gk_list)
else:
    st.sidebar.error("No goalkeeper data available for the selected team.")
    selected_gk = ""

if st.sidebar.button("Run Analysis"):
    with st.spinner("Fetching shot data and calculating OGkPos..."):
        gk_positions, shot_locations = get_shot_data(competition_id, season_id, selected_team, selected_gk)
        st.write(f"Found {len(gk_positions)} goalkeeper positions and {len(shot_locations)} shot locations.")
        if not gk_positions or not shot_locations:
            st.error("No valid data found for the given parameters.")
        else:
            season_avg, metrics, figures = calculate_OPM(gk_positions, shot_locations)
            st.success(f"Season Average Optimal Positioning %: {season_avg:.2f}%")
            
            # Combine metrics and figures and sort by similarity score descending (highest first)
            shot_data = list(zip(metrics, figures))
            shot_data_sorted = sorted(shot_data, key=lambda x: x[0], reverse=False)
            top_shots = shot_data_sorted[:10]  # Only the top 10 shots

            st.header("Top 10 Shot-by-Shot Analysis")
            for i in range(0, len(top_shots), 2):
                cols = st.columns(2)
                for j, (score, fig) in enumerate(top_shots[i:i+2]):
                    fig.set_size_inches(8,8)  # Make the figure smaller
                    with cols[j]:
                        st.subheader(f"\nOGkPos%: {score:.2f}%")
                        st.pyplot(fig, use_container_width=True)
