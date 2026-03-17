import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
import warnings
import os

warnings.filterwarnings('ignore')

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="IPL Match Predictor",
    page_icon="🏏",
    layout="wide"
)

# ============================================================
# TEAM AND VENUE DATA
# ============================================================
TEAMS = [
    'Chennai Super Kings',
    'Delhi Capitals',
    'Gujarat Titans',
    'Kolkata Knight Riders',
    'Lucknow Super Giants',
    'Mumbai Indians',
    'Punjab Kings',
    'Rajasthan Royals',
    'Royal Challengers Bengaluru',
    'Sunrisers Hyderabad'
]

TEAM_SHORT = {
    'Chennai Super Kings': 'CSK',
    'Delhi Capitals': 'DC',
    'Gujarat Titans': 'GT',
    'Kolkata Knight Riders': 'KKR',
    'Lucknow Super Giants': 'LSG',
    'Mumbai Indians': 'MI',
    'Punjab Kings': 'PBKS',
    'Rajasthan Royals': 'RR',
    'Royal Challengers Bengaluru': 'RCB',
    'Sunrisers Hyderabad': 'SRH'
}

TEAM_COLORS = {
    'Chennai Super Kings': '#FDB913',
    'Delhi Capitals': '#004C93',
    'Gujarat Titans': '#1B2133',
    'Kolkata Knight Riders': '#3A225D',
    'Lucknow Super Giants': '#005DA0',
    'Mumbai Indians': '#004BA0',
    'Punjab Kings': '#ED1B24',
    'Rajasthan Royals': '#EA1A85',
    'Royal Challengers Bengaluru': '#EC1C24',
    'Sunrisers Hyderabad': '#FF822A'
}

VENUES = {
    'Chennai': 'MA Chidambaram Stadium',
    'Mumbai': 'Wankhede Stadium',
    'Bengaluru': 'M Chinnaswamy Stadium',
    'Kolkata': 'Eden Gardens',
    'Hyderabad': 'Rajiv Gandhi International Stadium',
    'Jaipur': 'Sawai Mansingh Stadium',
    'Delhi': 'Arun Jaitley Stadium',
    'Mohali': 'Punjab Cricket Association Stadium',
    'Ahmedabad': 'Narendra Modi Stadium',
    'Lucknow': 'Bharat Ratna Shri Atal Bihari Vajpayee Ekana Cricket Stadium'
}

TEAM_MAPPING = {
    'Delhi Daredevils': 'Delhi Capitals',
    'Deccan Chargers': 'Sunrisers Hyderabad',
    'Gujarat Lions': 'Gujarat Titans',
    'Rising Pune Supergiant': 'Rising Pune Supergiants',
    'Rising Pune Supergiants': 'Rising Pune Supergiants',
    'Kings XI Punjab': 'Punjab Kings',
    'Pune Warriors': 'Rising Pune Supergiants'
}

# ============================================================
# DATA LOADING AND CACHING
# ============================================================
@st.cache_data
def load_and_clean_data():
    """Load and clean the IPL dataset."""
    script_dir = os.path.dirname(os.path.abspath(__file__))

    matches_path = os.path.join(script_dir, 'matches.csv')
    deliveries_path = os.path.join(script_dir, 'deliveries.csv')

    if not os.path.exists(matches_path) or not os.path.exists(deliveries_path):
        st.error("matches.csv and deliveries.csv not found! Place them in the same folder as this script.")
        st.stop()

    matches = pd.read_csv(matches_path)
    deliveries = pd.read_csv(deliveries_path)

    # Clean team names
    for col in ['team1', 'team2', 'toss_winner', 'winner']:
        if col in matches.columns:
            matches[col] = matches[col].replace(TEAM_MAPPING)

    for col in ['batting_team', 'bowling_team']:
        if col in deliveries.columns:
            deliveries[col] = deliveries[col].replace(TEAM_MAPPING)

    # Remove no-result matches
    if 'result' in matches.columns:
        matches = matches[matches['result'] != 'no result']
    matches = matches.dropna(subset=['winner'])
    matches['date'] = pd.to_datetime(matches['date'])
    matches = matches.sort_values('date').reset_index(drop=True)

    return matches, deliveries

# ============================================================
# FEATURE ENGINEERING FUNCTIONS
# ============================================================
def get_team_win_pct(team, date, df, last_n=10):
    past = df[(df['date'] < date) & ((df['team1'] == team) | (df['team2'] == team))].tail(last_n)
    if len(past) == 0:
        return 0.5
    return (past['winner'] == team).sum() / len(past)

def get_head_to_head(team1, team2, date, df):
    past = df[(df['date'] < date) &
              (((df['team1'] == team1) & (df['team2'] == team2)) |
               ((df['team1'] == team2) & (df['team2'] == team1)))]
    if len(past) == 0:
        return 0.5
    return (past['winner'] == team1).sum() / len(past)

def get_venue_win_rate(team, venue, date, df):
    past = df[(df['date'] < date) & (df['venue'] == venue) &
              ((df['team1'] == team) | (df['team2'] == team))]
    if len(past) == 0:
        return 0.5
    return (past['winner'] == team).sum() / len(past)

def get_team_batting_stats(team, date, matches_df, deliveries_df, last_n=10):
    past_ids = matches_df[
        (matches_df['date'] < date) &
        ((matches_df['team1'] == team) | (matches_df['team2'] == team))
    ].tail(last_n)['id'].values

    batting = deliveries_df[
        (deliveries_df['match_id'].isin(past_ids)) &
        (deliveries_df['batting_team'] == team)
    ]

    if len(batting) == 0:
        return 7.5, 140

    total_runs = batting['total_runs'].sum()
    total_overs = batting.groupby('match_id')['over'].nunique().sum()
    avg_run_rate = total_runs / max(total_overs, 1)
    avg_score = batting.groupby('match_id')['total_runs'].sum().mean()

    return round(avg_run_rate, 2), round(avg_score, 2)

def get_team_bowling_stats(team, date, matches_df, deliveries_df, last_n=10):
    past_ids = matches_df[
        (matches_df['date'] < date) &
        ((matches_df['team1'] == team) | (matches_df['team2'] == team))
    ].tail(last_n)['id'].values

    bowling = deliveries_df[
        (deliveries_df['match_id'].isin(past_ids)) &
        (deliveries_df['bowling_team'] == team)
    ]

    if len(bowling) == 0:
        return 8.0, 3.0

    total_runs = bowling['total_runs'].sum()
    total_overs = bowling.groupby('match_id')['over'].nunique().sum()
    economy = total_runs / max(total_overs, 1)

    wickets = bowling['is_wicket'].sum() if 'is_wicket' in bowling.columns else 0
    wickets_per_match = wickets / max(len(past_ids), 1)

    return round(economy, 2), round(wickets_per_match, 2)

def get_powerplay_avg(team, date, matches_df, deliveries_df, last_n=10):
    past_ids = matches_df[
        (matches_df['date'] < date) &
        ((matches_df['team1'] == team) | (matches_df['team2'] == team))
    ].tail(last_n)['id'].values

    pp = deliveries_df[
        (deliveries_df['match_id'].isin(past_ids)) &
        (deliveries_df['batting_team'] == team) &
        (deliveries_df['over'] < 6)
    ]

    if len(pp) == 0:
        return 45
    return round(pp.groupby('match_id')['total_runs'].sum().mean(), 2)

def get_death_overs_avg(team, date, matches_df, deliveries_df, last_n=10):
    past_ids = matches_df[
        (matches_df['date'] < date) &
        ((matches_df['team1'] == team) | (matches_df['team2'] == team))
    ].tail(last_n)['id'].values

    death = deliveries_df[
        (deliveries_df['match_id'].isin(past_ids)) &
        (deliveries_df['batting_team'] == team) &
        (deliveries_df['over'] >= 16)
    ]

    if len(death) == 0:
        return 40
    return round(death.groupby('match_id')['total_runs'].sum().mean(), 2)

def get_middle_overs_avg(team, date, matches_df, deliveries_df, last_n=10):
    past_ids = matches_df[
        (matches_df['date'] < date) &
        ((matches_df['team1'] == team) | (matches_df['team2'] == team))
    ].tail(last_n)['id'].values

    middle = deliveries_df[
        (deliveries_df['match_id'].isin(past_ids)) &
        (deliveries_df['batting_team'] == team) &
        (deliveries_df['over'] >= 6) &
        (deliveries_df['over'] < 16)
    ]

    if len(middle) == 0:
        return 70
    return round(middle.groupby('match_id')['total_runs'].sum().mean(), 2)

def get_chase_rate(venue, date, df):
    past = df[(df['date'] < date) & (df['venue'] == venue)]
    if len(past) == 0:
        return 0.52
    if 'win_by_wickets' in df.columns:
        chasing_wins = past[past['win_by_wickets'] > 0].shape[0]
    elif 'result_margin' in df.columns and 'result' in df.columns:
        chasing_wins = past[past['result'] == 'wickets'].shape[0]
    else:
        chasing_wins = past[past['winner'] == past['team2']].shape[0]
    return round(chasing_wins / len(past), 2)

# ============================================================
# MODEL TRAINING
# ============================================================
@st.cache_resource
def train_model(_matches, _deliveries):
    """Train the XGBoost model on historical data."""
    features = []
    for idx, row in _matches.iterrows():
        t1_rr, t1_avg = get_team_batting_stats(row['team1'], row['date'], _matches, _deliveries)
        t2_rr, t2_avg = get_team_batting_stats(row['team2'], row['date'], _matches, _deliveries)
        t1_econ, t1_wpm = get_team_bowling_stats(row['team1'], row['date'], _matches, _deliveries)
        t2_econ, t2_wpm = get_team_bowling_stats(row['team2'], row['date'], _matches, _deliveries)

        features.append({
            'team1_win_pct': get_team_win_pct(row['team1'], row['date'], _matches),
            'team2_win_pct': get_team_win_pct(row['team2'], row['date'], _matches),
            'head_to_head': get_head_to_head(row['team1'], row['team2'], row['date'], _matches),
            'toss_win': 1 if row['toss_winner'] == row['team1'] else 0,
            'chose_bat': 1 if row['toss_decision'] == 'bat' else 0,
            'venue_win_rate': get_venue_win_rate(row['team1'], row['venue'], row['date'], _matches),
            'chase_rate_venue': get_chase_rate(row['venue'], row['date'], _matches),
            'team1_run_rate': t1_rr,
            'team2_run_rate': t2_rr,
            'team1_avg_score': t1_avg,
            'team2_avg_score': t2_avg,
            'team1_bowl_economy': t1_econ,
            'team2_bowl_economy': t2_econ,
            'team1_wickets_pm': t1_wpm,
            'team2_wickets_pm': t2_wpm,
            'team1_powerplay': get_powerplay_avg(row['team1'], row['date'], _matches, _deliveries),
            'team2_powerplay': get_powerplay_avg(row['team2'], row['date'], _matches, _deliveries),
            'team1_death': get_death_overs_avg(row['team1'], row['date'], _matches, _deliveries),
            'team2_death': get_death_overs_avg(row['team2'], row['date'], _matches, _deliveries),
            'team1_middle': get_middle_overs_avg(row['team1'], row['date'], _matches, _deliveries),
            'team2_middle': get_middle_overs_avg(row['team2'], row['date'], _matches, _deliveries),
            'target': 1 if row['winner'] == row['team1'] else 0
        })

    feature_df = pd.DataFrame(features)

    X = feature_df.drop('target', axis=1)
    y = feature_df['target']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    model = XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric='logloss'
    )

    model.fit(X_train, y_train)

    test_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, test_pred)

    feature_importance = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)

    return model, accuracy, feature_importance, X.columns.tolist()

# ============================================================
# PREDICTION FUNCTION
# ============================================================
def make_prediction(model, team1, team2, venue, toss_winner, toss_decision, matches, deliveries):
    """Generate predictions for a match."""
    ref_date = pd.Timestamp('2025-04-01')

    t1_rr, t1_avg = get_team_batting_stats(team1, ref_date, matches, deliveries)
    t2_rr, t2_avg = get_team_batting_stats(team2, ref_date, matches, deliveries)
    t1_econ, t1_wpm = get_team_bowling_stats(team1, ref_date, matches, deliveries)
    t2_econ, t2_wpm = get_team_bowling_stats(team2, ref_date, matches, deliveries)
    t1_pp = get_powerplay_avg(team1, ref_date, matches, deliveries)
    t2_pp = get_powerplay_avg(team2, ref_date, matches, deliveries)
    t1_death = get_death_overs_avg(team1, ref_date, matches, deliveries)
    t2_death = get_death_overs_avg(team2, ref_date, matches, deliveries)
    t1_middle = get_middle_overs_avg(team1, ref_date, matches, deliveries)
    t2_middle = get_middle_overs_avg(team2, ref_date, matches, deliveries)

    input_data = {
        'team1_win_pct': get_team_win_pct(team1, ref_date, matches),
        'team2_win_pct': get_team_win_pct(team2, ref_date, matches),
        'head_to_head': get_head_to_head(team1, team2, ref_date, matches),
        'toss_win': 1 if toss_winner == team1 else 0,
        'chose_bat': 1 if toss_decision == 'Bat' else 0,
        'venue_win_rate': get_venue_win_rate(team1, venue, ref_date, matches),
        'chase_rate_venue': get_chase_rate(venue, ref_date, matches),
        'team1_run_rate': t1_rr,
        'team2_run_rate': t2_rr,
        'team1_avg_score': t1_avg,
        'team2_avg_score': t2_avg,
        'team1_bowl_economy': t1_econ,
        'team2_bowl_economy': t2_econ,
        'team1_wickets_pm': t1_wpm,
        'team2_wickets_pm': t2_wpm,
        'team1_powerplay': t1_pp,
        'team2_powerplay': t2_pp,
        'team1_death': t1_death,
        'team2_death': t2_death,
        'team1_middle': t1_middle,
        'team2_middle': t2_middle,
    }

    prob = model.predict_proba(pd.DataFrame([input_data]))[0]

    # Additional stats for display
    stats = {
        'team1_win_prob': prob[1],
        'team2_win_prob': prob[0],
        'team1_win_pct': input_data['team1_win_pct'],
        'team2_win_pct': input_data['team2_win_pct'],
        'head_to_head': input_data['head_to_head'],
        'team1_run_rate': t1_rr,
        'team2_run_rate': t2_rr,
        'team1_avg_score': t1_avg,
        'team2_avg_score': t2_avg,
        'team1_bowl_economy': t1_econ,
        'team2_bowl_economy': t2_econ,
        'team1_powerplay': t1_pp,
        'team2_powerplay': t2_pp,
        'team1_death': t1_death,
        'team2_death': t2_death,
        'team1_middle': t1_middle,
        'team2_middle': t2_middle,
        'venue_win_rate': input_data['venue_win_rate'],
        'chase_rate': input_data['chase_rate_venue'],
    }

    return stats

# ============================================================
# UI HELPER FUNCTIONS
# ============================================================
def create_probability_gauge(team1, team2, prob1, prob2, color1, color2):
    """Create a horizontal bar showing win probabilities."""
    fig = go.Figure()

    fig.add_trace(go.Bar(
        y=['Match Winner'],
        x=[prob1 * 100],
        orientation='h',
        name=TEAM_SHORT[team1],
        marker_color=color1,
        text=f"{TEAM_SHORT[team1]} {prob1:.1%}",
        textposition='inside',
        textfont=dict(size=16, color='white')
    ))

    fig.add_trace(go.Bar(
        y=['Match Winner'],
        x=[prob2 * 100],
        orientation='h',
        name=TEAM_SHORT[team2],
        marker_color=color2,
        text=f"{TEAM_SHORT[team2]} {prob2:.1%}",
        textposition='inside',
        textfont=dict(size=16, color='white')
    ))

    fig.update_layout(
        barmode='stack',
        height=100,
        margin=dict(l=0, r=0, t=0, b=0),
        showlegend=False,
        xaxis=dict(visible=False, range=[0, 100]),
        yaxis=dict(visible=False),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
    )

    return fig

def create_phase_comparison(team1, team2, stats, color1, color2):
    """Create a bar chart comparing teams across match phases."""
    phases = ['Powerplay (1-6)', 'Middle (7-16)', 'Death (17-20)']
    t1_vals = [stats['team1_powerplay'], stats['team1_middle'], stats['team1_death']]
    t2_vals = [stats['team2_powerplay'], stats['team2_middle'], stats['team2_death']]

    fig = go.Figure()

    fig.add_trace(go.Bar(
        name=TEAM_SHORT[team1],
        x=phases,
        y=t1_vals,
        marker_color=color1,
        text=[f"{v:.0f}" for v in t1_vals],
        textposition='outside'
    ))

    fig.add_trace(go.Bar(
        name=TEAM_SHORT[team2],
        x=phases,
        y=t2_vals,
        marker_color=color2,
        text=[f"{v:.0f}" for v in t2_vals],
        textposition='outside'
    ))

    fig.update_layout(
        title="Average Runs by Match Phase (Last 10 Matches)",
        barmode='group',
        height=350,
        margin=dict(l=40, r=20, t=50, b=40),
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='center', x=0.5),
        yaxis_title="Avg Runs",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
    )

    return fig

def create_stats_comparison(team1, team2, stats, color1, color2):
    """Create a radar chart comparing team stats."""
    categories = ['Win Form', 'Bat Run Rate', 'Bowl Economy', 'Powerplay', 'Death Overs']

    # Normalize values to 0-10 scale for radar chart
    t1_vals = [
        stats['team1_win_pct'] * 10,
        min(stats['team1_run_rate'] / 1.2, 10),
        max(10 - stats['team1_bowl_economy'], 0),
        min(stats['team1_powerplay'] / 6, 10),
        min(stats['team1_death'] / 5, 10),
    ]
    t2_vals = [
        stats['team2_win_pct'] * 10,
        min(stats['team2_run_rate'] / 1.2, 10),
        max(10 - stats['team2_bowl_economy'], 0),
        min(stats['team2_powerplay'] / 6, 10),
        min(stats['team2_death'] / 5, 10),
    ]

    # Close the radar
    t1_vals.append(t1_vals[0])
    t2_vals.append(t2_vals[0])
    categories.append(categories[0])

    fig = go.Figure()

    fig.add_trace(go.Scatterpolar(
        r=t1_vals, theta=categories, fill='toself',
        name=TEAM_SHORT[team1], line_color=color1, fillcolor=color1, opacity=0.3
    ))

    fig.add_trace(go.Scatterpolar(
        r=t2_vals, theta=categories, fill='toself',
        name=TEAM_SHORT[team2], line_color=color2, fillcolor=color2, opacity=0.3
    ))

    fig.update_layout(
        title="Team Strength Comparison",
        polar=dict(radialaxis=dict(visible=True, range=[0, 10])),
        height=400,
        margin=dict(l=60, r=60, t=50, b=40),
        legend=dict(orientation='h', yanchor='bottom', y=1.05, xanchor='center', x=0.5),
        paper_bgcolor='rgba(0,0,0,0)',
    )

    return fig

def get_key_factors(stats, team1, team2):
    """Determine the key factors influencing the prediction."""
    factors = []

    # Recent form
    if abs(stats['team1_win_pct'] - stats['team2_win_pct']) > 0.2:
        better = team1 if stats['team1_win_pct'] > stats['team2_win_pct'] else team2
        factors.append(f"**Recent Form:** {TEAM_SHORT[better]} in significantly better form")

    # Head to head
    h2h = stats['head_to_head']
    if h2h > 0.6:
        factors.append(f"**Head-to-Head:** {TEAM_SHORT[team1]} dominates this matchup ({h2h:.0%} wins)")
    elif h2h < 0.4:
        factors.append(f"**Head-to-Head:** {TEAM_SHORT[team2]} dominates this matchup ({1-h2h:.0%} wins)")

    # Venue
    if stats['venue_win_rate'] > 0.6:
        factors.append(f"**Venue:** {TEAM_SHORT[team1]} has a strong record here ({stats['venue_win_rate']:.0%} wins)")
    elif stats['venue_win_rate'] < 0.4:
        factors.append(f"**Venue:** {TEAM_SHORT[team2]} has a strong record here ({1-stats['venue_win_rate']:.0%} wins)")

    # Batting
    if abs(stats['team1_avg_score'] - stats['team2_avg_score']) > 10:
        better = team1 if stats['team1_avg_score'] > stats['team2_avg_score'] else team2
        factors.append(f"**Batting:** {TEAM_SHORT[better]} scoring higher on average recently")

    # Bowling
    if abs(stats['team1_bowl_economy'] - stats['team2_bowl_economy']) > 0.5:
        better = team1 if stats['team1_bowl_economy'] < stats['team2_bowl_economy'] else team2
        factors.append(f"**Bowling:** {TEAM_SHORT[better]} has tighter bowling economy")

    # Powerplay
    if abs(stats['team1_powerplay'] - stats['team2_powerplay']) > 5:
        better = team1 if stats['team1_powerplay'] > stats['team2_powerplay'] else team2
        factors.append(f"**Powerplay:** {TEAM_SHORT[better]} stronger in overs 1-6")

    # Death overs
    if abs(stats['team1_death'] - stats['team2_death']) > 5:
        better = team1 if stats['team1_death'] > stats['team2_death'] else team2
        factors.append(f"**Death Overs:** {TEAM_SHORT[better]} stronger in overs 17-20")

    if not factors:
        factors.append("**Evenly matched** — no single dominant factor")

    return factors

# ============================================================
# MAIN APP
# ============================================================
def main():
    # Header
    st.markdown("""
    <h1 style='text-align: center;'>🏏 IPL Match Predictor</h1>
    <p style='text-align: center; color: gray;'>Powered by XGBoost ML Model | Trained on 17 Years of IPL Data (2008-2024)</p>
    <hr>
    """, unsafe_allow_html=True)

    # Load data
    with st.spinner("Loading IPL data..."):
        matches, deliveries = load_and_clean_data()

    # Train model
    with st.spinner("Training prediction model... (this takes a few minutes on first run, then it's cached)"):
        model, accuracy, feature_importance, feature_names = train_model(matches, deliveries)

    # Model accuracy badge
    st.markdown(f"""
    <p style='text-align: center;'>
        Model Accuracy: <strong>{accuracy:.1%}</strong> |
        Matches Analysed: <strong>{len(matches)}</strong> |
        Features: <strong>21</strong> |
        Algorithm: <strong>XGBoost</strong>
    </p>
    <hr>
    """, unsafe_allow_html=True)

    # ── INPUT SECTION ──
    st.subheader("Match Details")

    col1, col2, col3 = st.columns(3)

    with col1:
        team1 = st.selectbox("Team A", TEAMS, index=9, key='team1')  # Default: SRH

    with col2:
        team2_options = [t for t in TEAMS if t != team1]
        team2 = st.selectbox("Team B", team2_options, index=7, key='team2')  # Default: RCB

    with col3:
        venue_city = st.selectbox("Venue", list(VENUES.keys()), index=3)
        venue = VENUES[venue_city]

    col4, col5 = st.columns(2)

    with col4:
        toss_winner = st.selectbox("Toss Won By", [team1, team2])

    with col5:
        toss_decision = st.selectbox("Toss Decision", ['Field', 'Bat'])

    st.markdown("<br>", unsafe_allow_html=True)

    # ── PREDICT BUTTON ──
    predict_clicked = st.button("🏏  Predict Match", type="primary", use_container_width=True)

    if predict_clicked:
        with st.spinner("Analysing match data..."):
            stats = make_prediction(model, team1, team2, venue, toss_winner, toss_decision, matches, deliveries)

        color1 = TEAM_COLORS.get(team1, '#333333')
        color2 = TEAM_COLORS.get(team2, '#666666')

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("---")

        # ── MAIN PREDICTION RESULT ──
        st.subheader("Match Winner Prediction")
        st.plotly_chart(
            create_probability_gauge(team1, team2, stats['team1_win_prob'], stats['team2_win_prob'], color1, color2),
            use_container_width=True
        )

        # Verdict
        if stats['team1_win_prob'] > 0.65:
            verdict = f"**STRONG PREDICTION: {team1} ({TEAM_SHORT[team1]})**"
        elif stats['team2_win_prob'] > 0.65:
            verdict = f"**STRONG PREDICTION: {team2} ({TEAM_SHORT[team2]})**"
        elif stats['team1_win_prob'] > 0.55:
            verdict = f"**SLIGHT EDGE: {team1} ({TEAM_SHORT[team1]})**"
        elif stats['team2_win_prob'] > 0.55:
            verdict = f"**SLIGHT EDGE: {team2} ({TEAM_SHORT[team2]})**"
        else:
            verdict = "**VERY CLOSE — Could go either way!**"

        st.markdown(f"<h3 style='text-align: center;'>{verdict}</h3>", unsafe_allow_html=True)

        st.markdown("---")

        # ── PHASE-WISE PREDICTION ──
        st.subheader("Phase-by-Phase Breakdown")

        pp_col, mid_col, death_col = st.columns(3)

        with pp_col:
            pp_leader = team1 if stats['team1_powerplay'] > stats['team2_powerplay'] else team2
            pp_diff = abs(stats['team1_powerplay'] - stats['team2_powerplay'])
            st.metric(
                label="Powerplay Leader (Overs 1-6)",
                value=TEAM_SHORT[pp_leader],
                delta=f"+{pp_diff:.0f} avg runs"
            )
            st.caption(f"{TEAM_SHORT[team1]}: {stats['team1_powerplay']:.0f} avg | {TEAM_SHORT[team2]}: {stats['team2_powerplay']:.0f} avg")

        with mid_col:
            mid_leader = team1 if stats['team1_middle'] > stats['team2_middle'] else team2
            mid_diff = abs(stats['team1_middle'] - stats['team2_middle'])
            st.metric(
                label="Middle Overs Leader (7-16)",
                value=TEAM_SHORT[mid_leader],
                delta=f"+{mid_diff:.0f} avg runs"
            )
            st.caption(f"{TEAM_SHORT[team1]}: {stats['team1_middle']:.0f} avg | {TEAM_SHORT[team2]}: {stats['team2_middle']:.0f} avg")

        with death_col:
            death_leader = team1 if stats['team1_death'] > stats['team2_death'] else team2
            death_diff = abs(stats['team1_death'] - stats['team2_death'])
            st.metric(
                label="Death Overs Leader (17-20)",
                value=TEAM_SHORT[death_leader],
                delta=f"+{death_diff:.0f} avg runs"
            )
            st.caption(f"{TEAM_SHORT[team1]}: {stats['team1_death']:.0f} avg | {TEAM_SHORT[team2]}: {stats['team2_death']:.0f} avg")

        st.markdown("<br>", unsafe_allow_html=True)

        # ── CHARTS ──
        chart_col1, chart_col2 = st.columns(2)

        with chart_col1:
            st.plotly_chart(
                create_phase_comparison(team1, team2, stats, color1, color2),
                use_container_width=True
            )

        with chart_col2:
            st.plotly_chart(
                create_stats_comparison(team1, team2, stats, color1, color2),
                use_container_width=True
            )

        st.markdown("---")

        # ── DETAILED STATS TABLE ──
        st.subheader("Head-to-Head Stats")

        stat_data = {
            'Stat': [
                'Recent Win Rate (Last 10)',
                'Avg Score (Last 10)',
                'Run Rate',
                'Bowl Economy',
                'Wickets/Match',
                'Avg Powerplay Score',
                'Avg Middle Overs',
                'Avg Death Overs',
                f'Win Rate at {venue_city}',
            ],
            TEAM_SHORT[team1]: [
                f"{stats['team1_win_pct']:.0%}",
                f"{stats['team1_avg_score']:.0f}",
                f"{stats['team1_run_rate']:.1f}",
                f"{stats['team1_bowl_economy']:.1f}",
                f"{stats.get('team1_wickets_pm', 'N/A')}",
                f"{stats['team1_powerplay']:.0f}",
                f"{stats['team1_middle']:.0f}",
                f"{stats['team1_death']:.0f}",
                f"{stats['venue_win_rate']:.0%}",
            ],
            TEAM_SHORT[team2]: [
                f"{stats['team2_win_pct']:.0%}",
                f"{stats['team2_avg_score']:.0f}",
                f"{stats['team2_run_rate']:.1f}",
                f"{stats['team2_bowl_economy']:.1f}",
                f"{stats.get('team2_wickets_pm', 'N/A')}",
                f"{stats['team2_powerplay']:.0f}",
                f"{stats['team2_middle']:.0f}",
                f"{stats['team2_death']:.0f}",
                f"{1 - stats['venue_win_rate']:.0%}",
            ]
        }

        st.dataframe(pd.DataFrame(stat_data).set_index('Stat'), use_container_width=True)

        st.markdown("---")

        # ── KEY FACTORS ──
        st.subheader("Key Factors")
        factors = get_key_factors(stats, team1, team2)
        for f in factors:
            st.markdown(f"- {f}")

        st.markdown("---")

        # ── VENUE INFO ──
        st.subheader("Venue Insight")
        chase_pct = stats['chase_rate'] * 100
        defend_pct = (1 - stats['chase_rate']) * 100
        st.markdown(f"**{venue}** ({venue_city})")
        st.markdown(f"- Chasing team wins: **{chase_pct:.0f}%** of matches")
        st.markdown(f"- Defending team wins: **{defend_pct:.0f}%** of matches")
        if chase_pct > 55:
            st.markdown(f"- This venue **favours chasing**")
        elif defend_pct > 55:
            st.markdown(f"- This venue **favours defending**")
        else:
            st.markdown(f"- This venue is **neutral** — no clear advantage")

    # ── SIDEBAR ──
    with st.sidebar:
        st.markdown("### About")
        st.markdown("""
        This app uses **XGBoost machine learning** trained on
        every IPL match from 2008-2024 to predict match outcomes.

        **21 features** including:
        - Team recent form
        - Batting & bowling strength
        - Powerplay performance
        - Middle overs performance
        - Death overs performance
        - Venue history
        - Toss impact
        - Head-to-head record
        - Chase vs defend rates
        """)

        st.markdown("---")
        st.markdown(f"**Model Accuracy:** {accuracy:.1%}")
        st.markdown(f"**Matches in Data:** {len(matches)}")
        st.markdown(f"**Algorithm:** XGBoost (300 trees)")

        st.markdown("---")
        st.markdown("### Top Prediction Factors")
        for feat, imp in feature_importance.head(5).items():
            clean_name = feat.replace('_', ' ').title()
            st.markdown(f"- {clean_name}: **{imp:.1%}**")

        st.markdown("---")
        st.markdown("*Built for data analysis & fun — not gambling*")

if __name__ == '__main__':
    main()
