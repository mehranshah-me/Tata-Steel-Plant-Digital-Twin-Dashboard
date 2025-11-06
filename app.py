"""
Interactive Dashboard for Steel Plant Digital Safety Twin
=========================================================

Streamlit-based web application for visualizing safety KPIs,
incident trends, and running what-if scenarios.

Features:
- Real-time KPI monitoring
- Zone-wise risk heatmaps
- Incident severity analysis
- Interactive what-if scenario modeling

Run with: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import json
from pathlib import Path
import sys

# Import simulation module
from safety_twin_simulation import SteelPlantDigitalTwin, ZONE_CONFIGS, BenefitCostAnalyzer


# Page configuration
st.set_page_config(
    page_title="Steel Plant Safety Twin",
    page_icon="⚙",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS with animations
st.markdown("""
<style>
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(-10px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    @keyframes slideIn {
        from { transform: translateX(-20px); opacity: 0; }
        to { transform: translateX(0); opacity: 1; }
    }
    
    @keyframes pulse {
        0%, 100% { transform: scale(1); }
        50% { transform: scale(1.02); }
    }
    
    @keyframes shimmer {
        0% { background-position: -1000px 0; }
        100% { background-position: 1000px 0; }
    }
    
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        padding: 1.5rem;
        background: linear-gradient(90deg, #e8f4f8 0%, #ffffff 50%, #e8f4f8 100%);
        background-size: 200% auto;
        animation: shimmer 3s linear infinite, fadeIn 0.8s ease-out;
        border-radius: 10px;
        margin-bottom: 2rem;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    
    .kpi-card {
        background-color: #f0f8ff;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
        margin-bottom: 1rem;
        animation: fadeIn 0.6s ease-out;
        transition: transform 0.3s, box-shadow 0.3s;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    
    .kpi-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.15);
    }
    
    .metric-label {
        font-size: 0.9rem;
        color: #666;
        font-weight: 600;
        animation: slideIn 0.5s ease-out;
    }
    
    .metric-value {
        font-size: 2rem;
        color: #1f77b4;
        font-weight: bold;
        animation: fadeIn 0.8s ease-out;
    }
    
    .stMetric {
        animation: fadeIn 0.6s ease-out;
    }
    
    .stPlotlyChart {
        animation: fadeIn 1s ease-out;
    }
    
    .success-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 8px;
        color: white;
        animation: pulse 2s infinite;
    }
    
    .warning-box {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 1rem;
        border-radius: 8px;
        color: white;
        animation: fadeIn 0.5s ease-out;
    }
    
    .info-box {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        padding: 1rem;
        border-radius: 8px;
        color: white;
        animation: fadeIn 0.5s ease-out;
    }
    
    /* Chart animation */
    .js-plotly-plot {
        animation: fadeIn 1s ease-out;
    }
    
    /* Button animations */
    .stButton > button {
        transition: all 0.3s ease;
        animation: fadeIn 0.5s ease-out;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_simulation_data():
    """Load simulation data from CSV."""
    data_path = Path('data/simulated_safety_data.csv')
    if data_path.exists():
        return pd.read_csv(data_path)
    return None


@st.cache_data
def load_kpi_data():
    """Load KPI data from JSON."""
    kpi_path = Path('data/safety_kpis.json')
    if kpi_path.exists():
        with open(kpi_path, 'r') as f:
            return json.load(f)
    return None


def calculate_rolling_kpis(df, window=30):
    """
    Calculate rolling KPIs for trend analysis.
    
    Args:
        df: Simulation data
        window: Rolling window size in days
    
    Returns:
        DataFrame with rolling KPIs
    """
    daily_data = df.groupby('day').agg({
        'incident_occurred': 'sum',
        'lost_hours': 'sum',
        'man_hours': 'sum',
        'is_recordable': 'sum'
    }).reset_index()
    
    # Calculate rolling metrics
    daily_data['rolling_ltifr'] = (
        (daily_data['incident_occurred'].rolling(window=window).sum() * 1e6) /
        daily_data['man_hours'].rolling(window=window).sum()
    )
    
    daily_data['rolling_lttr'] = (
        daily_data['lost_hours'].rolling(window=window).sum() /
        daily_data['incident_occurred'].rolling(window=window).sum()
    )
    
    daily_data['rolling_severity'] = (
        (daily_data['lost_hours'].rolling(window=window).sum() / 8 * 1e3) /
        daily_data['man_hours'].rolling(window=window).sum()
    )
    
    daily_data['rolling_trir'] = (
        (daily_data['is_recordable'].rolling(window=window).sum() * 1e6) /
        daily_data['man_hours'].rolling(window=window).sum()
    )
    
    return daily_data


def plot_kpi_trends(df):
    """Create multi-panel KPI trend chart with enhanced animations."""
    daily_data = calculate_rolling_kpis(df, window=30)
    
    # Create subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('LTIFR Trend (30-day rolling)', 
                       'LTTR Trend (30-day rolling)',
                       'Severity Rate Trend (30-day rolling)',
                       'TRIR Trend (30-day rolling)'),
        vertical_spacing=0.12,
        horizontal_spacing=0.10
    )
    
    # LTIFR with gradient fill
    fig.add_trace(
        go.Scatter(x=daily_data['day'], y=daily_data['rolling_ltifr'],
                  mode='lines+markers', name='LTIFR', 
                  line=dict(color='#ff7f0e', width=3, shape='spline'),
                  marker=dict(size=4, color='#ff7f0e'),
                  fill='tozeroy', fillcolor='rgba(255, 127, 14, 0.1)'),
        row=1, col=1
    )
    
    # LTTR with gradient fill
    fig.add_trace(
        go.Scatter(x=daily_data['day'], y=daily_data['rolling_lttr'],
                  mode='lines+markers', name='LTTR',
                  line=dict(color='#2ca02c', width=3, shape='spline'),
                  marker=dict(size=4, color='#2ca02c'),
                  fill='tozeroy', fillcolor='rgba(44, 160, 44, 0.1)'),
        row=1, col=2
    )
    
    # Severity Rate with gradient fill
    fig.add_trace(
        go.Scatter(x=daily_data['day'], y=daily_data['rolling_severity'],
                  mode='lines+markers', name='Severity',
                  line=dict(color='#d62728', width=3, shape='spline'),
                  marker=dict(size=4, color='#d62728'),
                  fill='tozeroy', fillcolor='rgba(214, 39, 40, 0.1)'),
        row=2, col=1
    )
    
    # TRIR with gradient fill
    fig.add_trace(
        go.Scatter(x=daily_data['day'], y=daily_data['rolling_trir'],
                  mode='lines+markers', name='TRIR',
                  line=dict(color='#9467bd', width=3, shape='spline'),
                  marker=dict(size=4, color='#9467bd'),
                  fill='tozeroy', fillcolor='rgba(148, 103, 189, 0.1)'),
        row=2, col=2
    )
    
    # Update axes
    fig.update_xaxes(title_text="Simulation Day", row=2, col=1)
    fig.update_xaxes(title_text="Simulation Day", row=2, col=2)
    fig.update_yaxes(title_text="LTIFR (per 1M man-hours)", row=1, col=1)
    fig.update_yaxes(title_text="LTTR (hours per LTI)", row=1, col=2)
    fig.update_yaxes(title_text="Severity Rate (lost days per 1k man-hours)", row=2, col=1)
    fig.update_yaxes(title_text="TRIR (per 1M man-hours)", row=2, col=2)
    
    fig.update_layout(
        height=600,
        showlegend=False,
        title_text="Safety KPI Trends Over Time",
        title_x=0.5,
        title_font_size=18,
        hovermode='x unified',
        template='plotly_white'
    )
    
    return fig


def plot_zone_heatmap(df):
    """Create zone-wise risk heatmap."""
    # Aggregate by zone and create risk matrix
    zone_metrics = df.groupby('zone').agg({
        'incident_probability': 'mean',
        'incident_occurred': 'sum',
        'lost_hours': 'sum',
        'equipment_load': 'mean',
        'maintenance_delay': 'mean',
        'fatigue_factor': 'mean',
        'man_hours': 'sum'
    }).reset_index()
    
    # Calculate LTIFR by zone
    zone_metrics['ltifr'] = (zone_metrics['incident_occurred'] * 1e6) / zone_metrics['man_hours']
    
    # Create heatmap data
    metrics = ['incident_probability', 'equipment_load', 'maintenance_delay', 
               'fatigue_factor', 'ltifr']
    metric_labels = ['Avg Incident Prob', 'Avg Equipment Load', 
                     'Avg Maint. Delay', 'Avg Fatigue', 'LTIFR']
    
    heatmap_data = []
    for metric in metrics:
        # Normalize to 0-1 scale for visualization
        values = zone_metrics[metric].values
        normalized = (values - values.min()) / (values.max() - values.min() + 1e-10)
        heatmap_data.append(normalized)
    
    heatmap_array = np.array(heatmap_data)
    
    fig = go.Figure(data=go.Heatmap(
        z=heatmap_array,
        x=zone_metrics['zone'].values,
        y=metric_labels,
        colorscale='Reds',
        text=np.round(heatmap_array, 3),
        texttemplate='%{text}',
        textfont={"size": 12},
        colorbar=dict(title="Normalized Risk")
    ))
    
    fig.update_layout(
        title="Zone-Wise Safety Risk Heatmap",
        title_x=0.5,
        xaxis_title="Production Zone",
        yaxis_title="Risk Metric",
        height=400
    )
    
    return fig


def plot_severity_histogram(df):
    """Create incident severity distribution histogram."""
    incident_df = df[df['incident_occurred'] == True].copy()
    
    if len(incident_df) == 0:
        # No incidents - return empty figure with message
        fig = go.Figure()
        fig.add_annotation(
            text="No incidents recorded in simulation",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=16)
        )
        return fig
    
    fig = go.Figure()
    
    # Create histogram by severity category
    severity_colors = {
        'Minor': '#2ca02c',
        'Moderate': '#ff7f0e',
        'Serious': '#d62728',
        'Severe': '#8b0000'
    }
    
    for severity in ['Minor', 'Moderate', 'Serious', 'Severe']:
        severity_data = incident_df[incident_df['severity'] == severity]
        if len(severity_data) > 0:
            fig.add_trace(go.Histogram(
                x=severity_data['lost_hours'],
                name=severity,
                marker_color=severity_colors[severity],
                opacity=0.7,
                nbinsx=30
            ))
    
    fig.update_layout(
        title="Incident Severity Distribution (Lost Hours)",
        title_x=0.5,
        xaxis_title="Lost Hours",
        yaxis_title="Frequency",
        barmode='overlay',
        height=400,
        showlegend=True,
        legend=dict(x=0.7, y=0.95)
    )
    
    return fig


def plot_plant_3d_view(df, ZONE_CONFIGS):
    """
    Simple production flow diagram with arrows.
    """
    # Production flow: Raw Materials → Hot Metal → Finishing
    zones = ['Coke_Oven', 'Sinter_Plant', 'Blast_Furnace', 'BOF', 'Ladle_Metallurgy', 'Caster', 'Rolling_Mill']
    zone_names = [ZONE_CONFIGS[z].name.replace('_', ' ') for z in zones]
    
    # Simple horizontal flow
    fig = go.Figure()
    
    # Add zones
    for i, zone in enumerate(zone_names):
        x_pos = i
        fig.add_annotation(x=x_pos, y=0, text=zone, showarrow=False, font=dict(size=10, color='#333'))
        
        # Flow arrow (except last)
        if i < len(zone_names) - 1:
            fig.add_annotation(x=x_pos + 0.45, y=0, ax=x_pos + 0.55, ay=0, 
                             arrowhead=2, arrowsize=1.2, arrowwidth=2, arrowcolor='#666')
    
    fig.update_layout(
        title="",
        xaxis=dict(showticklabels=False, showgrid=False, zeroline=False, range=[-0.5, 6.5]),
        yaxis=dict(showticklabels=False, showgrid=False, zeroline=False, range=[-0.5, 0.5]),
        plot_bgcolor='white',
        paper_bgcolor='white',
        height=200,
        margin=dict(l=20, r=20, t=20, b=20)
    )
    
    return fig


def plot_zone_comparison(df):
    """Create bar chart comparing incidents by zone."""
    zone_summary = df.groupby('zone').agg({
        'incident_occurred': 'sum',
        'lost_hours': 'sum',
        'man_hours': 'sum'
    }).reset_index()
    
    zone_summary['ltifr'] = (zone_summary['incident_occurred'] * 1e6) / zone_summary['man_hours']
    zone_summary['avg_lost_hours'] = zone_summary['lost_hours'] / zone_summary['incident_occurred']
    
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Total Incidents by Zone', 'LTIFR by Zone'),
        specs=[[{"type": "bar"}, {"type": "bar"}]]
    )
    
    fig.add_trace(
        go.Bar(x=zone_summary['zone'], y=zone_summary['incident_occurred'],
               marker_color='#1f77b4', name='Incidents'),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Bar(x=zone_summary['zone'], y=zone_summary['ltifr'],
               marker_color='#ff7f0e', name='LTIFR'),
        row=1, col=2
    )
    
    fig.update_xaxes(tickangle=45)
    fig.update_layout(height=400, showlegend=False)
    
    return fig


def run_whatif_scenario(equipment_load_adj, maintenance_delay_adj, days=90):
    """
    Run what-if scenario with adjusted parameters.
    Enhanced with realistic scenario validation.
    
    Args:
        equipment_load_adj: Adjustment factor for equipment load (0.5-1.5)
        maintenance_delay_adj: Adjustment factor for maintenance delay (0.5-2.0)
        days: Number of days to simulate
    
    Returns:
        Dictionary with scenario KPIs
    """
    # Validate realistic parameters
    equipment_load_adj = np.clip(equipment_load_adj, 0.5, 1.5)
    maintenance_delay_adj = np.clip(maintenance_delay_adj, 0.5, 2.0)
    
    # Create modified zone configs
    import copy
    modified_configs = copy.deepcopy(ZONE_CONFIGS)
    
    for zone in modified_configs:
        # Apply adjustments with realistic bounds
        new_load = modified_configs[zone].equipment_load_mean * equipment_load_adj
        modified_configs[zone].equipment_load_mean = np.clip(new_load, 0.5, 0.95)
        
        new_delay = modified_configs[zone].maintenance_delay_mean * maintenance_delay_adj
        modified_configs[zone].maintenance_delay_mean = max(5.0, new_delay)  # Min 5 days
    
    # Temporarily replace configs
    from safety_twin_simulation import ZONE_CONFIGS as original_configs
    import safety_twin_simulation
    safety_twin_simulation.ZONE_CONFIGS = modified_configs
    
    # Run scenario
    twin = SteelPlantDigitalTwin(seed=np.random.randint(0, 10000))
    df = twin.run_simulation(days=days, workers_per_shift=50, shifts_per_day=3)
    kpis = twin.calculate_kpis(df)
    
    # Restore original configs
    safety_twin_simulation.ZONE_CONFIGS = original_configs
    
    return kpis


def run_enhanced_whatif_scenario(equipment_load_adj, maintenance_delay_adj, 
                                  ppe_adj, fatigue_adj, training_adj,
                                  ventilation_adj, dust_reduction,
                                  seasonal_mode, shift_pattern, days=90):
    """
    Run enhanced what-if scenario with all adjustable parameters.
    
    Args:
        equipment_load_adj: Equipment load multiplier
        maintenance_delay_adj: Maintenance delay multiplier
        ppe_adj: PPE compliance multiplier
        fatigue_adj: Fatigue factor multiplier
        training_adj: Training hours multiplier
        ventilation_adj: Ventilation effectiveness multiplier
        dust_reduction: Dust concentration reduction factor
        seasonal_mode: Seasonal pattern mode
        shift_pattern: Shift pattern mode
        days: Number of days to simulate
    
    Returns:
        Dictionary with scenario KPIs
    """
    # Validate realistic parameters
    equipment_load_adj = np.clip(equipment_load_adj, 0.5, 1.5)
    maintenance_delay_adj = np.clip(maintenance_delay_adj, 0.5, 2.0)
    ppe_adj = np.clip(ppe_adj, 0.8, 1.2)
    fatigue_adj = np.clip(fatigue_adj, 0.7, 1.3)
    training_adj = np.clip(training_adj, 0.5, 2.0)
    ventilation_adj = np.clip(ventilation_adj, 0.7, 1.3)
    dust_reduction = np.clip(dust_reduction, 0.5, 1.0)
    
    # Create modified zone configs
    import copy
    modified_configs = copy.deepcopy(ZONE_CONFIGS)
    
    for zone in modified_configs:
        # Apply operational adjustments
        new_load = modified_configs[zone].equipment_load_mean * equipment_load_adj
        modified_configs[zone].equipment_load_mean = np.clip(new_load, 0.5, 0.95)
        
        new_delay = modified_configs[zone].maintenance_delay_mean * maintenance_delay_adj
        modified_configs[zone].maintenance_delay_mean = max(5.0, new_delay)
        
        # Apply human factor adjustments
        new_ppe = modified_configs[zone].ppe_compliance_rate * ppe_adj
        modified_configs[zone].ppe_compliance_rate = np.clip(new_ppe, 0.7, 1.0)
        
        # Note: Fatigue is adjusted per-shift in simulation, but we adjust baseline
        new_fatigue = modified_configs[zone].fatigue_factor_mean * fatigue_adj
        modified_configs[zone].fatigue_factor_mean = np.clip(new_fatigue, 0.2, 0.6)
        
        # Training hours
        modified_configs[zone].safety_training_hours *= training_adj
        
        # Apply environmental adjustments
        new_vent = modified_configs[zone].ventilation_effectiveness * ventilation_adj
        modified_configs[zone].ventilation_effectiveness = np.clip(new_vent, 0.6, 1.0)
        
        # Dust reduction
        modified_configs[zone].dust_concentration_mean *= dust_reduction
    
    # Temporarily replace configs
    from safety_twin_simulation import ZONE_CONFIGS as original_configs
    import safety_twin_simulation
    safety_twin_simulation.ZONE_CONFIGS = modified_configs
    
    # Run scenario (seasonal and shift patterns are handled in the simulation code)
    twin = SteelPlantDigitalTwin(seed=np.random.randint(0, 10000))
    df = twin.run_simulation(days=days, workers_per_shift=50, shifts_per_day=3)
    kpis = twin.calculate_kpis(df)
    
    # Restore original configs
    safety_twin_simulation.ZONE_CONFIGS = original_configs
    
    return kpis


def main():
    """Main dashboard application."""
    
    # Header
    st.markdown('<div class="main-header">Tata Steel Digital Safety Twin Dashboard</div>', 
                unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Select Page",
        ["Overview", "KPI Trends", "Zone Analysis", "What-If Scenarios", 
         "Benefit-Cost Analysis", "ℹAbout"]
    )
    
    # Load data
    df = load_simulation_data()
    kpis = load_kpi_data()
    
    if df is None:
        st.error("⚠️ No simulation data found. Please run the simulation first:")
        st.code("python safety_twin_simulation.py")
        st.stop()
    
    # ==================== OVERVIEW PAGE (MERGED WITH PLANT OVERVIEW) ====================
    if page == "Overview":
        # Executive Dashboard Layout
        st.markdown("""
        <div style='background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%); 
                    padding: 30px 40px; border-radius: 15px; margin-bottom: 30px;'>
            <h1 style='color: white; margin: 0; font-size: 2.5rem; text-align: center;'>
                TATA STEEL DIGITAL SAFETY TWIN
            </h1>
            <p style='color: #E8F0FE; margin: 10px 0 0 0; text-align: center; font-size: 1.1rem;'>
                Real-Time Plant Safety Monitoring Dashboard
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # Get current operational data
        last_day_data = df[df['day'] == df['day'].max()].iloc[0]
        
        # Key Safety Metrics Row - Executive Summary
        st.markdown("### Executive Safety Summary")
        exec_col1, exec_col2, exec_col3, exec_col4, exec_col5 = st.columns(5)
        
        with exec_col1:
            st.markdown(f"""
            <div style='background: #f0f8ff; padding: 20px; border-radius: 10px; border-left: 5px solid #1f77b4;'>
                <div style='font-size: 2rem; font-weight: bold; color: #1f77b4;'>{kpis['LTIFR']:.2f}</div>
                <div style='color: #666; font-size: 0.9rem;'>LTIFR</div>
                <div style='color: #999; font-size: 0.8rem;'>per million man-hours</div>
            </div>
            """, unsafe_allow_html=True)
        
        with exec_col2:
            st.markdown(f"""
            <div style='background: #fff5f5; padding: 20px; border-radius: 10px; border-left: 5px solid #d62728;'>
                <div style='font-size: 2rem; font-weight: bold; color: #d62728;'>{kpis['Total_LTI']:.0f}</div>
                <div style='color: #666; font-size: 0.9rem;'>Lost Time Injuries</div>
                <div style='color: #999; font-size: 0.8rem;'>Total incidents</div>
            </div>
            """, unsafe_allow_html=True)
        
        with exec_col3:
            st.markdown(f"""
            <div style='background: #f0fff0; padding: 20px; border-radius: 10px; border-left: 5px solid #2ca02c;'>
                <div style='font-size: 2rem; font-weight: bold; color: #2ca02c;'>{kpis['LTTR']:.0f}</div>
                <div style='color: #666; font-size: 0.9rem;'>Avg Lost Hours</div>
                <div style='color: #999; font-size: 0.8rem;'>per incident</div>
            </div>
            """, unsafe_allow_html=True)
        
        with exec_col4:
            st.markdown(f"""
            <div style='background: #fffbf0; padding: 20px; border-radius: 10px; border-left: 5px solid #ff7f0e;'>
                <div style='font-size: 2rem; font-weight: bold; color: #ff7f0e;'>{kpis['TRIR']:.2f}</div>
                <div style='color: #666; font-size: 0.9rem;'>TRIR</div>
                <div style='color: #999; font-size: 0.8rem;'>per million man-hours</div>
            </div>
            """, unsafe_allow_html=True)
        
        with exec_col5:
            st.markdown(f"""
            <div style='background: #f5f5ff; padding: 20px; border-radius: 10px; border-left: 5px solid #9467bd;'>
                <div style='font-size: 2rem; font-weight: bold; color: #9467bd;'>{int(kpis['Total_Man_Hours']):,}</div>
                <div style='color: #666; font-size: 0.9rem;'>Man-Hours</div>
                <div style='color: #999; font-size: 0.8rem;'>Total exposure</div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Operations Status - Full width
        col_status1, col_status2 = st.columns(2)
        
        with col_status1:
            st.markdown("### Operations Status")
            
            # Current conditions
            season_names = ["Summer", "Monsoon", "Post-Monsoon", "Winter"]
            season = int(last_day_data['season']) if 'season' in last_day_data else 0
            
            shift_names = {1: "Day Shift", 2: "Evening Shift", 3: "Night Shift"}
            shift_num = int(last_day_data['shift']) if 'shift' in last_day_data else 1
            
            st.markdown(f"""
            <div style='background: #2a2a2a; padding: 20px; border-radius: 10px; margin-bottom: 20px; border: 1px solid #444;'>
                <h4 style='color: #fff; margin-top: 0;'>Current Status</h4>
                <p style='color: #ccc;'><strong>Day:</strong> {int(last_day_data['day'])}</p>
                <p style='color: #ccc;'><strong>Season:</strong> {season_names[season]}</p>
                <p style='color: #ccc;'><strong>Shift:</strong> {shift_names.get(shift_num, "Day")}</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Key Parameters
            st.markdown(f"""
            <div style='background: #2a2a2a; padding: 20px; border-radius: 10px; border: 1px solid #444;'>
                <h4 style='color: #fff; margin-top: 0;'>Operational Parameters</h4>
                <p style='color: #ccc;'><strong>Equipment Load:</strong> {last_day_data['equipment_load']:.1%}</p>
                <p style='color: #ccc;'><strong>Avg Temperature:</strong> {last_day_data['temperature']:.0f}°C</p>
                <p style='color: #ccc;'><strong>PPE Compliance:</strong> {last_day_data['ppe_compliance_rate']:.1%}</p>
                <p style='color: #ccc;'><strong>Ventilation:</strong> {last_day_data['ventilation_effectiveness']:.1%}</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col_status2:
            st.markdown("### Zone Performance")
            # Zone comparison
            fig_zone = plot_zone_comparison(df)
            st.plotly_chart(fig_zone, use_container_width=True)
        
        # Severity distribution
        st.markdown("---")
        st.subheader("Incident Severity Distribution")
        fig_severity = plot_severity_histogram(df)
        st.plotly_chart(fig_severity, width='stretch')
    
    # ==================== KPI TRENDS PAGE ====================
    elif page == "KPI Trends":
        st.header("KPI Trend Analysis")
        
        st.info("Showing 30-day rolling averages for trend smoothing")
        
        # KPI trends
        fig_trends = plot_kpi_trends(df)
        st.plotly_chart(fig_trends, width='stretch')
        
        # Daily incident chart with enhanced visualization
        st.subheader("Daily Incident Occurrences")
        daily_incidents = df.groupby('day')['incident_occurred'].sum().reset_index()
        
        fig_daily = go.Figure()
        fig_daily.add_trace(go.Scatter(
            x=daily_incidents['day'],
            y=daily_incidents['incident_occurred'],
            mode='markers+lines',
            marker=dict(size=8, color='#d62728', line=dict(width=1, color='white')),
            line=dict(width=2, color='#d62728', shape='spline'),
            fill='tozeroy',
            fillcolor='rgba(214, 39, 40, 0.2)',
            name='Daily Incidents'
        ))
        
        # Add moving average
        window = 7
        daily_incidents['ma'] = daily_incidents['incident_occurred'].rolling(window=window, center=True).mean()
        fig_daily.add_trace(go.Scatter(
            x=daily_incidents['day'],
            y=daily_incidents['ma'],
            mode='lines',
            line=dict(width=3, color='#1f77b4', dash='dash'),
            name=f'{window}-Day Moving Average'
        ))
        
        fig_daily.update_layout(
            title="Daily Incident Count with Moving Average",
            xaxis_title="Simulation Day",
            yaxis_title="Number of Incidents",
            height=400,
            hovermode='x unified',
            template='plotly_white',
            showlegend=True
        )
        st.plotly_chart(fig_daily, width='stretch')
    
    # ==================== ZONE ANALYSIS PAGE ====================
    elif page == "Zone Analysis":
        st.header("Zone-Wise Safety Analysis")
        
        # Risk heatmap
        st.subheader("Risk Factor Heatmap")
        fig_heatmap = plot_zone_heatmap(df)
        st.plotly_chart(fig_heatmap, width='stretch')
        
        # Zone selector
        st.subheader("Detailed Zone Analysis")
        selected_zone = st.selectbox("Select Zone", df['zone'].unique())
        
        zone_df = df[df['zone'] == selected_zone]
        
        # Zone statistics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Incidents", int(zone_df['incident_occurred'].sum()))
        with col2:
            st.metric("Avg Incident Probability", f"{zone_df['incident_probability'].mean():.4f}")
        with col3:
            st.metric("Total Lost Hours", f"{zone_df['lost_hours'].sum():.1f}")
        
        # Parameter distribution plots
        st.subheader(f"{selected_zone} - Parameter Distributions")
        
        fig_params = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Temperature', 'Equipment Load', 
                          'Maintenance Delay', 'Fatigue Factor')
        )
        
        fig_params.add_trace(
            go.Histogram(x=zone_df['temperature'], marker_color='#ff7f0e', nbinsx=30),
            row=1, col=1
        )
        fig_params.add_trace(
            go.Histogram(x=zone_df['equipment_load'], marker_color='#2ca02c', nbinsx=30),
            row=1, col=2
        )
        fig_params.add_trace(
            go.Histogram(x=zone_df['maintenance_delay'], marker_color='#d62728', nbinsx=30),
            row=2, col=1
        )
        fig_params.add_trace(
            go.Histogram(x=zone_df['fatigue_factor'], marker_color='#9467bd', nbinsx=30),
            row=2, col=2
        )
        
        fig_params.update_layout(height=600, showlegend=False)
        st.plotly_chart(fig_params, width='stretch')
    
    # ==================== WHAT-IF SCENARIOS PAGE ====================
    elif page == "What-If Scenarios":
        st.header("What-If Scenario Modeling")
        
        st.markdown("""
        <div class="info-box">
        <strong>🎯 Enhanced Scenario Modeling</strong><br>
        Adjust operational, environmental, human factors, and seasonal patterns to see predicted impact on safety KPIs.
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Initialize all parameters with defaults
        equipment_adj = 1.0
        maintenance_adj = 1.0
        simulation_days = 90
        ppe_adj = 1.0
        fatigue_adj = 1.0
        training_adj = 1.0
        ventilation_adj = 1.0
        dust_reduction = 1.0
        seasonal_mode = "Normal"
        shift_pattern = "Normal"
        
        # Create tabs for different scenario types
        tab1, tab2, tab3, tab4 = st.tabs(["🔧 Operational", "👥 Human Factors", "🌡️ Environmental", "📅 Seasonal Patterns"])
        
        with tab1:
            st.subheader("Operational Parameters")
            col1, col2 = st.columns(2)
            
            with col1:
                equipment_adj = st.slider(
                    "Equipment Load Adjustment",
                    min_value=0.5,
                    max_value=1.5,
                    value=1.0,
                    step=0.05,
                    help="1.0 = baseline, >1.0 = increased load, <1.0 = reduced load"
                )
            
            with col2:
                maintenance_adj = st.slider(
                    "Maintenance Delay Adjustment",
                    min_value=0.5,
                    max_value=2.0,
                    value=1.0,
                    step=0.1,
                    help="1.0 = baseline, >1.0 = longer delays, <1.0 = shorter delays"
                )
            
            simulation_days = st.slider(
                "Simulation Duration (days)",
                min_value=30,
                max_value=180,
                value=90,
                step=30
            )
        
        with tab2:
            st.subheader("Human Factors")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                ppe_adj = st.slider(
                    "PPE Compliance Adjustment",
                    min_value=0.8,
                    max_value=1.2,
                    value=1.0,
                    step=0.05,
                    help="1.0 = baseline, >1.0 = improved compliance, <1.0 = reduced compliance"
                )
            
            with col2:
                fatigue_adj = st.slider(
                    "Fatigue Factor Adjustment",
                    min_value=0.7,
                    max_value=1.3,
                    value=1.0,
                    step=0.05,
                    help="1.0 = baseline, <1.0 = reduced fatigue, >1.0 = increased fatigue"
                )
            
            with col3:
                training_adj = st.slider(
                    "Safety Training Hours Adjustment",
                    min_value=0.5,
                    max_value=2.0,
                    value=1.0,
                    step=0.1,
                    help="1.0 = baseline, >1.0 = more training, <1.0 = less training"
                )
            
            st.info("💡 Human factors have the strongest impact on safety incidents")
        
        with tab3:
            st.subheader("Environmental Conditions")
            col1, col2 = st.columns(2)
            
            with col1:
                ventilation_adj = st.slider(
                    "Ventilation Effectiveness Adjustment",
                    min_value=0.7,
                    max_value=1.3,
                    value=1.0,
                    step=0.05,
                    help="1.0 = baseline, >1.0 = improved ventilation, <1.0 = reduced ventilation"
                )
            
            with col2:
                dust_reduction = st.slider(
                    "Dust Concentration Reduction",
                    min_value=0.5,
                    max_value=1.0,
                    value=1.0,
                    step=0.05,
                    help="1.0 = baseline (no reduction), <1.0 = reduced dust (e.g., 0.7 = 30% reduction)"
                )
            
            st.info("🌬️ Better ventilation and dust control can significantly reduce environmental risk")
        
        with tab4:
            st.subheader("Seasonal and Shift Patterns")
            col1, col2 = st.columns(2)
            
            with col1:
                seasonal_mode = st.selectbox(
                    "Seasonal Pattern",
                    ["Normal", "Summer Focus", "Monsoon Focus", "Winter Optimized"],
                    help="Normal = baseline patterns, Others = adjusted patterns"
                )
            
            with col2:
                shift_pattern = st.selectbox(
                    "Shift Pattern",
                    ["Normal", "Day Shift Heavy", "Night Shift Heavy", "Balanced"],
                    help="Normal = baseline patterns, Others = adjusted shift distributions"
                )
            
            st.info("Seasonal and shift patterns affect fatigue and compliance rates")
        
        # Summary of all parameters
        st.markdown("---")
        st.subheader("Scenario Parameters Summary")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Equipment Load", f"{equipment_adj:.2f}×")
            st.metric("Maintenance Delay", f"{maintenance_adj:.2f}×")
        with col2:
            st.metric("PPE Compliance", f"{ppe_adj:.2f}×")
            st.metric("Fatigue Factor", f"{fatigue_adj:.2f}×")
        with col3:
            st.metric("Ventilation", f"{ventilation_adj:.2f}×")
            st.metric("Dust Reduction", f"{dust_reduction:.2f}×")
        with col4:
            st.metric("Seasonal Mode", seasonal_mode)
            st.metric("Shift Pattern", shift_pattern)
        
        if st.button("Run Enhanced Scenario", type="primary"):
            with st.spinner("Running enhanced scenario simulation..."):
                scenario_kpis = run_enhanced_whatif_scenario(
                    equipment_adj, 
                    maintenance_adj,
                    ppe_adj,
                    fatigue_adj,
                    training_adj,
                    ventilation_adj,
                    dust_reduction,
                    seasonal_mode,
                    shift_pattern,
                    days=simulation_days
                )
            
            st.success("Scenario completed!")
            
            # Comparison table
            st.subheader("KPI Comparison: Baseline vs. Scenario")
            
            comparison_df = pd.DataFrame({
                'KPI': ['LTIFR', 'LTTR', 'Severity Rate', 'TRIR'],
                'Baseline': [kpis['LTIFR'], kpis['LTTR'], 
                           kpis['Severity_Rate'], kpis['TRIR']],
                'Scenario': [scenario_kpis['LTIFR'], scenario_kpis['LTTR'],
                           scenario_kpis['Severity_Rate'], scenario_kpis['TRIR']]
            })
            
            comparison_df['Change (%)'] = (
                (comparison_df['Scenario'] - comparison_df['Baseline']) / 
                comparison_df['Baseline'] * 100
            )
            
            # Style the dataframe
            def color_change(val):
                color = 'red' if val > 0 else 'green' if val < 0 else 'black'
                return f'color: {color}'
            
            styled_df = comparison_df.style.applymap(
                color_change, subset=['Change (%)']
            ).format({
                'Baseline': '{:.2f}',
                'Scenario': '{:.2f}',
                'Change (%)': '{:+.1f}%'
            })
            
            st.dataframe(styled_df, width='stretch')
            
            # Visualization
            fig_comparison = go.Figure()
            
            x_labels = comparison_df['KPI'].tolist()
            
            fig_comparison.add_trace(go.Bar(
                name='Baseline',
                x=x_labels,
                y=comparison_df['Baseline'],
                marker_color='#1f77b4'
            ))
            
            fig_comparison.add_trace(go.Bar(
                name='Scenario',
                x=x_labels,
                y=comparison_df['Scenario'],
                marker_color='#ff7f0e'
            ))
            
            fig_comparison.update_layout(
                title="KPI Comparison",
                yaxis_title="Value",
                barmode='group',
                height=400
            )
            
            st.plotly_chart(fig_comparison, width='stretch')
    
    # ==================== BENEFIT-COST ANALYSIS PAGE ====================
    elif page == "Benefit-Cost Analysis":
        st.header("Benefit-Cost Analysis for Safety Equipment")
        
        st.markdown("""
        <div class="info-box">
        <strong>Value of Statistical Life (VoSL)</strong><br>
        This analysis uses VoSL estimates for India (~45 lakhs INR) to evaluate safety equipment investments.
        Compare costs, benefits, ROI, and NPV to make informed decisions. Use the comparison tool to evaluate multiple equipment options across different zones.
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Create tabs for single vs. multiple equipment comparison
        bca_tab1, bca_tab2 = st.tabs(["Single Equipment Analysis", "Multi-Equipment Comparison"])
        
        # Initialize analyzer (shared across tabs)
        col1, col2 = st.columns(2)
        with col1:
            vosl_lakh = st.number_input(
                "Value of Statistical Life (VoSL) in Lakhs INR",
                min_value=30.0,
                max_value=80.0,
                value=45.0,
                step=1.0,
                help="VoSL estimate for India (typically 40-50 lakhs INR)"
            )
        with col2:
            discount_rate = st.number_input(
                "Discount Rate (%)",
                min_value=5.0,
                max_value=12.0,
                value=8.0,
                step=0.5,
                help="Cost of capital / discount rate"
            )
        
        analyzer = BenefitCostAnalyzer(vosl_india_lakh=vosl_lakh)
        analyzer.discount_rate = discount_rate / 100
        
        # Calculate baseline costs (shared)
        baseline_costs = analyzer.calculate_incident_costs(df)
        
        # Tab 1: Single Equipment Analysis
        with bca_tab1:
            st.subheader("Current Incident Costs (Annual)")
            
            cost_cols = st.columns(4)
            with cost_cols[0]:
                st.metric("Total Cost", f"₹{baseline_costs['total_cost_lakh']:.2f} L")
            with cost_cols[1]:
                st.metric("Direct Costs", f"₹{baseline_costs['direct_costs_lakh']:.2f} L")
            with cost_cols[2]:
                st.metric("Indirect Costs", f"₹{baseline_costs['indirect_costs_lakh']:.2f} L")
            with cost_cols[3]:
                st.metric("Lost Productivity", f"₹{baseline_costs['lost_productivity_lakh']:.2f} L")
            
            if baseline_costs['estimated_fatalities'] > 0:
                st.warning(f"⚠️ Estimated Fatalities: {baseline_costs['estimated_fatalities']} "
                          f"(Cost: ₹{baseline_costs['fatal_costs_lakh']:.2f} L)")
            
            st.markdown("---")
            st.subheader("🔧 Safety Equipment Investment Analysis")
            
            # Equipment parameters
            col1, col2, col3 = st.columns(3)
            
            with col1:
                equipment_name = st.text_input(
                    "Equipment Name",
                    value="Automated Safety Monitoring System",
                    help="Name of the safety equipment/intervention",
                    key="single_equip_name"
                )
                equipment_cost = st.number_input(
                    "Initial Cost (Lakhs INR)",
                    min_value=10.0,
                    max_value=1000.0,
                    value=150.0,
                    step=10.0,
                    key="single_equip_cost"
                )
                annual_maintenance = st.number_input(
                    "Annual Maintenance Cost (Lakhs INR)",
                    min_value=1.0,
                    max_value=100.0,
                    value=15.0,
                    step=1.0,
                    key="single_equip_maint"
                )
            
            with col2:
                equipment_life = st.number_input(
                    "Equipment Lifetime (Years)",
                    min_value=5,
                    max_value=20,
                    value=10,
                    step=1,
                    key="single_equip_life"
                )
                risk_reduction = st.slider(
                    "Expected Risk Reduction (%)",
                    min_value=10,
                    max_value=50,
                    value=25,
                    step=5,
                    help="Percentage reduction in incidents",
                    key="single_equip_risk"
                ) / 100
            
            with col3:
                st.info(f"""
                **Investment Summary:**
                - Initial Cost: ₹{equipment_cost:.1f} L
                - Annual Maintenance: ₹{annual_maintenance:.1f} L
                - Lifetime: {equipment_life} years
                - Risk Reduction: {risk_reduction*100:.0f}%
                """)
            
            if st.button("Analyze Investment", type="primary", key="single_analyze"):
                with st.spinner("Running benefit-cost analysis..."):
                    # Create improved scenario (with risk reduction)
                    # Simulate improved conditions
                    improved_params = df.copy()
                    improved_params['incident_probability'] = improved_params['incident_probability'] * (1 - risk_reduction)
                    improved_params['incident_occurred'] = np.random.random(len(improved_params)) < improved_params['incident_probability']
                    
                    # Recalculate lost hours for incidents that still occur
                    incident_mask = improved_params['incident_occurred']
                    improved_params.loc[~incident_mask, 'lost_hours'] = 0
                    improved_params.loc[~incident_mask, 'severity'] = 'None'
                    
                    # Analyze
                    bca_results = analyzer.analyze_safety_equipment(
                        df,
                        improved_params,
                        equipment_cost,
                        annual_maintenance,
                        equipment_life,
                        risk_reduction
                    )
                    
                    st.success("Analysis completed!")
                    
                    # Display results
                    st.subheader("Investment Analysis Results")
                    
                    result_cols = st.columns(3)
                    
                    with result_cols[0]:
                        st.markdown('<div class="kpi-card">', unsafe_allow_html=True)
                        npv_color = "green" if bca_results['npv_lakh'] > 0 else "red"
                        st.metric(
                            "Net Present Value (NPV)",
                            f"₹{bca_results['npv_lakh']:.2f} L",
                            delta=f"{bca_results['npv_lakh']/equipment_cost*100:.1f}% ROI"
                        )
                        st.markdown('</div>', unsafe_allow_html=True)
                    
                    with result_cols[1]:
                        st.markdown('<div class="kpi-card">', unsafe_allow_html=True)
                        st.metric(
                            "Benefit-Cost Ratio",
                            f"{bca_results['benefit_cost_ratio']:.2f}",
                            help=">1.0 indicates viable investment"
                        )
                        st.markdown('</div>', unsafe_allow_html=True)
                    
                    with result_cols[2]:
                        st.markdown('<div class="kpi-card">', unsafe_allow_html=True)
                        payback = bca_results['payback_period_years'] if bca_results['payback_period_years'] else "N/A"
                        st.metric(
                            "Payback Period",
                            f"{payback} years" if payback != "N/A" else "N/A",
                            help="Time to recover initial investment"
                        )
                        st.markdown('</div>', unsafe_allow_html=True)
                    
                    # Detailed metrics
                    st.subheader("Detailed Financial Metrics")
                    
                    metrics_df = pd.DataFrame({
                        'Metric': [
                            'Initial Investment',
                            'Annual Savings',
                            'Total Costs (PV)',
                            'Total Benefits (PV)',
                            'NPV',
                            'BCR',
                            'ROI (%)',
                            'Payback Period (years)'
                        ],
                        'Value (Lakhs INR)': [
                            f"₹{bca_results['equipment_cost_lakh']:.2f}",
                            f"₹{bca_results['annual_savings_lakh']:.2f}",
                            f"₹{bca_results['total_costs_pv_lakh']:.2f}",
                            f"₹{bca_results['total_benefits_pv_lakh']:.2f}",
                            f"₹{bca_results['npv_lakh']:.2f}",
                            f"{bca_results['benefit_cost_ratio']:.2f}",
                            f"{bca_results['roi_pct']:.2f}%",
                            f"{bca_results['payback_period_years'] if bca_results['payback_period_years'] else 'N/A'}"
                        ]
                    })
                    st.dataframe(metrics_df, width='stretch', hide_index=True)
                    
                    # Visualization
                    st.subheader("Cost-Benefit Visualization")
                    
                    fig_bca = go.Figure()
                    
                    categories = ['Total Costs', 'Total Benefits', 'NPV']
                    values = [
                        bca_results['total_costs_pv_lakh'],
                        bca_results['total_benefits_pv_lakh'],
                        bca_results['npv_lakh']
                    ]
                    colors = ['#d62728', '#2ca02c', '#1f77b4']
                    
                    fig_bca.add_trace(go.Bar(
                        x=categories,
                        y=values,
                        marker_color=colors,
                        text=[f"₹{v:.1f}L" for v in values],
                        textposition='auto',
                    ))
                    
                    fig_bca.update_layout(
                        title="Cost-Benefit Analysis Comparison",
                        xaxis_title="Category",
                        yaxis_title="Value (Lakhs INR)",
                        height=400,
                        showlegend=False
                    )
                    
                    st.plotly_chart(fig_bca, width='stretch')
                    
                    # Yearly cash flow
                    st.subheader("💵 Annual Cash Flow Projection")
                    
                    years = list(range(1, equipment_life + 1))
                    annual_benefits = [bca_results['annual_savings_lakh'] / ((1 + analyzer.discount_rate) ** y) 
                                     for y in years]
                    annual_costs = [annual_maintenance / ((1 + analyzer.discount_rate) ** y) 
                                   for y in years]
                    
                    # Create cumulative cash flow (starting from year 0)
                    cumulative = []
                    running_total = -equipment_cost  # Year 0: initial investment
                    cumulative.append(running_total)
                    for year in range(1, equipment_life + 1):
                        running_total += annual_benefits[year-1] - annual_costs[year-1]
                        cumulative.append(running_total)
                    
                    fig_cashflow = go.Figure()
                    
                    fig_cashflow.add_trace(go.Scatter(
                        x=list(range(0, equipment_life + 1)),
                        y=cumulative,
                        mode='lines+markers',
                        name='Cumulative NPV',
                        line=dict(color='#1f77b4', width=3),
                        marker=dict(size=8)
                    ))
                    
                    fig_cashflow.add_hline(y=0, line_dash="dash", line_color="gray",
                                          annotation_text="Break-even")
                    
                    fig_cashflow.update_layout(
                        title="Cumulative Net Present Value Over Time",
                        xaxis_title="Year",
                        yaxis_title="Cumulative NPV (Lakhs INR)",
                        height=400,
                        showlegend=True
                    )
                    
                    st.plotly_chart(fig_cashflow, width='stretch')
                    
                    # Investment recommendation
                    st.markdown("---")
                    if bca_results['is_viable']:
                        st.markdown(f"""
                        <div class="success-box">
                        <h3>Investment Recommendation: APPROVED</h3>
                        <p>The investment in <strong>{equipment_name}</strong> is financially viable.</p>
                        <ul>
                            <li>NPV: ₹{bca_results['npv_lakh']:.2f} Lakhs (positive)</li>
                            <li>BCR: {bca_results['benefit_cost_ratio']:.2f} (>1.0)</li>
                            <li>ROI: {bca_results['roi_pct']:.2f}%</li>
                            <li>Payback: {bca_results['payback_period_years']} years</li>
                        </ul>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div class="warning-box">
                        <h3>⚠️ Investment Recommendation: REVIEW REQUIRED</h3>
                        <p>The investment in <strong>{equipment_name}</strong> may not be financially viable at current parameters.</p>
                        <ul>
                            <li>NPV: ₹{bca_results['npv_lakh']:.2f} Lakhs</li>
                            <li>BCR: {bca_results['benefit_cost_ratio']:.2f}</li>
                            <li>Consider: Reducing cost, increasing risk reduction, or extending lifetime</li>
                        </ul>
                        </div>
                        """, unsafe_allow_html=True)
        
        # Tab 2: Multi-Equipment Comparison
        with bca_tab2:
            st.subheader("Multi-Equipment Investment Comparison")
            st.markdown("""
            <div class="info-box">
            <strong>Cost-Effectiveness Analysis</strong><br>
            Compare multiple safety equipment investments across different zones. Evaluate which investments provide the best value using NPV, BCR, ROI, and cost-effectiveness metrics.
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            # Number of equipment options to compare
            num_equipment = st.slider(
                "Number of Equipment Options to Compare",
                min_value=2,
                max_value=5,
                value=3,
                step=1,
                help="Compare 2-5 different equipment investments"
            )
            
            # Collect equipment details
            equipment_list = []
            zones_list = df['zone'].unique().tolist()
            
            st.subheader("🔧 Equipment Configuration")
            
            for i in range(num_equipment):
                with st.expander(f"Equipment Option {i+1}", expanded=(i < 2)):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        equip_name = st.text_input(
                            f"Equipment Name {i+1}",
                            value=f"Safety Equipment Option {i+1}",
                            key=f"equip_name_{i}"
                        )
                        equip_zone = st.selectbox(
                            f"Target Zone {i+1}",
                            zones_list,
                            key=f"equip_zone_{i}",
                            help="Select the zone where this equipment will be installed"
                        )
                        equip_cost = st.number_input(
                            f"Initial Cost (Lakhs INR) {i+1}",
                            min_value=10.0,
                            max_value=1000.0,
                            value=150.0 + i * 50,
                            step=10.0,
                            key=f"equip_cost_{i}"
                        )
                    
                    with col2:
                        equip_maintenance = st.number_input(
                            f"Annual Maintenance (Lakhs INR) {i+1}",
                            min_value=1.0,
                            max_value=100.0,
                            value=15.0 + i * 2,
                            step=1.0,
                            key=f"equip_maint_{i}"
                        )
                        equip_life = st.number_input(
                            f"Equipment Lifetime (Years) {i+1}",
                            min_value=5,
                            max_value=20,
                            value=10,
                            step=1,
                            key=f"equip_life_{i}"
                        )
                        equip_risk_reduction = st.slider(
                            f"Expected Risk Reduction (%) {i+1}",
                            min_value=10,
                            max_value=50,
                            value=25 + i * 5,
                            step=5,
                            key=f"equip_risk_{i}",
                            help="Expected reduction in incidents for this zone"
                        ) / 100
                    
                    equipment_list.append({
                        'name': equip_name,
                        'zone': equip_zone,
                        'cost': equip_cost,
                        'maintenance': equip_maintenance,
                        'life': equip_life,
                        'risk_reduction': equip_risk_reduction
                    })
            
            if st.button("Compare Equipment Options", type="primary"):
                with st.spinner("Analyzing all equipment options..."):
                    interventions = []
                    
                    for equip in equipment_list:
                        # Get zone-specific baseline data
                        zone_baseline_df = df[df['zone'] == equip['zone']].copy()
                        
                        if len(zone_baseline_df) == 0:
                            st.warning(f"No data found for zone {equip['zone']}")
                            continue
                        
                        # Create improved scenario for this zone
                        improved_df = zone_baseline_df.copy()
                        improved_df['incident_probability'] = improved_df['incident_probability'] * (1 - equip['risk_reduction'])
                        improved_df['incident_occurred'] = np.random.random(len(improved_df)) < improved_df['incident_probability']
                        
                        # Recalculate lost hours
                        incident_mask = improved_df['incident_occurred']
                        improved_df.loc[~incident_mask, 'lost_hours'] = 0
                        improved_df.loc[~incident_mask, 'severity'] = 'None'
                        
                        interventions.append({
                            'name': f"{equip['name']} - {equip['zone']}",
                            'baseline_df': zone_baseline_df,
                            'improved_df': improved_df,
                            'equipment_cost_lakh': equip['cost'],
                            'annual_maintenance_lakh': equip['maintenance'],
                            'equipment_life_years': equip['life'],
                            'risk_reduction_pct': equip['risk_reduction']
                        })
                    
                    if len(interventions) > 0:
                        # Compare interventions
                        comparison_df = analyzer.compare_interventions(interventions)
                        
                        # Calculate cost-effectiveness metrics
                        comparison_df['cost_per_risk_reduction'] = comparison_df['equipment_cost_lakh'] / (comparison_df['risk_reduction_pct'] / 100)
                        comparison_df['npv_per_cost'] = comparison_df['npv_lakh'] / comparison_df['equipment_cost_lakh']
                        comparison_df['cost_effectiveness_ratio'] = comparison_df['benefit_cost_ratio'] / comparison_df['equipment_cost_lakh'] * 100
                        
                        st.success(f"Analysis completed for {len(interventions)} equipment options!")
                        
                        # Display results
                        st.subheader("Equipment Comparison Results")
                        
                        # Sort by NPV (best first)
                        comparison_df = comparison_df.sort_values('npv_lakh', ascending=False)
                        
                        # Key metrics comparison
                        col1, col2, col3, col4 = st.columns(4)
                        
                        with col1:
                            best_npv = comparison_df.iloc[0]
                            st.metric(
                                "Best NPV",
                                f"₹{best_npv['npv_lakh']:.2f} L",
                                delta=best_npv['intervention_name']
                            )
                        
                        with col2:
                            best_bcr = comparison_df.loc[comparison_df['benefit_cost_ratio'].idxmax()]
                            st.metric(
                                "Best BCR",
                                f"{best_bcr['benefit_cost_ratio']:.2f}",
                                delta=best_bcr['intervention_name']
                            )
                        
                        with col3:
                            best_roi = comparison_df.loc[comparison_df['roi_pct'].idxmax()]
                            st.metric(
                                "Best ROI",
                                f"{best_roi['roi_pct']:.2f}%",
                                delta=best_roi['intervention_name']
                            )
                        
                        with col4:
                            most_cost_effective = comparison_df.loc[comparison_df['npv_per_cost'].idxmax()]
                            st.metric(
                                "Most Cost-Effective",
                                f"{most_cost_effective['npv_per_cost']:.2f}×",
                                delta=most_cost_effective['intervention_name']
                            )
                        
                        # Detailed comparison table
                        st.subheader("Detailed Comparison Table")
                        
                        display_df = comparison_df[[
                            'intervention_name',
                            'equipment_cost_lakh',
                            'annual_savings_lakh',
                            'npv_lakh',
                            'benefit_cost_ratio',
                            'roi_pct',
                            'payback_period_years',
                            'cost_per_risk_reduction',
                            'npv_per_cost',
                            'is_viable'
                        ]].copy()
                        
                        display_df.columns = [
                            'Equipment',
                            'Initial Cost (L)',
                            'Annual Savings (L)',
                            'NPV (L)',
                            'BCR',
                            'ROI (%)',
                            'Payback (Years)',
                            'Cost/Risk Reduction',
                            'NPV/Cost Ratio',
                            'Viable'
                        ]
                        
                        # Format the dataframe
                        styled_df = display_df.style.format({
                            'Initial Cost (L)': '₹{:.2f}',
                            'Annual Savings (L)': '₹{:.2f}',
                            'NPV (L)': '₹{:.2f}',
                            'BCR': '{:.2f}',
                            'ROI (%)': '{:.2f}%',
                            'Payback (Years)': lambda x: f"{x:.1f}" if pd.notna(x) else "N/A",
                            'Cost/Risk Reduction': '₹{:.2f}',
                            'NPV/Cost Ratio': '{:.2f}×',
                            'Viable': lambda x: 'Yes' if x else 'No'
                        }).background_gradient(
                            subset=['NPV (L)', 'BCR', 'ROI (%)'],
                            cmap='RdYlGn'
                        )
                        
                        st.dataframe(styled_df, width='stretch', height=400)
                        
                        # Visualizations
                        st.subheader("Comparison Visualizations")
                        
                        # NPV Comparison
                        fig_npv = go.Figure()
                        fig_npv.add_trace(go.Bar(
                            x=comparison_df['intervention_name'],
                            y=comparison_df['npv_lakh'],
                            marker_color=['#2ca02c' if v > 0 else '#d62728' for v in comparison_df['npv_lakh']],
                            text=[f"₹{v:.1f}L" for v in comparison_df['npv_lakh']],
                            textposition='auto',
                            name='NPV'
                        ))
                        fig_npv.update_layout(
                            title="Net Present Value Comparison",
                            xaxis_title="Equipment",
                            yaxis_title="NPV (Lakhs INR)",
                            height=400,
                            xaxis_tickangle=-45
                        )
                        st.plotly_chart(fig_npv, width='stretch')
                        
                        # BCR Comparison
                        fig_bcr = go.Figure()
                        fig_bcr.add_trace(go.Bar(
                            x=comparison_df['intervention_name'],
                            y=comparison_df['benefit_cost_ratio'],
                            marker_color=['#2ca02c' if v > 1.0 else '#d62728' for v in comparison_df['benefit_cost_ratio']],
                            text=[f"{v:.2f}" for v in comparison_df['benefit_cost_ratio']],
                            textposition='auto',
                            name='BCR'
                        ))
                        fig_bcr.add_hline(y=1.0, line_dash="dash", line_color="gray",
                                         annotation_text="Break-even (BCR=1.0)")
                        fig_bcr.update_layout(
                            title="Benefit-Cost Ratio Comparison",
                            xaxis_title="Equipment",
                            yaxis_title="Benefit-Cost Ratio",
                            height=400,
                            xaxis_tickangle=-45
                        )
                        st.plotly_chart(fig_bcr, width='stretch')
                        
                        # Cost-Effectiveness Scatter Plot
                        st.subheader("💡 Cost-Effectiveness Analysis")
                        fig_ce = go.Figure()
                        fig_ce.add_trace(go.Scatter(
                            x=comparison_df['equipment_cost_lakh'],
                            y=comparison_df['npv_lakh'],
                            mode='markers+text',
                            marker=dict(
                                size=comparison_df['roi_pct'] * 2,
                                color=comparison_df['roi_pct'],
                                colorscale='RdYlGn',
                                showscale=True,
                                colorbar=dict(title="ROI (%)")
                            ),
                            text=comparison_df['intervention_name'],
                            textposition='top center',
                            name='Equipment Options'
                        ))
                        fig_ce.add_hline(y=0, line_dash="dash", line_color="gray")
                        fig_ce.update_layout(
                            title="Cost-Effectiveness: Investment Cost vs. NPV",
                            xaxis_title="Initial Investment Cost (Lakhs INR)",
                            yaxis_title="Net Present Value (Lakhs INR)",
                            height=500
                        )
                        st.plotly_chart(fig_ce, width='stretch')
                        
                        # Investment Recommendation
                        st.markdown("---")
                        best_option = comparison_df.iloc[0]
                        if best_option['is_viable']:
                            st.markdown(f"""
                            <div class="success-box">
                            <h3>Recommended Investment: {best_option['intervention_name']}</h3>
                            <p>This equipment option provides the <strong>highest NPV</strong> and is financially viable.</p>
                            <ul>
                                <li>NPV: ₹{best_option['npv_lakh']:.2f} Lakhs</li>
                                <li>BCR: {best_option['benefit_cost_ratio']:.2f}</li>
                                <li>ROI: {best_option['roi_pct']:.2f}%</li>
                                <li>Payback: {best_option['payback_period_years'] if pd.notna(best_option['payback_period_years']) else 'N/A'} years</li>
                            </ul>
                            </div>
                            """, unsafe_allow_html=True)
                        else:
                            st.markdown(f"""
                            <div class="warning-box">
                            <h3>⚠️ Investment Review Required</h3>
                            <p>While {best_option['intervention_name']} has the highest NPV, consider reviewing parameters for better financial viability.</p>
                            <p><strong>Consider:</strong> Reducing cost, increasing risk reduction, or extending equipment lifetime</p>
                            </div>
                            """, unsafe_allow_html=True)
                    else:
                        st.error("No valid equipment configurations found. Please check your inputs.")
    
    # ==================== ABOUT PAGE ====================
    elif page == "ℹAbout":
        st.header("About the Tata Steel Digital Safety Twin")
        
        st.markdown("""
        ### Overview
        
        This digital safety twin simulates a **Tata Steel plant's** production line and safety performance.
        It models operational parameters, human factors, and environmental conditions to predict
        safety incidents and compute industry-standard KPIs. Enhanced with realistic Tata Steel
        plant parameters and comprehensive benefit-cost analysis capabilities.
        
        ### Plant Configuration
        
        **Production Flow:**
        - Coke Oven → Blast Furnace
        - Sinter Plant → Blast Furnace
        - Blast Furnace → Basic Oxygen Furnace (BOF)
        - BOF → Ladle Metallurgy Furnace → Continuous Caster → Rolling Mill
        - Captive Power Plant (auxiliary support)
        
        **Simulated Parameters:**
        - Process: Temperature, Pressure
        - Operational: Equipment Load, Maintenance Delay
        - Human: Exposure, Fatigue Factor, PPE Compliance
        - Environmental: Noise, Dust, Ventilation
        
        ### Incident Model
        
        Incidents are modeled using a logistic regression approach:
        
        P(incident) = logistic(β₀ + Σ βᵢ × xᵢ)
        
        Where factors include equipment load, human exposure, fatigue, maintenance delays,
        and environmental conditions.
        
        ### Safety KPIs
        
        - **LTIFR**: Lost-time injuries per 1,000,000 man-hours
        - **LTTR**: Average lost hours per lost-time injury (hours per LTI)
        - **Severity Rate**: Lost workdays per 1,000 man-hours
        - **TRIR**: Recordable incidents per 1,000,000 man-hours
        
        ### Benefit-Cost Analysis
        
        The system includes comprehensive benefit-cost analysis using:
        - **VoSL (Value of Statistical Life)**: ~45 lakhs INR for India
        - **NPV (Net Present Value)**: Lifecycle financial assessment
        - **BCR (Benefit-Cost Ratio)**: Investment viability metric
        - **ROI (Return on Investment)**: Annual return percentage
        - **Payback Period**: Time to recover initial investment
        
        ### Literature References
        
        1. **Towards Safer Steel Operations** (Sci Rep 2025) – Risk modeling methodology
        2. **Hybrid Digital Twin for Process Industry** (2021) – Process simulation framework
        3. **Digital Twin-Based Safety Risk Coupling** (2021) – Human-equipment-environment interaction
        4. **Digital Twin of Hot Metal Ladle System** (2024) – Steel process parameters
        
        ### Technical Stack
        
        - **Simulation**: NumPy, SciPy, NetworkX, Pandas
        - **Visualization**: Plotly, Streamlit
        - **Modeling**: Logistic regression, Lognormal distributions
        
        ### Features
        
        - **Enhanced Simulation**: 8 production zones with realistic Tata Steel parameters
        - **Seasonal Effects**: Summer, Monsoon, Post-Monsoon, Winter patterns
        - **Shift Analysis**: Day, Evening, and Night shift modeling
        - **Benefit-Cost Analysis**: ROI, NPV, and VoSL-based evaluation
        - **Interactive Visualizations**: Animated charts and real-time updates
        - **What-If Scenarios**: Test operational changes and their impact
        
        ### Usage
        
        1. Run simulation: `python safety_twin_simulation.py`
        2. Launch dashboard: `streamlit run app.py`
        3. Explore KPIs, trends, run what-if scenarios, and analyze safety investments
        
        ---
        
        **Version**: 2.0  
        **Plant**: Tata Steel  
        **Last Updated**: January 2025
        """)
        
        # Display raw data option
        if st.checkbox("Show Raw Simulation Data"):
            st.subheader("Raw Data Preview")
            st.dataframe(df.head(100), width='stretch')
            
            if st.button("Download Full Dataset"):
                csv = df.to_csv(index=False)
                st.download_button(
                    label="📥 Download CSV",
                    data=csv,
                    file_name="safety_simulation_data.csv",
                    mime="text/csv"
                )


if __name__ == '__main__':
    main()

