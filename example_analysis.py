"""
Example Analysis Script
=======================

Demonstrates how to use the Digital Safety Twin API for custom analysis.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from safety_twin_simulation import SteelPlantDigitalTwin

def custom_analysis_example():
    """
    Example: Custom analysis using the digital twin.
    """
    print("="*60)
    print("CUSTOM ANALYSIS EXAMPLE")
    print("="*60)
    
    # Initialize and run simulation
    print("\n1. Running simulation...")
    twin = SteelPlantDigitalTwin(seed=123)
    df = twin.run_simulation(days=180, workers_per_shift=60, shifts_per_day=3)
    
    # Calculate KPIs
    print("\n2. Calculating KPIs...")
    overall_kpis = twin.calculate_kpis(df)
    zone_kpis = twin.calculate_zone_kpis(df)
    
    # Print results
    print("\n" + "="*60)
    print("OVERALL KPIS")
    print("="*60)
    for key, value in overall_kpis.items():
        if 'Rate' in key or 'LTIFR' in key or 'TRIR' in key or 'LTTR' in key:
            print(f"{key:.<30} {value:.2f}")
        else:
            print(f"{key:.<30} {value:.0f}")
    
    print("\n" + "="*60)
    print("ZONE-WISE KPIS")
    print("="*60)
    print(zone_kpis[['zone', 'LTIFR', 'LTTR', 'Severity_Rate', 'Total_LTI']].to_string(index=False))
    
    # Custom analysis: Correlation between maintenance delay and incidents
    print("\n" + "="*60)
    print("CUSTOM ANALYSIS: Maintenance Impact")
    print("="*60)
    
    # Group by maintenance delay bins
    df['maint_bin'] = pd.cut(df['maintenance_delay'], 
                             bins=[0, 5, 10, 15, 20, 100],
                             labels=['0-5', '5-10', '10-15', '15-20', '20+'])
    
    maintenance_analysis = df.groupby('maint_bin').agg({
        'incident_occurred': 'sum',
        'man_hours': 'sum'
    })
    
    maintenance_analysis['incident_rate'] = (
        maintenance_analysis['incident_occurred'] * 1e6 / 
        maintenance_analysis['man_hours']
    )
    
    print("\nIncident Rate by Maintenance Delay (days since last maintenance):")
    print(maintenance_analysis[['incident_occurred', 'incident_rate']].to_string())
    
    # Custom analysis: Fatigue impact
    print("\n" + "="*60)
    print("CUSTOM ANALYSIS: Fatigue Factor Impact")
    print("="*60)
    
    df['fatigue_bin'] = pd.cut(df['fatigue_factor'], 
                               bins=[0, 0.2, 0.4, 0.6, 1.0],
                               labels=['Low', 'Medium', 'High', 'Very High'])
    
    fatigue_analysis = df.groupby('fatigue_bin').agg({
        'incident_occurred': 'sum',
        'man_hours': 'sum',
        'incident_probability': 'mean'
    })
    
    fatigue_analysis['incident_rate'] = (
        fatigue_analysis['incident_occurred'] * 1e6 / 
        fatigue_analysis['man_hours']
    )
    
    print("\nIncident Rate by Fatigue Level:")
    print(fatigue_analysis[['incident_occurred', 'incident_probability', 'incident_rate']].to_string())
    
    print("\n" + "="*60)
    print("✓ Custom analysis completed!")
    print("="*60)


def scenario_comparison_example():
    """
    Example: Compare multiple scenarios.
    """
    print("\n" + "="*60)
    print("SCENARIO COMPARISON EXAMPLE")
    print("="*60)
    
    scenarios = {
        'Baseline': (1.0, 1.0),
        'Reduced Load': (0.8, 1.0),
        'Better Maintenance': (1.0, 0.6),
        'Optimized': (0.8, 0.6),
    }
    
    results = []
    
    for scenario_name, (load_factor, maint_factor) in scenarios.items():
        print(f"\nRunning scenario: {scenario_name}...")
        
        # Modify configurations
        from safety_twin_simulation import ZONE_CONFIGS
        import copy
        
        modified_configs = copy.deepcopy(ZONE_CONFIGS)
        for zone in modified_configs:
            modified_configs[zone].equipment_load_mean *= load_factor
            modified_configs[zone].maintenance_delay_mean *= maint_factor
        
        # Run simulation with modified configs
        import safety_twin_simulation
        original_configs = safety_twin_simulation.ZONE_CONFIGS
        safety_twin_simulation.ZONE_CONFIGS = modified_configs
        
        twin = SteelPlantDigitalTwin(seed=42)
        df = twin.run_simulation(days=90, workers_per_shift=50, shifts_per_day=3)
        kpis = twin.calculate_kpis(df)
        
        # Restore original configs
        safety_twin_simulation.ZONE_CONFIGS = original_configs
        
        results.append({
            'Scenario': scenario_name,
            'LTIFR': kpis['LTIFR'],
            'LTTR': kpis['LTTR'],
            'Severity_Rate': kpis['Severity_Rate'],
            'Incidents': kpis['Incident_Count']
        })
    
    # Display comparison
    print("\n" + "="*60)
    print("SCENARIO COMPARISON RESULTS")
    print("="*60)
    comparison_df = pd.DataFrame(results)
    print(comparison_df.to_string(index=False))
    
    # Calculate improvements vs baseline
    baseline = comparison_df[comparison_df['Scenario'] == 'Baseline'].iloc[0]
    print("\n" + "="*60)
    print("IMPROVEMENT vs BASELINE (%)")
    print("="*60)
    
    for _, row in comparison_df.iterrows():
        if row['Scenario'] != 'Baseline':
            ltifr_change = (row['LTIFR'] - baseline['LTIFR']) / baseline['LTIFR'] * 100
            incidents_change = (row['Incidents'] - baseline['Incidents']) / baseline['Incidents'] * 100
            print(f"\n{row['Scenario']}:")
            print(f"  LTIFR Change: {ltifr_change:+.1f}%")
            print(f"  Incident Change: {incidents_change:+.1f}%")


def data_exploration_example():
    """
    Example: Explore generated simulation data.
    """
    print("\n" + "="*60)
    print("DATA EXPLORATION EXAMPLE")
    print("="*60)
    
    # Load data
    print("\nLoading simulation data...")
    df = pd.read_csv('data/simulated_safety_data.csv')
    
    print(f"Data shape: {df.shape}")
    print(f"Date range: Day {df['day'].min()} to {df['day'].max()}")
    print(f"Zones: {df['zone'].unique()}")
    
    # Basic statistics
    print("\n" + "="*60)
    print("OPERATIONAL PARAMETER STATISTICS")
    print("="*60)
    
    params = ['temperature', 'pressure', 'equipment_load', 
             'human_exposure', 'maintenance_delay']
    
    for param in params:
        print(f"\n{param}:")
        print(f"  Mean: {df[param].mean():.2f}")
        print(f"  Std:  {df[param].std():.2f}")
        print(f"  Min:  {df[param].min():.2f}")
        print(f"  Max:  {df[param].max():.2f}")
    
    # Incident statistics
    print("\n" + "="*60)
    print("INCIDENT STATISTICS")
    print("="*60)
    
    incident_df = df[df['incident_occurred'] == True]
    print(f"Total incidents: {len(incident_df)}")
    print(f"Incident rate: {len(incident_df)/len(df)*100:.2f}%")
    
    if len(incident_df) > 0:
        print(f"\nLost hours statistics:")
        print(f"  Mean: {incident_df['lost_hours'].mean():.2f}")
        print(f"  Median: {incident_df['lost_hours'].median():.2f}")
        print(f"  Std: {incident_df['lost_hours'].std():.2f}")
        
        print(f"\nSeverity distribution:")
        severity_counts = incident_df['severity'].value_counts()
        for severity, count in severity_counts.items():
            print(f"  {severity}: {count} ({count/len(incident_df)*100:.1f}%)")
    
    print("\n✓ Data exploration completed!")


if __name__ == '__main__':
    # Run examples
    try:
        # Example 1: Custom analysis
        custom_analysis_example()
        
        # Example 2: Scenario comparison
        scenario_comparison_example()
        
        # Example 3: Data exploration
        data_exploration_example()
        
        print("\n" + "="*60)
        print("ALL EXAMPLES COMPLETED SUCCESSFULLY!")
        print("="*60)
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()

