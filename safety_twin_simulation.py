"""
Digital Safety Twin for Tata Steel Plant Operations
===================================================

This module simulates a Tata Steel plant's operational processes and safety incidents
using a directed graph model of the production line. Enhanced with realistic Tata Steel
plant parameters and comprehensive safety analysis.

Based on:
- Towards Safer Steel Operations (Sci Rep 2025) – risk modeling
- Hybrid Digital Twin for Process Industry (2021) – process simulation
- Digital Twin-Based Safety Risk Coupling (2021) – human-equipment-environment
- Digital Twin of Hot Metal Ladle System (2024) – steel process parameters
- Tata Steel operational data and industry benchmarks

Author: Digital Safety Twin System
Date: 2025
"""

import numpy as np
import pandas as pd
import networkx as nx
from scipy import stats
from scipy.special import expit  # logistic function
from dataclasses import dataclass
from typing import Dict, List, Tuple
import json
from pathlib import Path


@dataclass
class ZoneParameters:
    """
    Operational parameters for each production zone.
    Based on steel industry literature and safety standards.
    """
    name: str
    
    # Process parameters (from Hot Metal Ladle Digital Twin, 2024)
    temperature_mean: float  # Celsius
    temperature_std: float
    pressure_mean: float  # bar
    pressure_std: float
    
    # Operational factors (from Process Industry DT, 2021)
    equipment_load_mean: float  # 0-1 utilization
    equipment_load_std: float
    
    # Human factors (from Safety Risk Coupling, 2021)
    human_exposure_mean: float  # workers per shift
    human_exposure_std: float
    fatigue_factor_mean: float  # 0-1 scale
    fatigue_factor_std: float
    
    # Maintenance factors
    maintenance_delay_mean: float  # days since last maintenance
    maintenance_delay_std: float
    
    # Environmental factors (from Safer Steel Operations, 2025)
    noise_level_mean: float  # dB
    dust_concentration_mean: float  # mg/m³
    ventilation_effectiveness: float  # 0-1 scale
    
    # Safety infrastructure
    ppe_compliance_rate: float  # 0-1
    safety_training_hours: float  # annual hours per worker
    
    # Base risk coefficients (calibrated from literature)
    base_risk: float  # baseline incident probability


# Define Tata Steel plant zones with realistic parameters based on industry data
ZONE_CONFIGS = {
    'Coke_Oven': ZoneParameters(
        name='Coke Oven Battery',
        temperature_mean=1100.0,  # Coke oven temperature (800-1100°C)
        temperature_std=75.0,
        pressure_mean=0.1,  # Slight positive pressure
        pressure_std=0.05,
        equipment_load_mean=0.88,
        equipment_load_std=0.08,
        human_exposure_mean=18.0,  # High exposure - manual charging/discharging
        human_exposure_std=3.0,
        fatigue_factor_mean=0.42,  # Higher due to heat and fumes
        fatigue_factor_std=0.16,
        maintenance_delay_mean=20.0,  # Critical maintenance
        maintenance_delay_std=6.0,
        noise_level_mean=88.0,
        dust_concentration_mean=15.0,  # High dust from coal
        ventilation_effectiveness=0.68,  # Challenging environment
        ppe_compliance_rate=0.85,
        safety_training_hours=42.0,
        base_risk=0.00008  # Higher risk zone (realistic scaling for LTIFR ~1.5)
    ),
    'Sinter_Plant': ZoneParameters(
        name='Sinter Plant',
        temperature_mean=1350.0,  # Sintering temperature
        temperature_std=65.0,
        pressure_mean=1.2,
        pressure_std=0.3,
        equipment_load_mean=0.82,
        equipment_load_std=0.11,
        human_exposure_mean=14.0,
        human_exposure_std=2.5,
        fatigue_factor_mean=0.38,
        fatigue_factor_std=0.14,
        maintenance_delay_mean=14.0,
        maintenance_delay_std=4.5,
        noise_level_mean=92.0,
        dust_concentration_mean=18.0,  # Very high dust
        ventilation_effectiveness=0.72,
        ppe_compliance_rate=0.87,
        safety_training_hours=40.0,
        base_risk=0.00007  # Realistic scaling for LTIFR ~1.5
    ),
    'Blast_Furnace': ZoneParameters(
        name='Blast Furnace',
        temperature_mean=1520.0,  # Molten iron temperature
        temperature_std=55.0,
        pressure_mean=2.8,  # Higher pressure in BF
        pressure_std=0.35,
        equipment_load_mean=0.87,
        equipment_load_std=0.09,
        human_exposure_mean=16.0,  # High exposure for casting
        human_exposure_std=2.5,
        fatigue_factor_mean=0.38,
        fatigue_factor_std=0.16,
        maintenance_delay_mean=18.0,  # Hot maintenance challenges
        maintenance_delay_std=6.0,
        noise_level_mean=98.0,  # High noise
        dust_concentration_mean=12.0,
        ventilation_effectiveness=0.74,
        ppe_compliance_rate=0.89,
        safety_training_hours=44.0,
        base_risk=0.00006  # Realistic scaling for LTIFR ~1.5
    ),
    'BOF': ZoneParameters(
        name='Basic Oxygen Furnace',
        temperature_mean=1670.0,  # Steel making temperature
        temperature_std=45.0,
        pressure_mean=3.2,
        pressure_std=0.45,
        equipment_load_mean=0.84,
        equipment_load_std=0.10,
        human_exposure_mean=12.0,
        human_exposure_std=2.2,
        fatigue_factor_mean=0.40,
        fatigue_factor_std=0.17,
        maintenance_delay_mean=13.0,
        maintenance_delay_std=4.5,
        noise_level_mean=105.0,  # Very high noise during blowing
        dust_concentration_mean=14.0,
        ventilation_effectiveness=0.71,
        ppe_compliance_rate=0.91,
        safety_training_hours=46.0,
        base_risk=0.00007  # Realistic scaling for LTIFR ~1.5
    ),
    'Ladle_Metallurgy': ZoneParameters(
        name='Ladle Metallurgy Furnace',
        temperature_mean=1620.0,
        temperature_std=50.0,
        pressure_mean=1.8,
        pressure_std=0.4,
        equipment_load_mean=0.78,
        equipment_load_std=0.13,
        human_exposure_mean=9.0,
        human_exposure_std=2.0,
        fatigue_factor_mean=0.33,
        fatigue_factor_std=0.13,
        maintenance_delay_mean=11.0,
        maintenance_delay_std=3.5,
        noise_level_mean=87.0,
        dust_concentration_mean=8.0,
        ventilation_effectiveness=0.79,
        ppe_compliance_rate=0.90,
        safety_training_hours=39.0,
        base_risk=0.00005  # Realistic scaling for LTIFR ~1.5
    ),
    'Caster': ZoneParameters(
        name='Continuous Caster',
        temperature_mean=1180.0,  # Solidification temperature
        temperature_std=65.0,
        pressure_mean=1.6,
        pressure_std=0.25,
        equipment_load_mean=0.76,
        equipment_load_std=0.14,
        human_exposure_mean=10.0,
        human_exposure_std=1.8,
        fatigue_factor_mean=0.31,
        fatigue_factor_std=0.13,
        maintenance_delay_mean=10.5,
        maintenance_delay_std=3.2,
        noise_level_mean=86.0,
        dust_concentration_mean=7.0,
        ventilation_effectiveness=0.81,
        ppe_compliance_rate=0.92,
        safety_training_hours=38.0,
        base_risk=0.00004  # Realistic scaling for LTIFR ~1.5
    ),
    'Rolling_Mill': ZoneParameters(
        name='Hot Rolling Mill',
        temperature_mean=920.0,  # Rolling temperature
        temperature_std=85.0,
        pressure_mean=4.5,  # High mechanical pressure
        pressure_std=0.6,
        equipment_load_mean=0.73,
        equipment_load_std=0.16,
        human_exposure_mean=20.0,  # Highest exposure - manual operations
        human_exposure_std=3.5,
        fatigue_factor_mean=0.28,
        fatigue_factor_std=0.11,
        maintenance_delay_mean=9.0,
        maintenance_delay_std=2.8,
        noise_level_mean=94.0,  # Very high noise
        dust_concentration_mean=5.0,
        ventilation_effectiveness=0.83,
        ppe_compliance_rate=0.93,
        safety_training_hours=37.0,
        base_risk=0.00004  # Realistic scaling for LTIFR ~1.5
    ),
    'Power_Plant': ZoneParameters(
        name='Captive Power Plant',
        temperature_mean=550.0,  # Steam turbine temperatures
        temperature_std=45.0,
        pressure_mean=85.0,  # High pressure steam
        pressure_std=8.0,
        equipment_load_mean=0.81,
        equipment_load_std=0.12,
        human_exposure_mean=8.0,  # Lower exposure
        human_exposure_std=1.5,
        fatigue_factor_mean=0.26,
        fatigue_factor_std=0.11,
        maintenance_delay_mean=12.0,
        maintenance_delay_std=4.0,
        noise_level_mean=82.0,
        dust_concentration_mean=3.5,
        ventilation_effectiveness=0.88,
        ppe_compliance_rate=0.94,
        safety_training_hours=36.0,
        base_risk=0.00003  # Lower risk (realistic scaling for LTIFR ~1.5)
    )
}


class SteelPlantDigitalTwin:
    """
    Digital twin of a steel plant production line with safety modeling.
    
    Uses a directed graph to represent material flow and simulates
    operational parameters, human factors, and safety incidents.
    """
    
    def __init__(self, seed: int = 42):
        """
        Initialize the digital twin.
        
        Args:
            seed: Random seed for reproducibility
        """
        np.random.seed(seed)
        self.graph = self._create_plant_graph()
        self.zones = ZONE_CONFIGS
        self.simulation_data = []
        
    def _create_plant_graph(self) -> nx.DiGraph:
        """
        Create directed graph of Tata Steel plant production flow.
        
        Returns:
            NetworkX directed graph
        """
        G = nx.DiGraph()
        
        # Add nodes (production zones) - Tata Steel plant flow
        zones = ['Coke_Oven', 'Sinter_Plant', 'Blast_Furnace', 'BOF', 
                 'Ladle_Metallurgy', 'Caster', 'Rolling_Mill', 'Power_Plant']
        G.add_nodes_from(zones)
        
        # Add edges (material flow) - realistic steel plant flow
        edges = [
            # Raw material preparation
            ('Coke_Oven', 'Blast_Furnace'),  # Coke to BF
            ('Sinter_Plant', 'Blast_Furnace'),  # Sinter to BF
            # Iron to steel conversion
            ('Blast_Furnace', 'BOF'),  # Hot metal to BOF
            # Steel processing
            ('BOF', 'Ladle_Metallurgy'),  # Steel to LMF
            ('Ladle_Metallurgy', 'Caster'),  # Steel to caster
            # Final processing
            ('Caster', 'Rolling_Mill'),  # Slabs to rolling
            # Power supply (auxiliary)
            ('Power_Plant', 'Blast_Furnace'),  # Power to major units
            ('Power_Plant', 'BOF'),
            ('Power_Plant', 'Rolling_Mill')
        ]
        G.add_edges_from(edges)
        
        return G
    
    def sample_zone_parameters(self, zone: str, day: int, shift: int = 1) -> Dict[str, float]:
        """
        Sample operational parameters for a zone on a given day and shift.
        Enhanced with realistic effects: seasonal, shift, and day-of-week patterns.
        
        Args:
            zone: Zone identifier
            day: Simulation day
            shift: Shift number (1=day, 2=evening, 3=night)
            
        Returns:
            Dictionary of sampled parameters
        """
        config = self.zones[zone]
        
        # Day-of-week effect (higher fatigue on weekends due to longer shifts)
        day_of_week = day % 7
        weekend_factor = 1.15 if day_of_week in [5, 6] else 1.0  # Weekend multiplier
        
        # Seasonal effect (summer heat increases fatigue and reduces compliance)
        # Assuming day 1-91 = summer, 92-182 = monsoon, 183-273 = post-monsoon, 274-365 = winter
        # Calculate season: 0=Summer, 1=Monsoon, 2=Post-Monsoon, 3=Winter
        # Use (day - 1) to map day 1 to season 0, then clamp to handle edge cases
        day_in_year = ((day - 1) % 365) + 1  # Convert to 1-365 range
        season = min(3, (day_in_year - 1) // 91)  # Clamp to 0-3
        seasonal_multipliers = {
            0: (1.12, 0.95),  # Summer: higher fatigue, lower compliance
            1: (1.08, 0.98),  # Monsoon: moderate effects
            2: (1.02, 1.00),  # Post-monsoon: near baseline
            3: (0.98, 1.02)   # Winter: lower fatigue, higher compliance
        }
        fatigue_seasonal, compliance_seasonal = seasonal_multipliers[season]
        
        # Shift effect (night shift has higher fatigue, day shift better compliance)
        # Ensure shift is valid (1, 2, or 3)
        shift = min(max(1, shift), 3)  # Clamp shift to 1-3
        shift_fatigue = {1: 1.0, 2: 1.08, 3: 1.15}  # Day, Evening, Night
        shift_compliance = {1: 1.02, 2: 1.0, 3: 0.97}  # Day, Evening, Night
        shift_exposure = {1: 1.0, 2: 0.95, 3: 0.90}  # Slightly fewer workers at night
        
        # Realistic parameter sampling with correlations
        base_temp = np.random.normal(config.temperature_mean, config.temperature_std)
        # Temperature affects equipment load and maintenance needs
        temp_factor = (base_temp - config.temperature_mean) / (config.temperature_std * 3)
        equipment_load_base = config.equipment_load_mean + temp_factor * 0.05
        
        params = {
            'zone': zone,
            'day': day,
            'shift': shift,
            'season': season,
            'day_of_week': day_of_week,
            'temperature': base_temp,
            'pressure': np.random.normal(config.pressure_mean, config.pressure_std),
            'equipment_load': np.clip(np.random.normal(equipment_load_base, 
                                                       config.equipment_load_std), 0, 1),
            'human_exposure': max(0, np.random.normal(config.human_exposure_mean * shift_exposure[shift], 
                                                      config.human_exposure_std)),
            'fatigue_factor': np.clip(np.random.normal(config.fatigue_factor_mean, 
                                                       config.fatigue_factor_std) * 
                                     weekend_factor * fatigue_seasonal * shift_fatigue[shift], 
                                     0, 1),
            'maintenance_delay': max(0, np.random.normal(config.maintenance_delay_mean, 
                                                         config.maintenance_delay_std)),
            'noise_level': np.random.normal(config.noise_level_mean, 4.0),  # Higher variability
            'dust_concentration': max(0, np.random.normal(config.dust_concentration_mean, 1.5)),
            'ventilation_effectiveness': np.clip(np.random.normal(config.ventilation_effectiveness, 0.06), 0, 1),
            'ppe_compliance_rate': np.clip(np.random.normal(config.ppe_compliance_rate * compliance_seasonal * shift_compliance[shift], 
                                                           0.04), 0, 1),
        }
        
        return params
    
    def compute_incident_probability(self, zone: str, params: Dict[str, float]) -> float:
        """
        Compute probability of incident using logistic regression model.
        
        Based on "Digital Twin-Based Safety Risk Coupling" (2021):
        P(incident) = logistic(β₀ + Σ βᵢ × xᵢ)
        
        Args:
            zone: Zone identifier
            params: Operational parameters
            
        Returns:
            Incident probability [0, 1]
        """
        config = self.zones[zone]
        
        # Normalize parameters to [0, 1] scale
        temp_norm = (params['temperature'] - 800) / 1000  # Rough normalization
        pressure_norm = params['pressure'] / 5.0
        
        # Feature engineering based on literature
        # "Towards Safer Steel Operations" (2025) - multiplicative effects
        equipment_risk = params['equipment_load'] * (1 + params['maintenance_delay'] / 30)
        human_risk = params['human_exposure'] * params['fatigue_factor'] * (1 - params['ppe_compliance_rate'])
        environmental_risk = (params['noise_level'] / 100) * (params['dust_concentration'] / 10) * \
                           (1 - params['ventilation_effectiveness'])
        
        # Logistic regression coefficients (scaled down for realistic rates)
        # Higher coefficients = stronger influence on incident probability
        logit = (
            np.log(config.base_risk / (1 - config.base_risk)) +  # Base log-odds
            0.2 * temp_norm +  # Temperature effect
            0.15 * pressure_norm +  # Pressure effect
            0.3 * equipment_risk +  # Equipment & maintenance
            0.4 * human_risk +  # Human factors (strongest predictor)
            0.2 * environmental_risk  # Environmental conditions
        )
        
        # Convert to probability using logistic function
        probability = expit(logit)
        
        # Cap at realistic maximum (0.5% per shift = ~18% per year per zone)
        # This ensures we never get unrealistic incident rates
        return np.clip(probability, 0, 0.005)  # Cap at 0.5% per shift
    
    def simulate_incident(self, probability: float, params: Dict[str, float] = None) -> Tuple[bool, float, str, bool]:
        """
        Simulate whether an incident occurs and its characteristics.
        
        Args:
            probability: Incident probability
            params: Operational parameters (optional, used to adjust severity)
            
        Returns:
            Tuple of (incident_occurred, lost_hours, severity_category, is_recordable)
        """
        # Draw random number to determine if incident occurs
        if np.random.random() < probability:
            # Sample lost hours from lognormal distribution
            # LogNormal(μ=ln(80), σ=0.8) gives median ≈ 80 hours (industry realistic for LTIs)
            base_lost_hours = np.random.lognormal(mean=np.log(80), sigma=0.8)
            
            # Adjust severity based on operational conditions if params provided
            if params:
                # Compute severity multiplier from operational risk
                equipment_risk = params['equipment_load'] * (1 + params['maintenance_delay'] / 30)
                human_risk = params['human_exposure'] * params['fatigue_factor'] * (1 - params['ppe_compliance_rate'])
                environmental_risk = (params['noise_level'] / 100) * (params['dust_concentration'] / 10) * \
                                   (1 - params['ventilation_effectiveness'])
                
                # Combined risk factor (0.6 to 1.8 range) that scales lost hours
                # Worse conditions → more severe incidents
                severity_multiplier = 0.6 + 1.2 * (
                    0.3 * equipment_risk + 
                    0.4 * human_risk + 
                    0.3 * environmental_risk
                )
                
                # Apply severity adjustment
                lost_hours = base_lost_hours * severity_multiplier
            else:
                lost_hours = base_lost_hours
            
            # Categorize severity based on lost hours
            if lost_hours < 8:
                severity = 'Minor'
            elif lost_hours < 24:
                severity = 'Moderate'
            elif lost_hours < 168:  # 1 week
                severity = 'Serious'
            else:
                severity = 'Severe'
            
            # Determine if this is an LTI or non-LTI incident
            # Industry: ~25% of incidents are LTIs, rest are non-LTI recordables or first aid
            is_lti = np.random.random() < 0.25  # 25% chance of LTI
            
            if is_lti:
                # Keep the calculated lost_hours for LTI
                is_recordable = True
            else:
                # Non-LTI incident: lost hours = 0, but may still be recordable
                lost_hours = 0
                severity = 'Minor'
                # ~90% of non-LTI incidents are still recordable (medical treatment, restricted work)
                is_recordable = np.random.random() < 0.90
            
            return True, lost_hours, severity, is_recordable
        
        return False, 0.0, 'None', False
    
    def run_simulation(self, days: int = 365, workers_per_shift: int = 50, 
                      shifts_per_day: int = 3) -> pd.DataFrame:
        """
        Run the digital twin simulation for specified number of days.
        Enhanced with shift-wise simulation for more realistic modeling.
        
        Args:
            days: Number of simulation days
            workers_per_shift: Workers per shift
            shifts_per_day: Number of shifts per day
            
        Returns:
            DataFrame with simulation results
        """
        print(f"Starting Tata Steel digital twin simulation for {days} days...")
        print(f"Plant configuration: {len(self.zones)} zones")
        print(f"Workers: {workers_per_shift} per shift, {shifts_per_day} shifts/day\n")
        
        simulation_records = []
        
        for day in range(1, days + 1):
            for zone in self.graph.nodes():
                for shift in range(1, shifts_per_day + 1):
                    # Sample operational parameters (shift-aware)
                    params = self.sample_zone_parameters(zone, day, shift)
                    
                    # Compute incident probability
                    incident_prob = self.compute_incident_probability(zone, params)
                    
                    # Simulate incident occurrence
                    incident_occurred, lost_hours, severity, is_recordable = self.simulate_incident(incident_prob, params)
                    
                    # Calculate man-hours for this zone-day-shift
                    zone_workers = int(params['human_exposure'])
                    man_hours = zone_workers * 8  # 8-hour shift
                    
                    # Record data
                    record = {
                        'day': day,
                        'shift': shift,
                        'zone': zone,
                        'temperature': params['temperature'],
                        'pressure': params['pressure'],
                        'equipment_load': params['equipment_load'],
                        'human_exposure': params['human_exposure'],
                        'fatigue_factor': params['fatigue_factor'],
                        'maintenance_delay': params['maintenance_delay'],
                        'noise_level': params['noise_level'],
                        'dust_concentration': params['dust_concentration'],
                        'ventilation_effectiveness': params['ventilation_effectiveness'],
                        'ppe_compliance_rate': params['ppe_compliance_rate'],
                        'incident_probability': incident_prob,
                        'incident_occurred': incident_occurred,
                        'lost_hours': lost_hours,
                        'severity': severity,
                        'man_hours': man_hours,
                        'is_recordable': is_recordable,
                        'season': params.get('season', 0),
                        'day_of_week': params.get('day_of_week', 0)
                    }
                    
                    simulation_records.append(record)
            
            if day % 50 == 0:
                print(f"Completed day {day}/{days}")
        
        self.simulation_data = pd.DataFrame(simulation_records)
        print(f"\n✓ Simulation complete. Generated {len(self.simulation_data)} records.")
        
        return self.simulation_data
    
    def calculate_kpis(self, df: pd.DataFrame) -> Dict[str, float]:
        """
        Calculate safety Key Performance Indicators.
        
        KPI definitions (OSHA/ANSI standards):
        - LTIFR: Lost Time Injury Frequency Rate (per million man-hours)
        - LTTR: Average Lost Hours per LTI (hours per LTI)
        - Severity Rate: Lost days per thousand man-hours
        - TRIR: Total Recordable Incident Rate (per million man-hours)
        
        Args:
            df: Simulation data
            
        Returns:
            Dictionary of KPIs
        """
        # Aggregate metrics
        total_man_hours = df['man_hours'].sum()
        
        # Lost Time Injuries (incidents with lost hours)
        lti_df = df[df['lost_hours'] > 0]
        lti_count = len(lti_df)
        total_lost_hours = lti_df['lost_hours'].sum()
        total_lost_days = total_lost_hours / 8  # Convert to days
        
        # Recordable incidents
        recordable_count = df['is_recordable'].sum()
        
        # Calculate KPIs
        kpis = {
            'LTIFR': (lti_count * 1e6) / total_man_hours if total_man_hours > 0 else 0,
            'LTTR': total_lost_hours / lti_count if lti_count > 0 else 0,
            'Severity_Rate': (total_lost_days * 1e3) / total_man_hours if total_man_hours > 0 else 0,
            'TRIR': (recordable_count * 1e6) / total_man_hours if total_man_hours > 0 else 0,
            'Total_LTI': lti_count,
            'Total_Lost_Hours': total_lost_hours,
            'Total_Lost_Days': total_lost_days,
            'Total_Man_Hours': total_man_hours,
            'Total_Recordable': recordable_count,
            'Incident_Count': df['incident_occurred'].sum()
        }
        
        return kpis
    
    def calculate_zone_kpis(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate KPIs by zone.
        
        Args:
            df: Simulation data
            
        Returns:
            DataFrame with zone-wise KPIs
        """
        zone_kpis = []
        
        for zone in df['zone'].unique():
            zone_df = df[df['zone'] == zone]
            kpis = self.calculate_kpis(zone_df)
            kpis['zone'] = zone
            zone_kpis.append(kpis)
        
        return pd.DataFrame(zone_kpis)
    
    def export_results(self, output_dir: str = 'data'):
        """
        Export simulation results and KPIs to CSV files.
        
        Args:
            output_dir: Output directory path
        """
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Export main simulation data
        sim_file = output_path / 'simulated_safety_data.csv'
        self.simulation_data.to_csv(sim_file, index=False)
        print(f"✓ Exported simulation data to {sim_file}")
        
        # Calculate and export overall KPIs
        overall_kpis = self.calculate_kpis(self.simulation_data)
        # Convert numpy types to Python native types for JSON serialization
        overall_kpis = {k: float(v) if isinstance(v, (np.integer, np.floating)) else v 
                       for k, v in overall_kpis.items()}
        kpi_file = output_path / 'safety_kpis.json'
        with open(kpi_file, 'w') as f:
            json.dump(overall_kpis, f, indent=2)
        print(f"✓ Exported overall KPIs to {kpi_file}")
        
        # Calculate and export zone KPIs
        zone_kpis = self.calculate_zone_kpis(self.simulation_data)
        zone_file = output_path / 'zone_kpis.csv'
        zone_kpis.to_csv(zone_file, index=False)
        print(f"✓ Exported zone KPIs to {zone_file}")
        
        # Print summary
        print("\n" + "="*60)
        print("SAFETY PERFORMANCE INDICATORS - SUMMARY")
        print("="*60)
        for key, value in overall_kpis.items():
            if 'Rate' in key or 'LTIFR' in key or 'TRIR' in key or 'LTTR' in key:
                print(f"{key:.<30} {value:.2f}")
            else:
                print(f"{key:.<30} {value:.0f}")
        print("="*60)


class BenefitCostAnalyzer:
    """
    Benefit-Cost Analysis module for safety equipment investments.
    
    Calculates ROI, Net Present Value, and cost-effectiveness of safety
    interventions including Value of Statistical Life (VoSL).
    
    Based on:
    - World Bank VoSL estimates for India (2024)
    - Tata Steel safety investment benchmarks
    - Industry cost-benefit analysis frameworks
    """
    
    def __init__(self, vosl_india_lakh: float = 45.0):
        """
        Initialize the benefit-cost analyzer.
        
        Args:
            vosl_india_lakh: Value of Statistical Life in lakhs INR (default ~45 lakhs = $60k USD)
        """
        self.vosl_lakh = vosl_india_lakh  # VoSL in lakhs INR
        self.discount_rate = 0.08  # 8% discount rate (India's typical rate)
        
    def calculate_incident_costs(self, df: pd.DataFrame, 
                                  include_fatal: bool = True) -> Dict[str, float]:
        """
        Calculate total costs from incidents.
        
        Costs include:
        - Direct costs: Medical, compensation, equipment damage
        - Indirect costs: Lost productivity, training replacement workers
        - Fatal costs: VoSL for fatalities
        
        Args:
            df: Simulation data
            include_fatal: Whether to include fatal incident costs using VoSL
            
        Returns:
            Dictionary of cost components
        """
        incident_df = df[df['incident_occurred'] == True].copy()
        
        if len(incident_df) == 0:
            return {
                'total_cost_lakh': 0,
                'direct_costs_lakh': 0,
                'indirect_costs_lakh': 0,
                'fatal_costs_lakh': 0,
                'lost_productivity_lakh': 0
            }
        
        # Direct costs per incident (medical, compensation, damage)
        # Scale by severity: Minor (0.2L), Moderate (2L), Serious (15L), Severe (50L)
        severity_costs = {
            'Minor': 0.2,
            'Moderate': 2.0,
            'Serious': 15.0,
            'Severe': 50.0,
            'None': 0.0
        }
        
        direct_costs = incident_df['severity'].map(severity_costs).sum()
        
        # Indirect costs (lost productivity, training, etc.)
        # Typically 2-4x direct costs for moderate-serious incidents
        indirect_multiplier = 2.5
        indirect_costs = direct_costs * indirect_multiplier
        
        # Lost productivity cost (man-hours lost)
        # Average wage: ₹600/hour for skilled steel worker = ₹4,800/shift
        wage_per_hour_lakh = 0.006  # ₹600/hour = 0.006 lakhs/hour
        total_lost_hours = incident_df['lost_hours'].sum()
        lost_productivity = total_lost_hours * wage_per_hour_lakh
        
        # Fatal incidents (using VoSL)
        # Assume 1% of severe incidents result in fatalities
        severe_incidents = len(incident_df[incident_df['severity'] == 'Severe'])
        estimated_fatalities = max(0, int(severe_incidents * 0.01))
        
        fatal_costs = 0
        if include_fatal and estimated_fatalities > 0:
            fatal_costs = estimated_fatalities * self.vosl_lakh
        
        total_costs = direct_costs + indirect_costs + lost_productivity + fatal_costs
        
        return {
            'total_cost_lakh': total_costs,
            'direct_costs_lakh': direct_costs,
            'indirect_costs_lakh': indirect_costs,
            'fatal_costs_lakh': fatal_costs,
            'lost_productivity_lakh': lost_productivity,
            'estimated_fatalities': estimated_fatalities,
            'total_incidents': len(incident_df),
            'total_lost_hours': total_lost_hours
        }
    
    def analyze_safety_equipment(self, baseline_df: pd.DataFrame,
                                   improved_df: pd.DataFrame,
                                   equipment_cost_lakh: float,
                                   annual_maintenance_lakh: float,
                                   equipment_life_years: int = 10,
                                   risk_reduction_pct: float = 0.25) -> Dict[str, float]:
        """
        Analyze benefit-cost of safety equipment investment.
        
        Args:
            baseline_df: Baseline simulation data
            improved_df: Simulation data after safety equipment installation
            equipment_cost_lakh: Initial cost in lakhs INR
            annual_maintenance_lakh: Annual maintenance cost in lakhs INR
            equipment_life_years: Expected equipment lifetime
            risk_reduction_pct: Percentage reduction in incidents (0-1)
            
        Returns:
            Dictionary with BCA metrics
        """
        # Calculate baseline costs
        baseline_costs = self.calculate_incident_costs(baseline_df)
        
        # Calculate improved costs (with risk reduction)
        improved_costs = self.calculate_incident_costs(improved_df)
        
        # Annual cost savings
        annual_savings = baseline_costs['total_cost_lakh'] * risk_reduction_pct
        
        # Total lifecycle costs
        total_costs = equipment_cost_lakh
        for year in range(1, equipment_life_years + 1):
            maintenance_pv = annual_maintenance_lakh / ((1 + self.discount_rate) ** year)
            total_costs += maintenance_pv
        
        # Total lifecycle benefits (NPV)
        total_benefits = 0
        for year in range(1, equipment_life_years + 1):
            benefit_pv = annual_savings / ((1 + self.discount_rate) ** year)
            total_benefits += benefit_pv
        
        # Net Present Value
        npv = total_benefits - total_costs
        
        # Benefit-Cost Ratio
        bcr = total_benefits / total_costs if total_costs > 0 else 0
        
        # Payback period
        cumulative_savings = 0
        payback_years = None
        for year in range(1, equipment_life_years + 1):
            cumulative_savings += annual_savings / ((1 + self.discount_rate) ** year)
            if cumulative_savings >= equipment_cost_lakh and payback_years is None:
                payback_years = year
        
        # ROI (Internal Rate of Return approximation)
        # Simplified: annual return / initial investment
        roi_pct = ((annual_savings - annual_maintenance_lakh) / equipment_cost_lakh) * 100 if equipment_cost_lakh > 0 else 0
        
        return {
            'equipment_cost_lakh': equipment_cost_lakh,
            'annual_maintenance_lakh': annual_maintenance_lakh,
            'equipment_life_years': equipment_life_years,
            'baseline_annual_cost_lakh': baseline_costs['total_cost_lakh'],
            'risk_reduction_pct': risk_reduction_pct * 100,
            'annual_savings_lakh': annual_savings,
            'total_costs_pv_lakh': total_costs,
            'total_benefits_pv_lakh': total_benefits,
            'npv_lakh': npv,
            'benefit_cost_ratio': bcr,
            'payback_period_years': payback_years,
            'roi_pct': roi_pct,
            'is_viable': npv > 0 and bcr > 1.0
        }
    
    def compare_interventions(self, interventions: List[Dict]) -> pd.DataFrame:
        """
        Compare multiple safety interventions.
        
        Args:
            interventions: List of intervention dictionaries with keys:
                - name: Intervention name
                - baseline_df: Baseline data
                - improved_df: Improved data
                - equipment_cost_lakh: Cost
                - annual_maintenance_lakh: Maintenance
                - equipment_life_years: Lifetime
                - risk_reduction_pct: Risk reduction
                
        Returns:
            DataFrame comparing interventions
        """
        results = []
        
        for intervention in interventions:
            bca = self.analyze_safety_equipment(
                intervention['baseline_df'],
                intervention['improved_df'],
                intervention['equipment_cost_lakh'],
                intervention['annual_maintenance_lakh'],
                intervention.get('equipment_life_years', 10),
                intervention.get('risk_reduction_pct', 0.25)
            )
            
            bca['intervention_name'] = intervention['name']
            results.append(bca)
        
        return pd.DataFrame(results)


def main():
    """
    Main execution function.
    """
    print("="*60)
    print("DIGITAL SAFETY TWIN - TATA STEEL PLANT SIMULATION")
    print("="*60)
    print()
    
    # Initialize digital twin
    twin = SteelPlantDigitalTwin(seed=42)
    
    # Run simulation
    df = twin.run_simulation(days=365, workers_per_shift=50, shifts_per_day=3)
    
    # Export results
    twin.export_results()
    
    print("\n✓ Digital twin simulation completed successfully!")
    print(f"  Total incidents: {df['incident_occurred'].sum()}")
    print(f"  Data points: {len(df)}")
    print(f"  Output directory: data/")


if __name__ == '__main__':
    main()

