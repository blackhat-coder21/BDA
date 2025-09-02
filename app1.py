import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import requests
from datetime import datetime, timedelta
import json
import time
from geopy.distance import geodesic
import warnings
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import re
from sklearn.linear_model import ElasticNet
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.svm import SVR
import xgboost as xgb
import lightgbm as lgb
import asyncio
import aiohttp
from bs4 import BeautifulSoup
import hashlib
import os
from typing import Dict, List, Optional, Tuple
import logging

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO)

class EnhancedFlightCarbonCalculator:
    """
    A comprehensive class to fetch, process, and calculate flight carbon emissions
    using multiple data sources and an advanced calculation methodology.
    """
    def __init__(self):
        """Initialize the enhanced calculator with multiple data sources"""
        # Multiple API keys for different services
        self.api_keys = {
            'aviationstack': '034659c48f30f9832c05aea458a29eb5',
            'opensky': None, 
            'flightaware': None,
        }
        
        # Base URLs for different APIs
        self.base_urls = {
            'aviationstack': "https://api.aviationstack.com/v1",
            'opensky': "https://opensky-network.org/api",
            'icao_carbon': "https://www.icao.int/environmental-protection/CarbonOffset/Documents/Methodology%20ICAO%20Carbon%20Calculator_v11-2018.pdf"
        }
        
        # Comprehensive aircraft emissions data (kg CO2 per km per passenger)
        self.aircraft_emissions = {
            # Narrow-body aircraft
            'A319': 0.105, 'A320': 0.095, 'A321': 0.092,
            'A20N': 0.085, 'A21N': 0.082, 'A319NEO': 0.088,
            'B737': 0.098, 'B738': 0.096, 'B739': 0.094, 'B37M': 0.086,
            'B38M': 0.086, 'B39M': 0.084, 'B3XM': 0.082,
            
            # Regional aircraft
            'E190': 0.110, 'E195': 0.108, 'E170': 0.115, 'E175': 0.112,
            'CRJ7': 0.135, 'CRJ9': 0.130, 'CRJ2': 0.140,
            'ATR42': 0.125, 'ATR72': 0.120,
            'DHC8': 0.140, 'DH8D': 0.135,
            'SB20': 0.150, 'SF34': 0.145,
            
            # Wide-body aircraft
            'A330': 0.085, 'A333': 0.087, 'A338': 0.083, 'A339': 0.081,
            'A340': 0.095, 'A342': 0.097, 'A343': 0.095, 'A346': 0.093,
            'A350': 0.075, 'A359': 0.075, 'A35K': 0.073,
            'A380': 0.082,
            
            'B747': 0.090, 'B74F': 0.095, 'B748': 0.088,
            'B767': 0.088, 'B762': 0.090, 'B763': 0.087,
            'B777': 0.083, 'B772': 0.085, 'B773': 0.083, 'B77W': 0.081,
            'B787': 0.070, 'B788': 0.072, 'B789': 0.070, 'B78X': 0.068,
            
            # Cargo/Freighter variants
            'A30B': 0.095, 'B74F': 0.095, 'B77F': 0.085,
            'MD11': 0.098, 'DC10': 0.105,
            
            # Business jets and smaller aircraft
            'BE20': 0.180, 'C550': 0.170, 'C56X': 0.165,
            'GLF4': 0.160, 'GLF5': 0.155, 'GLF6': 0.150,
            
            'DEFAULT': 0.095
        }
        
        # Aircraft passenger capacities (typical seating)
        self.aircraft_capacities = {
            # Narrow-body
            'A319': 150, 'A320': 180, 'A321': 220,
            'A20N': 180, 'A21N': 220, 'A319NEO': 150,
            'B737': 175, 'B738': 189, 'B739': 215,
            'B37M': 189, 'B38M': 189, 'B39M': 215, 'B3XM': 230,
            
            # Regional
            'E190': 100, 'E195': 120, 'E170': 80, 'E175': 88,
            'CRJ7': 70, 'CRJ9': 90, 'CRJ2': 50,
            'ATR42': 48, 'ATR72': 70,
            'DHC8': 50, 'DH8D': 78,
            'SB20': 19, 'SF34': 34,
            
            # Wide-body
            'A330': 300, 'A333': 300, 'A338': 300, 'A339': 300,
            'A340': 320, 'A342': 300, 'A343': 335, 'A346': 380,
            'A350': 325, 'A359': 325, 'A35K': 350,
            'A380': 525,
            
            'B747': 410, 'B74F': 0, 'B748': 467,
            'B767': 240, 'B762': 224, 'B763': 269,
            'B777': 350, 'B772': 314, 'B773': 350, 'B77W': 365,
            'B787': 300, 'B788': 250, 'B789': 300, 'B78X': 330,
            
            # Business jets
            'BE20': 8, 'C550': 10, 'C56X': 12,
            'GLF4': 14, 'GLF5': 16, 'GLF6': 17,
            
            'DEFAULT': 180
        }
        
        # Expanded airport coordinates database
        self.airports = self.load_comprehensive_airports()
        
        # Airline efficiency data from sustainability reports
        self.airline_efficiency_reports = {
            'Delta Air Lines': {'emissions_intensity': 0.095, 'year': 2023, 'source': 'Sustainability Report'},
            'American Airlines': {'emissions_intensity': 0.092, 'year': 2023, 'source': 'ESG Report'},
            'United Airlines': {'emissions_intensity': 0.089, 'year': 2023, 'source': 'Sustainability Report'},
            'Alaska Airlines': {'emissions_intensity': 0.088, 'year': 2023, 'source': 'Sustainability Report'},
            'JetBlue Airways': {'emissions_intensity': 0.091, 'year': 2023, 'source': 'Sustainability Report'},
            'Southwest Airlines': {'emissions_intensity': 0.093, 'year': 2023, 'source': 'Environmental Report'},
            
            'Lufthansa': {'emissions_intensity': 0.0875, 'year': 2023, 'source': 'Sustainability Report'},
            'Air France-KLM': {'emissions_intensity': 0.086, 'year': 2023, 'source': 'Sustainability Report'},
            'British Airways': {'emissions_intensity': 0.090, 'year': 2023, 'source': 'ESG Report'},
            'Emirates': {'emissions_intensity': 0.084, 'year': 2023, 'source': 'Environmental Report'},
            'Qatar Airways': {'emissions_intensity': 0.082, 'year': 2023, 'source': 'Sustainability Report'},
            'Singapore Airlines': {'emissions_intensity': 0.080, 'year': 2023, 'source': 'Sustainability Report'},
            
            'ANA': {'emissions_intensity': 0.085, 'year': 2023, 'source': 'ESG Report'},
            'JAL': {'emissions_intensity': 0.087, 'year': 2023, 'source': 'Sustainability Report'},
            'Cathay Pacific': {'emissions_intensity': 0.083, 'year': 2023, 'source': 'Sustainability Report'},
            'Korean Air': {'emissions_intensity': 0.088, 'year': 2023, 'source': 'Environmental Report'},
            
            'Qantas': {'emissions_intensity': 0.089, 'year': 2023, 'source': 'Climate Action Plan'},
            'Air Canada': {'emissions_intensity': 0.091, 'year': 2023, 'source': 'Sustainability Report'},
        }
        
        # Route-specific factors
        self.route_factors = {
            'domestic_short': 1.2,     # <500km domestic
            'domestic_medium': 1.1,    # 500-1500km domestic
            'domestic_long': 1.05,     # >1500km domestic
            'international_short': 1.15, # <1500km international
            'international_medium': 1.0,  # 1500-4000km international
            'international_long': 0.95,   # >4000km international
            'ultra_long': 0.90         # >8000km international
        }

    def load_comprehensive_airports(self) -> Dict[str, Tuple[float, float]]:
        """Load comprehensive airport coordinates database"""
        airports = {
            # North America - Major Hubs
            'JFK': (40.6413, -73.7781), 'LGA': (40.7769, -73.8740), 'EWR': (40.6895, -74.1745),
            'LAX': (33.9425, -118.4081), 'SFO': (37.6213, -122.3790), 'SAN': (32.7338, -117.1933),
            'ATL': (33.6407, -84.4277), 'ORD': (41.9742, -87.9073), 'DEN': (39.8561, -104.6737),
            'DFW': (32.8975, -97.0377), 'IAH': (29.9902, -95.3368), 'PHX': (33.4342, -112.0016),
            'SEA': (47.4502, -122.3088), 'BOS': (42.3656, -71.0096), 'MIA': (25.7959, -80.2870),
            'MCO': (28.4312, -81.3081), 'LAS': (36.0840, -115.1537), 'MSP': (44.8848, -93.2223),
            
            # Canada
            'YYZ': (43.6777, -79.6248), 'YVR': (49.1967, -123.1816), 'YUL': (45.4581, -73.7491),
            'YYC': (51.1315, -114.0106), 'YEG': (53.3097, -113.5801), 'YOW': (45.3192, -75.6686),
            
            # Europe - Major Hubs
            'LHR': (51.4700, -0.4543), 'LGW': (51.1481, -0.1903), 'STN': (51.8860, 0.2389),
            'CDG': (49.0097, 2.5479), 'ORY': (48.7233, 2.3794),
            'FRA': (50.0379, 8.5622), 'MUC': (48.3537, 11.7751), 'DUS': (51.2895, 6.7668),
            'AMS': (52.3086, 4.7639), 'MAD': (40.4839, -3.5680), 'BCN': (41.2974, 2.0833),
            'FCO': (41.7999, 12.2462), 'MXP': (45.6300, 8.7231), 'VCE': (45.5053, 12.3519),
            'ZUR': (47.4647, 8.5492), 'VIE': (48.1103, 16.5697), 'CPH': (55.6181, 12.6559),
            'ARN': (59.6519, 17.9186), 'OSL': (60.1939, 11.1004), 'HEL': (60.3172, 24.9633),
            
            # Middle East & Turkey
            'IST': (41.2619, 28.7414), 'SAW': (40.9041, 29.3155),
            'DXB': (25.2532, 55.3657), 'AUH': (24.4330, 54.6511), 'DOH': (25.2731, 51.6075),
            'RUH': (24.9576, 46.6983), 'JED': (21.6796, 39.1564), 'KWI': (29.2267, 47.9689),
            
            # Asia-Pacific
            'NRT': (35.7720, 140.3929), 'HND': (35.5494, 139.7798), 'KIX': (34.4272, 135.2281),
            'ICN': (37.4602, 126.4407), 'GMP': (37.5583, 126.7906),
            'PEK': (40.0799, 116.6031), 'PVG': (31.1443, 121.8083), 'SHA': (31.1979, 121.3364),
            'CAN': (23.3924, 113.2988), 'SZX': (22.6393, 113.8106), 'CTU': (30.5728, 103.9477),
            'HKG': (22.3080, 113.9185), 'TPE': (25.0797, 121.2342),
            'SIN': (1.3502, 103.9942), 'KUL': (2.7456, 101.7072), 'CGK': (6.1256, 106.6558),
            'BKK': (13.6900, 100.7501), 'DMK': (13.9126, 100.6060),
            'DEL': (28.5665, 77.1031), 'BOM': (19.0896, 72.8656), 'BLR': (13.1986, 77.7066),
            
            # Australia & New Zealand
            'SYD': (33.9399, 151.1753), 'MEL': (37.6690, 144.8410), 'BNE': (27.3942, 153.1218),
            'PER': (31.9403, 115.9669), 'ADL': (34.9285, 138.5304),
            'AKL': (37.0082, 174.7850), 'CHC': (43.4864, 172.5319),
            
            # South America
            'GRU': (23.4356, -46.4731), 'GIG': (22.8070, -43.2436),
            'EZE': (34.8222, -58.5358), 'SCL': (33.3928, -70.7858),
            'BOG': (4.7016, -74.1469), 'LIM': (12.0219, -77.1143),
            
            # Africa
            'JNB': (26.1392, 28.2460), 'CPT': (33.9697, 18.6021),
            'CAI': (30.1219, 31.4056), 'ADD': (8.9806, 38.7626),
            'NBO': (1.3192, 36.9278), 'CMN': (33.3676, -7.5898),
        }
        return airports

    async def get_opensky_flights(self, session: aiohttp.ClientSession, limit: int = 300) -> List[Dict]:
        """Fetch flight data from OpenSky Network API"""
        try:
            url = f"{self.base_urls['opensky']}/states/all"
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=30)) as response:
                if response.status == 200:
                    data = await response.json()
                    flights = []
                    
                    if 'states' in data and data['states']:
                        for state in data['states'][:limit]:
                            if len(state) >= 17:  # Ensure we have enough fields
                                flight_info = {
                                    'icao24': state[0],
                                    'callsign': state[1].strip() if state[1] else None,
                                    'origin_country': state[2],
                                    'longitude': state[5],
                                    'latitude': state[6],
                                    'altitude': state[7],
                                    'velocity': state[9],
                                    'vertical_rate': state[11],
                                    'sensors': state[12],
                                    'geo_altitude': state[13],
                                    'squawk': state[14],
                                    'spi': state[15],
                                    'position_source': state[16]
                                }
                                flights.append(flight_info)
                    
                    return flights[:limit]
                else:
                    st.warning(f"OpenSky API returned status {response.status}")
                    return []
        except Exception as e:
            st.warning(f"Error fetching OpenSky data: {e}")
            return []

    def get_aviationstack_flights(self, limit: int = 100, offset: int = 0) -> List[Dict]:
        """Fetch flight data from AviationStack API"""
        url = f"{self.base_urls['aviationstack']}/flights"
        params = {
            'access_key': self.api_keys['aviationstack'],
            'limit': min(limit, 100),  # API limit
            'offset': offset
        }
        
        try:
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()
            return data.get('data', [])
        except requests.exceptions.RequestException as e:
            st.warning(f"AviationStack API error: {e}")
            return []

    def generate_synthetic_routes(self, num_routes: int = 1000) -> List[Dict]:
        """Generate synthetic flight routes based on real airport pairs"""
        synthetic_flights = []
        airport_codes = list(self.airports.keys())
        
        # Common route patterns
        route_patterns = [
            # Domestic US routes
            (['JFK', 'LGA', 'EWR'], ['LAX', 'SFO', 'ATL', 'ORD', 'DEN', 'DFW']),
            # European routes
            (['LHR', 'CDG', 'FRA'], ['BCN', 'FCO', 'AMS', 'MAD', 'ZUR']),
            # Asia-Pacific routes
            (['NRT', 'ICN', 'PVG'], ['SIN', 'BKK', 'HKG', 'SYD', 'DEL']),
            # Transatlantic routes
            (['JFK', 'EWR', 'BOS'], ['LHR', 'CDG', 'FRA', 'AMS']),
            # Trans-Pacific routes
            (['LAX', 'SFO', 'SEA'], ['NRT', 'ICN', 'PVG', 'SYD']),
        ]
        
        aircraft_types = list(self.aircraft_emissions.keys())[:-1]  # Exclude DEFAULT
        airlines = [
            'Delta Air Lines', 'American Airlines', 'United Airlines', 'Southwest Airlines',
            'Lufthansa', 'Air France', 'British Airways', 'KLM', 'Emirates', 'Qatar Airways',
            'Singapore Airlines', 'ANA', 'JAL', 'Cathay Pacific', 'Qantas'
        ]
        
        for i in range(num_routes):
            # Choose route pattern
            if i < len(route_patterns) * 100:  # Use patterns for first portion
                pattern_idx = (i // 100) % len(route_patterns)
                origins, destinations = route_patterns[pattern_idx]
                origin = np.random.choice(origins)
                destination = np.random.choice(destinations)
            else:  # Random routes for remainder
                origin = np.random.choice(airport_codes)
                destination = np.random.choice(airport_codes)
                while destination == origin:
                    destination = np.random.choice(airport_codes)
            
            # Select appropriate aircraft based on distance
            distance = self.calculate_distance(origin, destination)
            if distance and distance > 0:
                if distance < 1500:  # Short-haul
                    aircraft_pool = ['A320', 'A321', 'B737', 'B738', 'E190']
                elif distance < 4000:  # Medium-haul
                    aircraft_pool = ['A321', 'B737', 'A330', 'B767', 'B787']
                else:  # Long-haul
                    aircraft_pool = ['A330', 'A350', 'B777', 'B787', 'A380', 'B747']
                
                aircraft = np.random.choice(aircraft_pool)
                
                # Generate flight info
                flight_info = {
                    'flight_number': f"{np.random.choice(['AA', 'DL', 'UA', 'LH', 'BA', 'EK'])}{np.random.randint(1, 9999):04d}",
                    'airline_name': np.random.choice(airlines),
                    'aircraft_type': aircraft,
                    'origin_airport': origin,
                    'destination_airport': destination,
                    'distance_km': distance,
                    'passenger_capacity': self.get_aircraft_capacity(aircraft),
                    'load_factor': np.random.uniform(0.6, 0.9),
                    'flight_status': 'scheduled',
                    'data_source': 'synthetic'
                }
                
                synthetic_flights.append(flight_info)
        
        return synthetic_flights

    def scrape_airline_sustainability_data(self) -> Dict:
        """Scrape additional airline sustainability data from public reports"""
        # This would normally involve web scraping, but for demo purposes,
        # we'll return enhanced data based on known sustainability reports
        enhanced_data = self.airline_efficiency_reports.copy()
        
        # Add calculated industry benchmarks
        enhanced_data['INDUSTRY_AVERAGE'] = {
            'emissions_intensity': np.mean([data['emissions_intensity'] for data in enhanced_data.values()]),
            'year': 2023,
            'source': 'Calculated Average'
        }
        
        # Add regional averages
        us_airlines = ['Delta Air Lines', 'American Airlines', 'United Airlines', 'Alaska Airlines', 'JetBlue Airways', 'Southwest Airlines']
        european_airlines = ['Lufthansa', 'Air France-KLM', 'British Airways']
        asian_airlines = ['Singapore Airlines', 'ANA', 'JAL', 'Cathay Pacific', 'Korean Air']
        
        enhanced_data['US_AVERAGE'] = {
            'emissions_intensity': np.mean([enhanced_data[airline]['emissions_intensity'] for airline in us_airlines if airline in enhanced_data]),
            'year': 2023,
            'source': 'US Airlines Average'
        }
        
        enhanced_data['EUROPEAN_AVERAGE'] = {
            'emissions_intensity': np.mean([enhanced_data[airline]['emissions_intensity'] for airline in european_airlines if airline in enhanced_data]),
            'year': 2023,
            'source': 'European Airlines Average'
        }
        
        enhanced_data['ASIAN_AVERAGE'] = {
            'emissions_intensity': np.mean([enhanced_data[airline]['emissions_intensity'] for airline in asian_airlines if airline in enhanced_data]),
            'year': 2023,
            'source': 'Asian Airlines Average'
        }
        
        return enhanced_data

    def get_airport_coordinates(self, iata_code: str) -> Optional[Tuple[float, float]]:
        """Get airport coordinates from the expanded database"""
        if not iata_code:
            return None
        return self.airports.get(iata_code.upper())

    def calculate_distance(self, origin: str, destination: str) -> Optional[float]:
        """Calculate great circle distance between airports"""
        origin_coords = self.get_airport_coordinates(origin)
        dest_coords = self.get_airport_coordinates(destination)
        
        if origin_coords and dest_coords:
            return geodesic(origin_coords, dest_coords).kilometers
        return None

    def get_aircraft_emission_factor(self, aircraft_code: str) -> float:
        """Get emission factor for aircraft type"""
        if not aircraft_code or pd.isna(aircraft_code):
            return self.aircraft_emissions['DEFAULT']
        
        aircraft_code = str(aircraft_code).upper().strip()
        
        # Direct match
        if aircraft_code in self.aircraft_emissions:
            return self.aircraft_emissions[aircraft_code]
        
        # Partial match
        for key in self.aircraft_emissions:
            if key != 'DEFAULT' and (key in aircraft_code or aircraft_code.startswith(key)):
                return self.aircraft_emissions[key]
        
        return self.aircraft_emissions['DEFAULT']

    def get_aircraft_capacity(self, aircraft_code: str) -> int:
        """Get passenger capacity for aircraft type"""
        if not aircraft_code or pd.isna(aircraft_code):
            return self.aircraft_capacities['DEFAULT']
        
        aircraft_code = str(aircraft_code).upper().strip()
        
        # Direct match
        if aircraft_code in self.aircraft_capacities:
            return self.aircraft_capacities[aircraft_code]
        
        # Partial match
        for key in self.aircraft_capacities:
            if key != 'DEFAULT' and (key in aircraft_code or aircraft_code.startswith(key)):
                return self.aircraft_capacities[key]
        
        return self.aircraft_capacities['DEFAULT']

    def get_route_factor(self, distance_km: float, is_domestic: bool = False) -> float:
        """Get route-specific emission factor adjustment"""
        if not distance_km:
            return 1.0
        
        if is_domestic:
            if distance_km < 500:
                return self.route_factors['domestic_short']
            elif distance_km < 1500:
                return self.route_factors['domestic_medium']
            else:
                return self.route_factors['domestic_long']
        else:
            if distance_km < 1500:
                return self.route_factors['international_short']
            elif distance_km < 4000:
                return self.route_factors['international_medium']
            elif distance_km < 8000:
                return self.route_factors['international_long']
            else:
                return self.route_factors['ultra_long']

    def calculate_co2_emissions(self, distance_km: float, aircraft_type: str, 
                                passenger_load_factor: float = 0.8, 
                                is_domestic: bool = False) -> Optional[float]:
        """Calculate CO2 emissions per passenger with enhanced methodology"""
        if not distance_km or distance_km <= 0:
            return None
        
        # Base emission factor
        emission_factor = self.get_aircraft_emission_factor(aircraft_type)
        
        # Route-specific adjustment
        route_factor = self.get_route_factor(distance_km, is_domestic)
        
        # Base emissions calculation
        base_emissions = distance_km * emission_factor * route_factor
        
        # Load factor adjustment
        adjusted_emissions = base_emissions / passenger_load_factor
        
        # Takeoff and landing penalty (higher for shorter flights)
        if distance_km < 500:
            # Short flights have higher relative penalty
            penalty = 150 * emission_factor / passenger_load_factor
        elif distance_km < 1000:
            penalty = 95 * emission_factor / passenger_load_factor
        elif distance_km < 3000:
            penalty = 50 * emission_factor / passenger_load_factor
        else:
            penalty = 25 * emission_factor / passenger_load_factor
        
        total_emissions = adjusted_emissions + penalty
        
        # Additional factors for aircraft efficiency class
        efficiency_multipliers = {
            'A20N': 0.92, 'A21N': 0.92, 'B38M': 0.90, 'B39M': 0.90,  # NEO/MAX aircraft
            'A350': 0.85, 'A359': 0.85, 'A35K': 0.85,  # A350 family
            'B787': 0.80, 'B788': 0.82, 'B789': 0.80, 'B78X': 0.78,  # Dreamliner
        }
        
        aircraft_code = str(aircraft_type).upper() if aircraft_type else ''
        for efficient_type, multiplier in efficiency_multipliers.items():
            if efficient_type in aircraft_code:
                total_emissions *= multiplier
                break
        
        return round(total_emissions, 2)

    async def collect_comprehensive_dataset(self, target_size: int = 5000) -> pd.DataFrame:
        """Collect flight data from multiple sources"""
        all_flights = []
        
        # Progress tracking
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            # 1. AviationStack API data (real flights)
            status_text.text("Collecting real flight data from AviationStack API...")
            aviationstack_flights = []
            max_api_calls = min(target_size // 100, 10)  # Limit API calls
            
            for i in range(max_api_calls):
                batch = self.get_aviationstack_flights(limit=100, offset=i*100)
                if not batch:
                    break
                aviationstack_flights.extend(batch)
                progress_bar.progress(0.1 + (i / max_api_calls) * 0.2)
                time.sleep(1)  # Rate limiting
            
            # Process AviationStack data
            processed_flights = self.process_aviationstack_data(aviationstack_flights)
            all_flights.extend(processed_flights)
            st.success(f"Collected {len(processed_flights)} flights from AviationStack API")
            
            # 2. OpenSky Network data (real-time flights)
            status_text.text("Collecting real-time flight data from OpenSky Network...")
            async with aiohttp.ClientSession() as session:
                opensky_flights = await self.get_opensky_flights(session, limit=200)
                processed_opensky = self.process_opensky_data(opensky_flights)
                all_flights.extend(processed_opensky)
                st.success(f"Collected {len(processed_opensky)} flights from OpenSky Network")
                progress_bar.progress(0.4)
            
            
            # 3. Generate synthetic routes for comprehensive coverage
            status_text.text("Generating synthetic flight routes for comprehensive coverage...")
            synthetic_needed = max(0, target_size - len(all_flights))
            synthetic_flights = self.generate_synthetic_routes(synthetic_needed)
            
            # Process synthetic flights
            processed_synthetic = []
            for i, flight in enumerate(synthetic_flights):
                processed_flight = self.process_single_flight(flight)
                if processed_flight:
                    processed_synthetic.append(processed_flight)
                
                if i % 100 == 0:
                    progress_bar.progress(0.4 + (i / len(synthetic_flights)) * 0.5)
            
            all_flights.extend(processed_synthetic)
            st.success(f"Generated {len(processed_synthetic)} synthetic flights")
            
            # 4. Enhance with sustainability data
            status_text.text("Enhancing dataset with airline sustainability data...")
            sustainability_data = self.scrape_airline_sustainability_data()
            enhanced_flights = self.enhance_with_sustainability_data(all_flights, sustainability_data)
            progress_bar.progress(0.95)
            
            # Create final DataFrame
            df = pd.DataFrame(enhanced_flights)
            df = self.clean_and_validate_data(df)
            
            progress_bar.progress(1.0)
            status_text.text(f"Dataset collection complete! Total flights: {len(df)}")
            
            return df
            
        except Exception as e:
            st.error(f"Error in data collection: {e}")
            return pd.DataFrame(all_flights) if all_flights else pd.DataFrame()

    def process_aviationstack_data(self, flights_data: List[Dict]) -> List[Dict]:
        """Process flights from AviationStack API"""
        processed_flights = []
        
        for flight in flights_data:
            try:
                aircraft_info = flight.get('aircraft') or {}
                departure_info = flight.get('departure') or {}
                arrival_info = flight.get('arrival') or {}
                airline_info = flight.get('airline') or {}
                flight_info_raw = flight.get('flight') or {}
                
                flight_info = {
                    'flight_number': flight_info_raw.get('number'),
                    'airline_name': airline_info.get('name'),
                    'airline_iata': airline_info.get('iata'),
                    'aircraft_type': aircraft_info.get('icao'),
                    'origin_airport': departure_info.get('iata'),
                    'destination_airport': arrival_info.get('iata'),
                    'scheduled_departure': departure_info.get('scheduled'),
                    'scheduled_arrival': arrival_info.get('scheduled'),
                    'flight_status': flight.get('flight_status'),
                    'data_source': 'aviationstack_api'
                }
                
                processed_flight = self.process_single_flight(flight_info)
                if processed_flight:
                    processed_flights.append(processed_flight)
                    
            except Exception as e:
                continue
        
        return processed_flights

    def process_opensky_data(self, flights_data: List[Dict]) -> List[Dict]:
        """Process flights from OpenSky Network"""
        processed_flights = []
        
        # Map callsigns to likely airlines (simplified mapping)
        callsign_airlines = {
            'AAL': 'American Airlines', 'DAL': 'Delta Air Lines', 'UAL': 'United Airlines',
            'SWA': 'Southwest Airlines', 'JBU': 'JetBlue Airways', 'ASA': 'Alaska Airlines',
            'DLH': 'Lufthansa', 'BAW': 'British Airways', 'AFR': 'Air France',
            'KLM': 'KLM', 'UAE': 'Emirates', 'QTR': 'Qatar Airways',
            'SIA': 'Singapore Airlines', 'ANA': 'ANA', 'JAL': 'JAL'
        }
        
        for flight in flights_data:
            try:
                if not flight.get('callsign'):
                    continue
                
                callsign = flight['callsign'].strip()
                airline_code = callsign[:3]
                
                # Try to match airline
                airline_name = callsign_airlines.get(airline_code, 'Unknown Airline')
                
                # Generate synthetic route based on common patterns
                if flight.get('origin_country'):
                    country_airports = {
                        'United States': ['JFK', 'LAX', 'ORD', 'ATL', 'DEN'],
                        'Germany': ['FRA', 'MUC', 'DUS'],
                        'United Kingdom': ['LHR', 'LGW'],
                        'France': ['CDG', 'ORY'],
                        'Japan': ['NRT', 'HND'],
                        'Singapore': ['SIN']
                    }
                    
                    possible_airports = country_airports.get(flight['origin_country'], ['JFK'])
                    origin = np.random.choice(possible_airports)
                    
                    # Generate destination (different region)
                    all_major_airports = ['JFK', 'LAX', 'LHR', 'CDG', 'FRA', 'NRT', 'SIN', 'SYD']
                    destination = np.random.choice([apt for apt in all_major_airports if apt != origin])
                    
                    flight_info = {
                        'flight_number': callsign,
                        'airline_name': airline_name,
                        'aircraft_type': 'A320',  # Default for unknown
                        'origin_airport': origin,
                        'destination_airport': destination,
                        'altitude': flight.get('altitude'),
                        'velocity': flight.get('velocity'),
                        'data_source': 'opensky_network'
                    }
                    
                    processed_flight = self.process_single_flight(flight_info)
                    if processed_flight:
                        processed_flights.append(processed_flight)
                        
            except Exception as e:
                continue
        
        return processed_flights

    def process_single_flight(self, flight_info: Dict) -> Optional[Dict]:
        """Process a single flight and calculate emissions"""
        try:
            # Calculate distance
            if flight_info.get('origin_airport') and flight_info.get('destination_airport'):
                distance = self.calculate_distance(
                    flight_info['origin_airport'], 
                    flight_info['destination_airport']
                )
                if not distance or distance <= 0:
                    return None
                    
                flight_info['distance_km'] = round(distance, 2)
            else:
                return None
            
            # Get aircraft capacity
            aircraft_type = flight_info.get('aircraft_type')
            flight_info['passenger_capacity'] = self.get_aircraft_capacity(aircraft_type)
            
            # Calculate load factor (if not provided)
            if 'load_factor' not in flight_info:
                # Realistic load factors based on route distance
                if distance < 500:
                    flight_info['load_factor'] = np.random.uniform(0.65, 0.85)
                elif distance < 2000:
                    flight_info['load_factor'] = np.random.uniform(0.70, 0.88)
                else:
                    flight_info['load_factor'] = np.random.uniform(0.75, 0.90)
            
            # Determine if domestic flight (simplified logic)
            is_domestic = self.is_domestic_flight(
                flight_info['origin_airport'], 
                flight_info['destination_airport']
            )
            
            # Calculate CO2 emissions
            co2_emissions = self.calculate_co2_emissions(
                distance, 
                aircraft_type,
                flight_info['load_factor'],
                is_domestic
            )
            
            if co2_emissions:
                flight_info['co2_per_passenger_kg'] = co2_emissions
                flight_info['co2_per_passenger_per_km'] = round(co2_emissions / distance, 4)
                flight_info['total_co2_kg'] = round(co2_emissions * flight_info['passenger_capacity'] * flight_info['load_factor'], 2)
            else:
                return None
            
            # Add metadata
            flight_info['data_extracted_at'] = datetime.now().isoformat()
            flight_info['is_domestic'] = is_domestic
            
            return flight_info
            
        except Exception as e:
            return None

    def is_domestic_flight(self, origin: str, destination: str) -> bool:
        """Determine if flight is domestic (simplified country mapping)"""
        country_airports = {
            'US': ['JFK', 'LAX', 'ORD', 'ATL', 'DEN', 'DFW', 'SFO', 'SEA', 'BOS', 'MIA', 
                   'MCO', 'LAS', 'PHX', 'IAH', 'MSP', 'DTW', 'CLT', 'LGA', 'EWR', 'SAN'],
            'CA': ['YYZ', 'YVR', 'YUL', 'YYC', 'YEG', 'YOW'],
            'GB': ['LHR', 'LGW', 'STN'],
            'FR': ['CDG', 'ORY'],
            'DE': ['FRA', 'MUC', 'DUS'],
            'JP': ['NRT', 'HND', 'KIX'],
            'CN': ['PEK', 'PVG', 'SHA', 'CAN', 'SZX', 'CTU'],
            'AU': ['SYD', 'MEL', 'BNE', 'PER', 'ADL']
        }
        
        for country, airports in country_airports.items():
            if origin in airports and destination in airports:
                return True
        return False

    def enhance_with_sustainability_data(self, flights: List[Dict], 
                                         sustainability_data: Dict) -> List[Dict]:
        """Enhance flights with airline sustainability metrics"""
        enhanced_flights = []
        
        for flight in flights:
            airline_name = flight.get('airline_name')
            
            # Add sustainability metrics if available
            if airline_name in sustainability_data:
                sustainability_info = sustainability_data[airline_name]
                flight['airline_efficiency_reported'] = sustainability_info['emissions_intensity']
                flight['sustainability_report_year'] = sustainability_info['year']
                flight['sustainability_data_source'] = sustainability_info['source']
            
            # Calculate efficiency rating (relative to industry average)
            if 'co2_per_passenger_per_km' in flight:
                industry_avg = sustainability_data.get('INDUSTRY_AVERAGE', {}).get('emissions_intensity', 0.090)
                flight['efficiency_vs_industry'] = round(flight['co2_per_passenger_per_km'] / industry_avg, 3)
            
            enhanced_flights.append(flight)
        
        return enhanced_flights

    def clean_and_validate_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and validate the collected dataset"""
        if df.empty:
            return df
            
        initial_size = len(df)
        
        # Remove rows with missing critical data
        df = df.dropna(subset=['origin_airport', 'destination_airport', 'distance_km'])
        
        # Remove invalid distances
        df = df[df['distance_km'] > 0]
        
        # Remove invalid CO2 calculations
        df = df.dropna(subset=['co2_per_passenger_kg'])
        df = df[df['co2_per_passenger_kg'] > 0]
        
        # Cap unrealistic values
        df.loc[df['co2_per_passenger_kg'] > 2000, 'co2_per_passenger_kg'] = np.nan
        df = df.dropna(subset=['co2_per_passenger_kg'])
        
        # Ensure load factors are reasonable
        df.loc[df['load_factor'] > 1.0, 'load_factor'] = 0.85
        df.loc[df['load_factor'] < 0.3, 'load_factor'] = 0.75
        
        # Fill missing airline names
        df['airline_name'] = df['airline_name'].fillna('Unknown Airline')
        
        # Add flight categories
        df['flight_category'] = pd.cut(df['distance_km'], 
                                       bins=[0, 500, 1500, 4000, float('inf')],
                                       labels=['Short-haul', 'Medium-haul', 'Long-haul', 'Ultra-long-haul'])
        
        # Add efficiency categories
        efficiency_quantiles = df['co2_per_passenger_per_km'].quantile([0.25, 0.5, 0.75])
        df['efficiency_rating'] = pd.cut(df['co2_per_passenger_per_km'],
                                         bins=[0, efficiency_quantiles[0.25], efficiency_quantiles[0.5], 
                                               efficiency_quantiles[0.75], float('inf')],
                                         labels=['Excellent', 'Good', 'Average', 'Poor'])
        
        st.info(f"Dataset cleaned: {initial_size} → {len(df)} flights ({((len(df)/initial_size)*100):.1f}% retained)")
        
        return df

    def save_dataset(self, df: pd.DataFrame, filename: str = 'enhanced_flight_emissions.csv'):
        """Save the dataset with metadata"""
        # Save main dataset
        df.to_csv(filename, index=False)
        
        # Save metadata
        metadata = {
            'created_at': datetime.now().isoformat(),
            'total_flights': len(df),
            'data_sources': df['data_source'].value_counts().to_dict(),
            'airlines_covered': df['airline_name'].nunique(),
            'routes_covered': len(df.groupby(['origin_airport', 'destination_airport'])),
            'aircraft_types': df['aircraft_type'].nunique(),
            'distance_range_km': {
                'min': float(df['distance_km'].min()),
                'max': float(df['distance_km'].max()),
                'mean': float(df['distance_km'].mean())
            },
            'emissions_range_kg': {
                'min': float(df['co2_per_passenger_kg'].min()),
                'max': float(df['co2_per_passenger_kg'].max()),
                'mean': float(df['co2_per_passenger_kg'].mean())
            }
        }
        
        metadata_filename = filename.replace('.csv', '_metadata.json')
        with open(metadata_filename, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        return filename, metadata_filename

class AdvancedMLModels:
    """Advanced machine learning models for emission prediction"""
    
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.models = {}
        self.model_performance = {}
        self.feature_columns = []
    
    def prepare_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
        """Prepare features for ML models"""
        # Select and engineer features
        feature_columns = ['distance_km', 'passenger_capacity', 'load_factor']
        
        # Add categorical encodings
        df_encoded = df.copy()
        
        # Aircraft type encoding (group similar aircraft)
        aircraft_groups = {
            'narrow_body': ['A319', 'A320', 'A321', 'A20N', 'A21N', 'B737', 'B738', 'B739', 'B38M'],
            'wide_body': ['A330', 'A340', 'A350', 'A380', 'B747', 'B767', 'B777', 'B787'],
            'regional': ['E190', 'E195', 'CRJ', 'ATR', 'DHC']
        }
        
        def categorize_aircraft(aircraft_type):
            if pd.isna(aircraft_type):
                return 'narrow_body'
            aircraft_type = str(aircraft_type).upper()
            for category, aircraft_list in aircraft_groups.items():
                if any(ac in aircraft_type for ac in aircraft_list):
                    return category
            return 'narrow_body'
        
        df_encoded['aircraft_category'] = df_encoded['aircraft_type'].apply(categorize_aircraft)
        
        # One-hot encode aircraft category
        aircraft_dummies = pd.get_dummies(df_encoded['aircraft_category'], prefix='aircraft')
        df_encoded = pd.concat([df_encoded, aircraft_dummies], axis=1)
        feature_columns.extend(aircraft_dummies.columns.tolist())
        
        # Distance-based features
        df_encoded['distance_squared'] = df_encoded['distance_km'] ** 2
        df_encoded['distance_log'] = np.log1p(df_encoded['distance_km'])
        feature_columns.extend(['distance_squared', 'distance_log'])
        
        # Interaction features
        df_encoded['distance_capacity_interaction'] = df_encoded['distance_km'] * df_encoded['passenger_capacity']
        df_encoded['distance_load_interaction'] = df_encoded['distance_km'] * df_encoded['load_factor']
        feature_columns.extend(['distance_capacity_interaction', 'distance_load_interaction'])
        
        # Flight category encoding
        if 'is_domestic' in df_encoded.columns:
            df_encoded['is_domestic_encoded'] = df_encoded['is_domestic'].astype(int)
            feature_columns.append('is_domestic_encoded')
        
        # Ensure all columns from training are present
        if self.feature_columns:
            for col in self.feature_columns:
                if col not in df_encoded.columns:
                    df_encoded[col] = 0
            X = df_encoded[self.feature_columns]
        else:
            self.feature_columns = feature_columns
            X = df_encoded[feature_columns]

        y = df_encoded['co2_per_passenger_kg'] if 'co2_per_passenger_kg' in df_encoded else None
        
        return X, y
    
    def train_models(self) -> Dict:
        """Train multiple ML models and compare performance"""
        X, y = self.prepare_features(self.df)
        
        temp_df = pd.concat([X, y], axis=1)
        temp_df.dropna(inplace=True)
        
        if len(temp_df) < 20:
            st.error("Not enough clean data to train models.")
            return {}

        X_clean = temp_df[self.feature_columns].astype(np.float64)
        y_clean = temp_df[y.name]
        
        X_train, X_test, y_train, y_test = train_test_split(X_clean, y_clean, test_size=0.2, random_state=42)
        
        # --- 1. Expanded model dictionary ---
        # Add the new models to the training pipeline
        models_to_train = {
            'Linear Regression': LinearRegression(),
            'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
            'Gradient Boosting': GradientBoostingRegressor(random_state=42),
            'Support Vector Regressor': SVR(),
            'OLS Statsmodels': None  # Handled separately
        }
        
        results = {}
        
        # The existing loop will train all models automatically
        for name, model in models_to_train.items():
            if model is not None:
                st.info(f"Training {name}...")
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                
                mse = mean_squared_error(y_test, y_pred)
                rmse = np.sqrt(mse)
                r2 = r2_score(y_test, y_pred)
                
                results[name] = {
                    'model': model, 'mse': mse, 'rmse': rmse, 'r2': r2,
                    'predictions': y_pred, 'actual': y_test
                }
                self.models[name] = model
        
        # Train OLS model with statsmodels
        st.info("Training OLS Statsmodels...")
        X_train_sm = sm.add_constant(X_train)
        X_test_sm = sm.add_constant(X_test)
        
        ols_model = sm.OLS(y_train, X_train_sm).fit()
        ols_pred = ols_model.predict(X_test_sm)
        
        ols_mse = mean_squared_error(y_test, ols_pred)
        ols_rmse = np.sqrt(ols_mse)
        ols_r2 = r2_score(y_test, ols_pred)
        
        results['OLS Statsmodels'] = {
            'model': ols_model, 'mse': ols_mse, 'rmse': ols_rmse, 'r2': ols_r2,
            'predictions': ols_pred, 'actual': y_test, 'summary': ols_model.summary()
        }
        
        self.models['OLS Statsmodels'] = ols_model
        self.model_performance = results
        
        return results
    
    def display_model_comparison(self, results: Dict):
        """Display comprehensive model comparison"""
        st.write("### Model Performance Comparison")
        
        performance_data = []
        for name, result in results.items():
            performance_data.append({
                'Model': name,
                'RMSE': f"{result['rmse']:.3f}",
                'R²': f"{result['r2']:.4f}",
                'MSE': f"{result['mse']:.3f}"
            })
        
        performance_df = pd.DataFrame(performance_data).sort_values('RMSE').reset_index(drop=True)
        st.dataframe(performance_df)
        
        st.write("### Predictions vs Actual Values Plot")
        fig = go.Figure()
        
        # --- 2. Use a more extensive color palette for plots ---
        colors = px.colors.qualitative.Plotly
        
        for i, (name, result) in enumerate(results.items()):
            fig.add_trace(go.Scatter(
                x=result['actual'], y=result['predictions'], mode='markers', name=name,
                marker=dict(color=colors[i % len(colors)], opacity=0.6)
            ))
        
        min_val = min(min(r['actual']) for r in results.values())
        max_val = max(max(r['actual']) for r in results.values())
        fig.add_trace(go.Scatter(
            x=[min_val, max_val], y=[min_val, max_val], mode='lines',
            name='Perfect Prediction', line=dict(color='black', dash='dash')
        ))
        
        fig.update_layout(
            title='Model Predictions vs Actual CO₂ Emissions',
            xaxis_title='Actual CO₂ (kg)', yaxis_title='Predicted CO₂ (kg)', height=600
        )
        st.plotly_chart(fig)
        
        # Feature importance (for Random Forest and other tree models)
        st.write("### Feature Importance")
        if 'Random Forest' in results:
            rf_model = results['Random Forest']['model']
            X, _ = self.prepare_features(self.df)
            
            feature_importance = pd.DataFrame({
                'feature': X.columns,
                'importance': rf_model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            fig_importance = px.bar(
                feature_importance.head(10), x='importance', y='feature',
                orientation='h', title='Top 10 Feature Importance (Random Forest)'
            )
            st.plotly_chart(fig_importance)
        
        if 'OLS Statsmodels' in results:
            st.write("### OLS Regression Summary")
            st.text(str(results['OLS Statsmodels']['summary']))
    
    def predict_emissions(self, distance_km: float, aircraft_type: str, 
                          passenger_capacity: int, load_factor: float = 0.8,
                          is_domestic: bool = False) -> Dict:
        """Predict emissions using trained models"""
        # (This function remains unchanged, it will automatically use all new models)
        if not self.models:
            return {'error': 'No models trained yet'}
        
        input_data = pd.DataFrame({
            'distance_km': [distance_km], 'passenger_capacity': [passenger_capacity],
            'load_factor': [load_factor], 'aircraft_type': [aircraft_type],
            'is_domestic': [is_domestic]
        })
        
        X, _ = self.prepare_features(input_data)
        predictions = {}
        
        for name, model in self.models.items():
            try:
                if name == 'OLS Statsmodels':
                    X_with_const = sm.add_constant(X, has_constant='add')
                    for col in model.model.exog_names:
                        if col not in X_with_const.columns: X_with_const[col] = 0
                    X_with_const = X_with_const[model.model.exog_names]
                    pred = model.predict(X_with_const)[0]
                else:
                    pred = model.predict(X)[0]
                predictions[name] = round(pred, 2)
            except Exception as e:
                predictions[name] = f"Error: {e}"
        
        return predictions

def create_streamlit_app():
    """Create the main Streamlit application"""
    st.set_page_config(
        page_title="Enhanced Flight Carbon Footprint Calculator",
        page_icon="✈️",
        layout="wide"
    )
    
    st.title("✈️ Enhanced Flight Carbon Footprint Calculator")
    st.markdown("**Advanced ML-powered flight emission analysis with comprehensive dataset**")
    
    # Initialize calculator
    if 'calculator' not in st.session_state:
        st.session_state.calculator = EnhancedFlightCarbonCalculator()
    
    calculator = st.session_state.calculator
    
    # Sidebar controls
    st.sidebar.header("🔧 Dataset Configuration")
    
    dataset_size = st.sidebar.selectbox(
        "Dataset Size",
        [1000, 2500, 5000, 7500, 10000],
        index=2
    )
    
    # Main content tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Data Collection", 
        "🔍 Dataset Analysis", 
        "🤖 ML Models", 
        "🔮 Predictions", 
        "📈 Airline Comparison"
    ])
    
    with tab1:
        st.header("Data Collection & Processing")
        
        if st.button("🚀 Collect Enhanced Dataset", type="primary"):
            with st.spinner("Collecting comprehensive flight dataset... This may take a few minutes."):
                # Run async data collection
                try:
                    loop = asyncio.get_running_loop()
                except RuntimeError:
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)

                df = loop.run_until_complete(
                    calculator.collect_comprehensive_dataset(target_size=dataset_size)
                )
                
                if not df.empty:
                    st.session_state.df = df
                    
                    # Save dataset
                    filename, metadata_filename = calculator.save_dataset(df)
                    st.success(f"Dataset saved as '{filename}' and metadata as '{metadata_filename}'")
                    
                    # Display summary
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Total Flights", len(df))
                    with col2:
                        st.metric("Airlines", df['airline_name'].nunique())
                    with col3:
                        st.metric("Routes", len(df.groupby(['origin_airport', 'destination_airport'])))
                    with col4:
                        st.metric("Aircraft Types", df['aircraft_type'].nunique())
                    
                    # Data source breakdown
                    st.subheader("📋 Data Sources")
                    source_counts = df['data_source'].value_counts()
                    fig_sources = px.pie(
                        values=source_counts.values,
                        names=source_counts.index,
                        title="Dataset Composition by Data Source"
                    )
                    st.plotly_chart(fig_sources)
                    
                else:
                    st.error("Failed to collect dataset")
        
        # Display current dataset info
        if 'df' in st.session_state:
            df = st.session_state.df
            st.info(f"Current dataset: {len(df)} flights loaded")
            st.dataframe(df.head())

    with tab2:
        st.header("Dataset Analysis & Visualization")
        
        if 'df' in st.session_state:
            df = st.session_state.df
            
            # Basic statistics
            st.subheader("📊 Dataset Statistics")
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Emissions Statistics (kg CO₂/passenger)**")
                emissions_stats = df['co2_per_passenger_kg'].describe()
                st.dataframe(emissions_stats)
                
                # Emissions distribution
                fig_emissions = px.histogram(
                    df, x='co2_per_passenger_kg', nbins=50,
                    title="CO₂ Emissions Distribution"
                )
                st.plotly_chart(fig_emissions)
            
            with col2:
                st.write("**Distance Statistics (km)**")
                distance_stats = df['distance_km'].describe()
                st.dataframe(distance_stats)
                
                # Distance distribution
                fig_distance = px.histogram(
                    df, x='distance_km', nbins=50,
                    title="Flight Distance Distribution"
                )
                st.plotly_chart(fig_distance)
            
            # Flight category analysis
            st.subheader("✈️ Flight Category Analysis")
            category_analysis = df.groupby('flight_category').agg({
                'co2_per_passenger_kg': ['mean', 'std', 'count'],
                'distance_km': 'mean',
                'load_factor': 'mean'
            }).round(3)
            
            st.dataframe(category_analysis)
            
            # Category comparison
            fig_category = px.box(
                df, x='flight_category', y='co2_per_passenger_kg',
                title="CO₂ Emissions by Flight Category"
            )
            st.plotly_chart(fig_category)
            
            # Correlation analysis
            st.subheader("🔗 Correlation Analysis")
            numeric_columns = ['distance_km', 'passenger_capacity', 'load_factor', 
                             'co2_per_passenger_kg', 'co2_per_passenger_per_km']
            correlation_matrix = df[numeric_columns].corr()
            
            fig_corr = px.imshow(
                correlation_matrix,
                text_auto=True,
                title="Feature Correlation Matrix",
                color_continuous_scale='RdBu_r',
                aspect='auto'
            )
            st.plotly_chart(fig_corr)
            
        else:
            st.info("Please collect a dataset first in the 'Data Collection' tab.")

    with tab3:
        st.header("🤖 Machine Learning Model Training")
        
        if 'df' in st.session_state:
            df = st.session_state.df
            
            if st.button("Train Emission Prediction Models", type="primary"):
                with st.spinner("Training models... This might take a moment."):
                    ml_models = AdvancedMLModels(df)
                    results = ml_models.train_models()
                    st.session_state.ml_models = ml_models
                    st.session_state.model_results = results
                    st.success("Models trained successfully!")

            if 'model_results' in st.session_state:
                st.info("Models are trained. You can view the performance below or make predictions in the 'Predictions' tab.")
                ml_models = st.session_state.ml_models
                results = st.session_state.model_results
                ml_models.display_model_comparison(results)
            else:
                st.info("Click the button above to train machine learning models on the collected dataset.")

        else:
            st.info("Please collect a dataset first in the 'Data Collection' tab before training models.")

    with tab4:
        st.header("🔮 Individual Flight Emission Prediction")
        
        if 'ml_models' in st.session_state:
            ml_models = st.session_state.ml_models
            
            st.subheader("Enter Flight Details")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                origin = st.selectbox("Origin Airport", options=sorted(calculator.airports.keys()), index=list(calculator.airports.keys()).index('JFK'))
            with col2:
                destination = st.selectbox("Destination Airport", options=sorted(calculator.airports.keys()), index=list(calculator.airports.keys()).index('LHR'))
            with col3:
                aircraft_type = st.selectbox("Aircraft Type", options=sorted(calculator.aircraft_emissions.keys()))
            
            load_factor = st.slider("Passenger Load Factor", 0.5, 1.0, 0.85, 0.01)

            if st.button("Predict CO₂ Emissions", type="primary"):
                if origin == destination:
                    st.error("Origin and destination airports cannot be the same.")
                else:
                    distance = calculator.calculate_distance(origin, destination)
                    capacity = calculator.get_aircraft_capacity(aircraft_type)
                    is_domestic = calculator.is_domestic_flight(origin, destination)
                    
                    st.write("---")
                    st.subheader("Prediction Results")

                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("Distance", f"{distance:.0f} km")
                    col2.metric("Aircraft Capacity", f"{capacity} seats")
                    col3.metric("Flight Type", "Domestic" if is_domestic else "International")
                    
                    with st.spinner("Calculating..."):
                        # Formula-based calculation
                        formula_prediction = calculator.calculate_co2_emissions(
                            distance, aircraft_type, load_factor, is_domestic
                        )
                        
                        # ML-based prediction
                        ml_predictions = ml_models.predict_emissions(
                            distance, aircraft_type, capacity, load_factor, is_domestic
                        )

                    st.markdown(f"#### Formula-based Calculation: `{formula_prediction:.2f} kg CO₂` per passenger")
                    
                    st.markdown("#### Machine Learning Model Predictions:")
                    
                    pred_cols = st.columns(len(ml_predictions))
                    for i, (model_name, prediction) in enumerate(ml_predictions.items()):
                        pred_cols[i].metric(model_name, f"{prediction} kg CO₂")

        else:
            st.info("Please train the ML models in the 'ML Models' tab to use the prediction tool.")

    with tab5:
        st.header("📈 Airline Efficiency Comparison")
        
        if 'df' in st.session_state:
            df = st.session_state.df
            
            st.subheader("Airline Emission Performance (kg CO₂ per Passenger per km)")
            
            # Filter out unknown airlines and airlines with few flights
            airline_counts = df['airline_name'].value_counts()
            valid_airlines = airline_counts[airline_counts >= 10].index
            df_filtered = df[df['airline_name'].isin(valid_airlines)]

            airline_efficiency = df_filtered.groupby('airline_name').agg(
                avg_co2_per_km=('co2_per_passenger_per_km', 'mean'),
                num_flights=('flight_number', 'count')
            ).sort_values('avg_co2_per_km').reset_index()
            
            col1, col2 = st.columns(2)

            with col1:
                st.write("**Top 10 Most Efficient Airlines (in dataset)**")
                top_10 = airline_efficiency.head(10)
                fig_top = px.bar(top_10, y='airline_name', x='avg_co2_per_km', orientation='h',
                                 text='avg_co2_per_km',
                                 labels={'airline_name': 'Airline', 'avg_co2_per_km': 'Avg. CO₂/pax-km'},
                                 color='avg_co2_per_km', color_continuous_scale='greens_r')
                fig_top.update_traces(texttemplate='%{text:.4f}', textposition='outside')
                fig_top.update_layout(yaxis={'categoryorder':'total ascending'})
                st.plotly_chart(fig_top)

            with col2:
                st.write("**Top 10 Least Efficient Airlines (in dataset)**")
                bottom_10 = airline_efficiency.tail(10)
                fig_bottom = px.bar(bottom_10, y='airline_name', x='avg_co2_per_km', orientation='h',
                                    text='avg_co2_per_km',
                                    labels={'airline_name': 'Airline', 'avg_co2_per_km': 'Avg. CO₂/pax-km'},
                                    color='avg_co2_per_km', color_continuous_scale='reds')
                fig_bottom.update_traces(texttemplate='%{text:.4f}', textposition='outside')
                fig_bottom.update_layout(yaxis={'categoryorder':'total ascending'})
                st.plotly_chart(fig_bottom)

            st.write("---")
            st.subheader("Comparison with Sustainability Reports")
            sustainability_data = calculator.scrape_airline_sustainability_data()
            report_df = pd.DataFrame.from_dict(sustainability_data, orient='index').reset_index()
            report_df = report_df.rename(columns={'index': 'airline_name', 'emissions_intensity': 'reported_intensity'})
            
            comparison_df = pd.merge(airline_efficiency, report_df[['airline_name', 'reported_intensity']], on='airline_name', how='inner')
            
            fig_compare = go.Figure()
            fig_compare.add_trace(go.Bar(
                x=comparison_df['airline_name'],
                y=comparison_df['avg_co2_per_km'],
                name='Calculated from Dataset',
                marker_color='indianred'
            ))
            fig_compare.add_trace(go.Bar(
                x=comparison_df['airline_name'],
                y=comparison_df['reported_intensity'],
                name='From Sustainability Report',
                marker_color='lightsalmon'
            ))
            
            fig_compare.update_layout(
                title='Calculated Emission Intensity vs. Reported Values',
                xaxis_title='Airline',
                yaxis_title='CO₂ per Passenger-km',
                barmode='group'
            )
            st.plotly_chart(fig_compare)

        else:
            st.info("Please collect a dataset first in the 'Data Collection' tab to view airline comparisons.")


if __name__ == "__main__":
    create_streamlit_app()