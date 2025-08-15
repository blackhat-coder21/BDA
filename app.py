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

warnings.filterwarnings('ignore')

class FlightCarbonCalculator:
    def __init__(self, aviationstack_api_key):
        self.aviationstack_key = aviationstack_api_key
        self.base_url = "https://api.aviationstack.com/v1"
        
        # Aircraft emissions data (kg CO2 per km per passenger) - expanded with more types
        self.aircraft_emissions = {
            'A320': 0.095, 'A321': 0.092, 'A319': 0.105,
            'A20N': 0.085, 'A21N': 0.082,
            'B737': 0.098, '737': 0.098, 'B738': 0.096, 'B739': 0.094,
            'B38M': 0.086, 'B39M': 0.084,
            'E190': 0.110, 'E195': 0.108,
            'A330': 0.085, 'A340': 0.095, 'A350': 0.075, 'A359': 0.075, 'A35K': 0.075,
            'A380': 0.082,
            'B747': 0.090, '747': 0.090, 'B767': 0.088, 'B777': 0.083,
            'B787': 0.070, '787': 0.070, 'B788': 0.072, 'B789': 0.070, 'B78X': 0.068,
            'CRJ': 0.130, 'ATR': 0.120, 'DHC': 0.140,
            'DEFAULT': 0.095
        }
        
        # Aircraft passenger capacities (average seats)
        self.aircraft_capacities = {
            'A320': 180, 'A20N': 180,
            'A321': 220, 'A21N': 220,
            'A319': 150,
            'B737': 175, 'B738': 189, 'B739': 215, 'B38M': 189, 'B39M': 215,
            'E190': 100, 'E195': 120,
            'A330': 300, 'A333': 300,
            'A340': 320,
            'A350': 325, 'A359': 325, 'A35K': 350,
            'A380': 525,
            'B747': 410,
            'B767': 240,
            'B777': 350, 'B77W': 365,
            'B787': 300, 'B788': 250, 'B789': 300, 'B78X': 330,
            'CRJ': 80, 'ATR': 70, 'DHC': 50,
            'DEFAULT': 200
        }
        
        # Airport coordinates (expanded list of major airports worldwide)
        self.airports = {
            'JFK': (40.6413, -73.7781), 'LAX': (33.9425, -118.4081), 'ATL': (33.6407, -84.4277),
            'ORD': (41.9742, -87.9073), 'DEN': (39.8561, -104.6737), 'DFW': (32.8975, -97.0377),
            'SFO': (37.6213, -122.3790), 'SEA': (47.4502, -122.3088), 'BOS': (42.3656, -71.0096),
            'MIA': (25.7959, -80.2870), 'YYZ': (43.6777, -79.6248), 'YVR': (49.1967, -123.1816),
            'LAS': (36.0840, -115.1537), 'PHX': (33.4342, -112.0016), 'IAH': (29.9902, -95.3368),
            'MSP': (44.8848, -93.2223), 'DTW': (42.2162, -83.3554), 'CLT': (35.2144, -80.9473),
            'LHR': (51.4700, -0.4543), 'CDG': (49.0097, 2.5479), 'FRA': (50.0379, 8.5622),
            'AMS': (52.3086, 4.7639), 'MAD': (40.4839, -3.5680), 'BCN': (41.2974, 2.0833),
            'FCO': (41.7999, 12.2462), 'MUC': (48.3537, 11.7751), 'ZUR': (47.4647, 8.5492),
            'VIE': (48.1103, 16.5697), 'CPH': (55.6181, 12.6559), 'ARN': (59.6519, 17.9186),
            'OSL': (60.1939, 11.1004), 'HEL': (60.3172, 24.9633), 'IST': (41.2619, 28.7414),
            'ATH': (37.9364, 23.9445), 'LIS': (38.7813, -9.1363), 'DUB': (53.4213, -6.2701),
            'BRU': (50.9010, 4.4856), 'WAW': (52.1672, 20.9679), 'PRG': (50.1008, 14.2632),
            'NRT': (35.7720, 140.3929), 'HND': (35.5494, 139.7798), 'SIN': (1.3502, 103.9942),
            'ICN': (37.4602, 126.4407), 'PVG': (31.1443, 121.8083), 'PEK': (40.0799, 116.6031),
            'HKG': (22.3080, 113.9185), 'TPE': (25.0797, 121.2342), 'KUL': (2.7456, 101.7072),
            'BKK': (13.6900, 100.7501), 'DEL': (28.5665, 77.1031), 'BOM': (19.0896, 72.8656),
            'SYD': (33.9399, 151.1753), 'MEL': (37.6690, 144.8410), 'PER': (31.9403, 115.9669),
            'AKL': (37.0082, 174.7850), 'MNL': (14.5086, 121.0194), 'CGK': (6.1256, 106.6558),
            'DXB': (25.2532, 55.3657), 'DOH': (25.2731, 51.6075), 'AUH': (24.4330, 54.6511),
            'CAI': (30.1219, 31.4056), 'JNB': (26.1392, 28.2460), 'CPT': (33.9697, 18.6021),
            'ADD': (8.9806, 38.7626), 'NBO': (1.3192, 36.9278), 'CMN': (33.3676, -7.5898),
            'UBN': (47.8439, 106.7664), 'ULN': (47.8439, 106.7664), 'ZMCK': (47.8439, 106.7664),
            'PUS': (35.1795, 129.0756), 'CAN': (23.3924, 113.2988), 'SHA': (31.1979, 121.3364),
            'XIY': (34.4471, 108.7519), 'CTU': (30.5728, 103.9477), 'KMG': (24.9929, 102.7431),
            'URC': (43.9071, 87.4744), 'TSN': (39.1244, 117.3464), 'SJW': (38.2807, 114.6963)
        }
        
        # Scraped data from airline sustainability reports (2024)
        self.airline_efficiency_from_reports = {
            'Delta Air Lines': {'emissions_intensity': 0.095, 'unit': 'estimated kg CO2e / pkm', 'note': '6.6% fuel efficiency improvement since 2019'},
            'Alaska Airlines': {'emissions_intensity': 1.333, 'unit': 'kg CO2 / 1000 RTM'},
            'American Airlines': {'emissions_intensity': 0.092, 'unit': 'kg CO2e / pkm'},
            'Singapore Airlines': {'emissions_intensity': 0.80, 'unit': 'kg CO2e / LTK'},
            'Lufthansa': {'emissions_intensity': 0.0875, 'unit': 'kg CO2 / pkm'},
        }
    
    def get_flights_data(self, limit=100, offset=0):
        url = f"{self.base_url}/flights"
        params = {
            'access_key': self.aviationstack_key,
            'limit': limit,
            'offset': offset
        }
        
        try:
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()
            return data.get('data', [])
        except requests.exceptions.RequestException as e:
            st.error(f"API request failed: {e}")
            return []
    
    def get_airport_coordinates(self, iata_code):
        if iata_code in self.airports:
            return self.airports[iata_code]
        return None
    
    def calculate_distance(self, origin, destination):
        origin_coords = self.get_airport_coordinates(origin)
        dest_coords = self.get_airport_coordinates(destination)
        
        if origin_coords and dest_coords:
            distance = geodesic(origin_coords, dest_coords).kilometers
            return distance
        return None
    
    def get_aircraft_emission_factor(self, aircraft_code):
        if not aircraft_code or aircraft_code == 'None' or pd.isna(aircraft_code):
            return self.aircraft_emissions['DEFAULT']
        
        aircraft_code = str(aircraft_code).upper().strip()
        
        if aircraft_code in self.aircraft_emissions:
            return self.aircraft_emissions[aircraft_code]
        
        for key in self.aircraft_emissions:
            if key != 'DEFAULT' and (key in aircraft_code or aircraft_code.startswith(key)):
                return self.aircraft_emissions[key]
        
        return self.aircraft_emissions['DEFAULT']
    
    def get_aircraft_capacity(self, aircraft_code):
        if not aircraft_code or aircraft_code == 'None' or pd.isna(aircraft_code):
            return self.aircraft_capacities['DEFAULT']
        
        aircraft_code = str(aircraft_code).upper().strip()
        
        if aircraft_code in self.aircraft_capacities:
            return self.aircraft_capacities[aircraft_code]
        
        for key in self.aircraft_capacities:
            if key != 'DEFAULT' and (key in aircraft_code or aircraft_code.startswith(key)):
                return self.aircraft_capacities[key]
        
        return self.aircraft_capacities['DEFAULT']
    
    def calculate_co2_emissions(self, distance_km, aircraft_type, passenger_load_factor=0.8):
        if not distance_km or distance_km <= 0:
            return None
        
        emission_factor = self.get_aircraft_emission_factor(aircraft_type)
        
        base_emissions = distance_km * emission_factor
        adjusted_emissions = base_emissions / passenger_load_factor
        
        if distance_km < 1000:
            adjusted_emissions += 95 * emission_factor / passenger_load_factor
        elif distance_km < 3000:
            adjusted_emissions += 50 * emission_factor / passenger_load_factor
        
        return round(adjusted_emissions, 2)
    
    def process_flight_data(self, flights_data):
        processed_flights = []
        
        for flight in flights_data:
            try:
                aircraft_info = flight.get('aircraft') or {}
                departure_info = flight.get('departure') or {}
                arrival_info = flight.get('arrival') or {}
                airline_info = flight.get('airline') or {}
                flight_info_raw = flight.get('flight') or {}
                
                aircraft_icao = aircraft_info.get('icao')
                flight_info = {
                    'flight_number': flight_info_raw.get('number'),
                    'airline_name': airline_info.get('name'),
                    'airline_iata': airline_info.get('iata'),
                    'aircraft_type': aircraft_icao,
                    'aircraft_registration': aircraft_info.get('registration'),
                    'origin_airport': departure_info.get('iata'),
                    'destination_airport': arrival_info.get('iata'),
                    'origin_city': departure_info.get('airport'),
                    'destination_city': arrival_info.get('airport'),
                    'scheduled_departure': departure_info.get('scheduled'),
                    'scheduled_arrival': arrival_info.get('scheduled'),
                    'flight_status': flight.get('flight_status'),
                    'flight_iata': flight_info_raw.get('iata'),
                    'flight_icao': flight_info_raw.get('icao')
                }
                
                flight_info['passenger_capacity'] = self.get_aircraft_capacity(aircraft_icao)
                
                distance = None
                if flight_info['origin_airport'] and flight_info['destination_airport']:
                    try:
                        distance = self.calculate_distance(
                            flight_info['origin_airport'], 
                            flight_info['destination_airport']
                        )
                        flight_info['distance_km'] = distance
                    except Exception as e:
                        st.warning(f"Error calculating distance for {flight_info['origin_airport']}->{flight_info['destination_airport']}: {e}")
                        flight_info['distance_km'] = None
                else:
                    flight_info['distance_km'] = None
                
                if distance and distance > 0:
                    try:
                        co2_emissions = self.calculate_co2_emissions(
                            distance, 
                            flight_info['aircraft_type']
                        )
                        flight_info['co2_per_passenger_kg'] = co2_emissions
                        flight_info['co2_per_passenger_per_km'] = round(co2_emissions / distance, 4) if distance > 0 else None
                    except Exception as e:
                        st.warning(f"Error calculating CO2 emissions: {e}")
                        flight_info['co2_per_passenger_kg'] = None
                        flight_info['co2_per_passenger_per_km'] = None
                else:
                    flight_info['co2_per_passenger_kg'] = None
                    flight_info['co2_per_passenger_per_km'] = None
                
                flight_info['data_extracted_at'] = datetime.now().isoformat()
                
                processed_flights.append(flight_info)
                
            except Exception as e:
                st.warning(f"Error processing flight {flight}: {e}")
                continue
        
        return processed_flights
    
    # def extract_and_save_flights(self, total_flights=500, output_file='flight_emissions.csv'):
    #     all_flights = []
    #     batch_size = 100
    #     successful_extractions = 0
        
    #     st.info(f"Extracting {total_flights} flights data...")
        
    #     for offset in range(0, total_flights, batch_size):
    #         batch_number = offset // batch_size + 1
    #         total_batches = (total_flights - 1) // batch_size + 1
            
    #         st.write(f"Processing batch {batch_number}/{total_batches} (offset: {offset})")
            
    #         flights_data = self.get_flights_data(limit=batch_size, offset=offset)
            
    #         if not flights_data:
    #             st.warning("No more data available from API")
    #             break
            
    #         st.write(f"  - Retrieved {len(flights_data)} flights from API")
            
    #         processed_flights = self.process_flight_data(flights_data)
    #         successful_batch = len(processed_flights)
    #         successful_extractions += successful_batch
            
    #         st.write(f"  - Successfully processed {successful_batch} flights")
            
    #         all_flights.extend(processed_flights)
            
    #         time.sleep(1)
        
    #     if all_flights:
    #         df = pd.DataFrame(all_flights)
            
    #         st.write(f"\nTotal flights extracted: {len(df)}")
    #         st.write(f"Flights with distance data: {len(df[df['distance_km'].notna()])}")
    #         st.write(f"Flights with CO2 calculations: {len(df[df['co2_per_passenger_kg'].notna()])}")
            
    #         df_clean = df.dropna(subset=['origin_airport', 'destination_airport'])
            
    #         df.to_csv(output_file, index=False)
    #         st.success(f"Saved full dataset ({len(df)} flights) to {output_file}")
            
    #         clean_file = output_file.replace('.csv', '_clean.csv')
    #         df_with_emissions = df_clean[df_clean['co2_per_passenger_kg'].notna()]
            
    #         if len(df_with_emissions) > 0:
    #             df_with_emissions.to_csv(clean_file, index=False)
    #             st.success(f"Saved clean dataset ({len(df_with_emissions)} flights) to {clean_file}")
                
    #             return df_with_emissions
    #         else:
    #             st.error("No flights with emissions data available for ML training")
    #             return df
    #     else:
    #         st.error("No flight data extracted")
    #         return pd.DataFrame()

    def extract_and_save_flights(self, total_flights=500, output_file='flight_emissions.csv'):
        all_flights = []
        batch_size = 100
        successful_extractions = 0
        
        st.info(f"Extracting {total_flights} flights data...")
        
        for offset in range(0, total_flights, batch_size):
            batch_number = offset // batch_size + 1
            total_batches = (total_flights - 1) // batch_size + 1
            
            st.write(f"Processing batch {batch_number}/{total_batches} (offset: {offset})")
            
            flights_data = self.get_flights_data(limit=batch_size, offset=offset)
            
            if not flights_data:
                st.warning("No more data available from API")
                break
            
            st.write(f"  - Retrieved {len(flights_data)} flights from API")
            
            processed_flights = self.process_flight_data(flights_data)
            successful_batch = len(processed_flights)
            successful_extractions += successful_batch
            
            st.write(f"  - Successfully processed {successful_batch} flights")
            
            all_flights.extend(processed_flights)
            
            time.sleep(1)
        
        if all_flights:
            df = pd.DataFrame(all_flights)
            
            st.write(f"\nTotal flights extracted: {len(df)}")
            st.write(f"Flights with distance data: {len(df[df['distance_km'].notna()])}")
            st.write(f"Flights with CO2 calculations: {len(df[df['co2_per_passenger_kg'].notna()])}")
            
            df_clean = df.dropna(subset=['origin_airport', 'destination_airport'])
            
            df.to_csv(output_file, index=False)
            st.success(f"Saved full dataset ({len(df)} flights) to {output_file}")
            
            clean_file = output_file.replace('.csv', '_clean.csv')
            df_with_emissions = df_clean[df_clean['co2_per_passenger_kg'].notna()]
            
            if len(df_with_emissions) > 0:
                df_with_emissions.to_csv(clean_file, index=False)
                st.success(f"Saved clean dataset ({len(df_with_emissions)} flights) to {clean_file}")
                
                # Pass unique ID to display_summary_stats
                unique_id = f"extract_{int(time.time() * 1000)}"
                self.display_summary_stats(df_with_emissions, unique_id=unique_id)
                
                return df_with_emissions
            else:
                st.error("No flights with emissions data available for ML training")
                return df
        else:
            st.error("No flight data extracted")
            return pd.DataFrame()
    
    # def display_summary_stats(self, df):
    #     st.write("### FLIGHT EMISSIONS SUMMARY")
        
    #     if len(df) > 0:
    #         st.write(f"**Total flights processed**: {len(df)}")
    #         st.write(f"**Airlines covered**: {df['airline_name'].nunique()}")
    #         st.write(f"**Routes covered**: {len(df.groupby(['origin_airport', 'destination_airport']))}")
            
    #         if 'co2_per_passenger_kg' in df.columns:
    #             emissions_stats = df['co2_per_passenger_kg'].describe()
    #             st.write("#### CO₂ Emissions per Passenger (kg)")
    #             st.write(f"- Mean: {emissions_stats['mean']:.2f} kg")
    #             st.write(f"- Median: {emissions_stats['50%']:.2f} kg")
    #             st.write(f"- Min: {emissions_stats['min']:.2f} kg")
    #             st.write(f"- Max: {emissions_stats['max']:.2f} kg")
                
    #             # Plot emissions distribution
    #             fig_emissions = px.histogram(df, x='co2_per_passenger_kg', nbins=50, title="CO₂ Emissions per Passenger Distribution")
    #             st.plotly_chart(fig_emissions)
            
    #         if 'distance_km' in df.columns:
    #             distance_stats = df['distance_km'].describe()
    #             st.write("#### Distance Statistics")
    #             st.write(f"- Mean: {distance_stats['mean']:.0f} km")
    #             st.write(f"- Median: {distance_stats['50%']:.0f} km")
    #             st.write(f"- Min: {distance_stats['min']:.0f} km")
    #             st.write(f"- Max: {distance_stats['max']:.0f} km")
                
    #             # Plot distance distribution
    #             fig_distance = px.histogram(df, x='distance_km', nbins=50, title="Flight Distance Distribution")
    #             st.plotly_chart(fig_distance)
            
    #         if 'co2_per_passenger_per_km' in df.columns and 'airline_name' in df.columns:
    #             airline_efficiency = df.groupby('airline_name')['co2_per_passenger_per_km'].mean().sort_values()
    #             st.write("#### Top 5 Most Efficient Airlines (kg CO₂/passenger/km)")
    #             for airline, efficiency in airline_efficiency.head().items():
    #                 st.write(f"- {airline}: {efficiency:.4f}")
                
    #             # Plot airline efficiency
    #             fig_efficiency = px.bar(
    #                 airline_efficiency.reset_index(),
    #                 x='airline_name',
    #                 y='co2_per_passenger_per_km',
    #                 title="Airline Efficiency Comparison (kg CO₂/passenger/km)",
    #                 labels={'co2_per_passenger_per_km': 'CO₂ per Passenger per km', 'airline_name': 'Airline'}
    #             )
    #             st.plotly_chart(fig_efficiency)

    def display_summary_stats(self, df, unique_id=None):
        # Generate a unique ID if none provided (e.g., timestamp or random string)
        if unique_id is None:
            unique_id = str(int(time.time() * 1000))  # Use timestamp in milliseconds
        
        st.write("### FLIGHT EMISSIONS SUMMARY")
        
        if len(df) > 0:
            st.write(f"**Total flights processed**: {len(df)}")
            st.write(f"**Airlines covered**: {df['airline_name'].nunique()}")
            st.write(f"**Routes covered**: {len(df.groupby(['origin_airport', 'destination_airport']))}")
            
            if 'co2_per_passenger_kg' in df.columns:
                emissions_stats = df['co2_per_passenger_kg'].describe()
                st.write("#### CO₂ Emissions per Passenger (kg)")
                st.write(f"- Mean: {emissions_stats['mean']:.2f} kg")
                st.write(f"- Median: {emissions_stats['50%']:.2f} kg")
                st.write(f"- Min: {emissions_stats['min']:.2f} kg")
                st.write(f"- Max: {emissions_stats['max']:.2f} kg")
                
                # Plot emissions distribution with unique key
                fig_emissions = px.histogram(df, x='co2_per_passenger_kg', nbins=50, title="CO₂ Emissions per Passenger Distribution")
                st.plotly_chart(fig_emissions, key=f"emissions_histogram_{unique_id}")
            
            if 'distance_km' in df.columns:
                distance_stats = df['distance_km'].describe()
                st.write("#### Distance Statistics")
                st.write(f"- Mean: {distance_stats['mean']:.0f} km")
                st.write(f"- Median: {distance_stats['50%']:.0f} km")
                st.write(f"- Min: {distance_stats['min']:.0f} km")
                st.write(f"- Max: {distance_stats['max']:.0f} km")
                
                # Plot distance distribution with unique key
                fig_distance = px.histogram(df, x='distance_km', nbins=50, title="Flight Distance Distribution")
                st.plotly_chart(fig_distance, key=f"distance_histogram_{unique_id}")
            
            if 'co2_per_passenger_per_km' in df.columns and 'airline_name' in df.columns:
                airline_efficiency = df.groupby('airline_name')['co2_per_passenger_per_km'].mean().sort_values()
                st.write("#### Top 5 Most Efficient Airlines (kg CO₂/passenger/km)")
                for airline, efficiency in airline_efficiency.head().items():
                    st.write(f"- {airline}: {efficiency:.4f}")
                
                # Plot airline efficiency with unique key
                fig_efficiency = px.bar(
                    airline_efficiency.reset_index(),
                    x='airline_name',
                    y='co2_per_passenger_per_km',
                    title="Airline Efficiency Comparison (kg CO₂/passenger/km)",
                    labels={'co2_per_passenger_per_km': 'CO₂ per Passenger per km', 'airline_name': 'Airline'}
                )
                st.plotly_chart(fig_efficiency, key=f"airline_efficiency_bar_{unique_id}")
    
    def compare_airlines_efficiency(self, df):
        if 'co2_per_passenger_per_km' not in df.columns:
            st.error("CO₂ efficiency data not available")
            return None
        
        airline_stats = df.groupby('airline_name').agg({
            'co2_per_passenger_per_km': ['mean', 'std', 'count'],
            'distance_km': 'mean',
            'co2_per_passenger_kg': 'mean'
        }).round(4)
        
        airline_stats.columns = ['avg_co2_per_km', 'std_co2_per_km', 'flight_count', 'avg_distance', 'avg_total_co2']
        airline_stats = airline_stats.sort_values('avg_co2_per_km')
        
        st.write("### Calculated Airline Efficiency")
        st.dataframe(airline_stats)
        
        # Plot average CO2 per km
        fig_airline_co2 = px.bar(
            airline_stats.reset_index(),
            x='airline_name',
            y='avg_co2_per_km',
            title="Average CO₂ per Passenger per km by Airline",
            labels={'avg_co2_per_km': 'Average CO₂ (kg/passenger/km)', 'airline_name': 'Airline'}
        )
        st.plotly_chart(fig_airline_co2)
        
        st.write("### Benchmark from Sustainability Reports")
        for airline, data in self.airline_efficiency_from_reports.items():
            st.write(f"{airline}: {data['emissions_intensity']} {data['unit']} {data.get('note', '')}")
        
        return airline_stats

def train_emission_predictor(clean_csv='flight_emissions_clean.csv'):
    try:
        df = pd.read_csv(clean_csv)
        
        df = df.dropna(subset=['distance_km', 'passenger_capacity', 'co2_per_passenger_kg'])
        
        if df.empty:
            st.error("No sufficient data available for training the model. Please collect more flight data with complete information.")
            return None
        
        X = df[['distance_km', 'passenger_capacity']]
        X = sm.add_constant(X)
        y = df['co2_per_passenger_kg']
        
        model = sm.OLS(y, X).fit()
        
        st.write("### ML Model Summary for Emission Prediction")
        st.text(model.summary().as_text())
        
        # Interactive prediction
        st.write("#### Predict CO₂ Emissions")
        distance_input = st.number_input("Enter distance (km)", min_value=0.0, value=1000.0)
        capacity_input = st.number_input("Enter passenger capacity", min_value=1, value=180)
        
        example_data = {'distance_km': [distance_input], 'passenger_capacity': [capacity_input]}
        example_df = pd.DataFrame(example_data)
        example_df = sm.add_constant(example_df, has_constant='add')
        
        if list(example_df.columns) != list(X.columns):
            st.warning("Feature mismatch in prediction data. Adjusting columns...")
            example_df = example_df[X.columns]
        
        prediction = model.predict(example_df)
        st.write(f"**Predicted CO₂ for {distance_input} km, {capacity_input} pax**: {prediction[0]:.2f} kg")
        
        # Plot actual vs predicted
        df['predicted_co2'] = model.predict(X)
        fig_pred = px.scatter(
            df, x='co2_per_passenger_kg', y='predicted_co2',
            title="Actual vs Predicted CO₂ Emissions per Passenger",
            labels={'co2_per_passenger_kg': 'Actual CO₂ (kg)', 'predicted_co2': 'Predicted CO₂ (kg)'}
        )
        st.plotly_chart(fig_pred)
        
        return model
    except Exception as e:
        st.error(f"Error in training model: {e}")
        return None

def automated_update_pipeline(total_flights=200, output_file='flight_emissions.csv'):
    st.write(f"### Running automated update at {datetime.now()}")
    calculator = FlightCarbonCalculator("034659c48f30f9832c05aea458a29eb5")
    df = calculator.extract_and_save_flights(total_flights=total_flights, output_file=output_file)
    if len(df) > 0:
        calculator.compare_airlines_efficiency(df)
        train_emission_predictor(output_file.replace('.csv', '_clean.csv'))
    st.success("Update complete.")

# Streamlit app
st.title("Flight Carbon Footprint Calculator")

# Initialize calculator
calculator = FlightCarbonCalculator("034659c48f30f9832c05aea458a29eb5")

# Sidebar for data update
st.sidebar.header("Data Update")
total_flights = st.sidebar.number_input("Number of flights to extract", min_value=100, max_value=1000, value=200, step=100)
output_file = "flight_emissions.csv"
if st.sidebar.button("Update Data"):
    with st.spinner("Updating flight data..."):
        df = calculator.extract_and_save_flights(total_flights=total_flights, output_file=output_file)
        if len(df) > 0:
            calculator.display_summary_stats(df)
            st.session_state['df'] = df
        else:
            st.error("No data available to display.")

# Display data if available
if 'df' in st.session_state and len(st.session_state['df']) > 0:
    df = st.session_state['df']
    st.write("### Current Data Overview")
    st.dataframe(df.head())
    
    # Display summary statistics
    calculator.display_summary_stats(df)
    
    # Compare airline efficiency
    st.write("### Airline Efficiency Analysis")
    calculator.compare_airlines_efficiency(df)
    
    # Train and display ML model results
    st.write("### Emission Prediction Model")
    train_emission_predictor(output_file.replace('.csv', '_clean.csv'))
else:
    st.info("No data loaded. Please update data using the sidebar.")

# Documentation
st.write("### Documentation")
st.markdown("""
- **API Usage**: AviationStack API for flight data (routes, aircraft, etc.).
- **CO2 Calculation**: Based on geodesic distance, aircraft-specific emission factors (kg CO2/km/pax), adjusted for load factor (0.8) and takeoff/landing penalties.
- **Features Extracted**: Airline, route (origin/dest), distance_km, aircraft_type, passenger_capacity, co2_per_passenger_kg, co2_per_passenger_per_km.
- **CSV Output**: Full dataset saved as `flight_emissions.csv`, clean dataset with emissions as `_clean.csv`.
- **ML Model**: OLS regression using statsmodels to predict CO2 emissions based on distance and passenger capacity.
- **Pipeline**: Automated update via sidebar button, fetching new data, recalculating emissions, and retraining the model.
""")