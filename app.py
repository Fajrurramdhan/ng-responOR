import io
import streamlit as st
import pandas as pd
import json
import os
import tempfile
from pathlib import Path
import sys
from itertools import combinations
import math
import folium
from streamlit_folium import st_folium
import plotly.express as px
import plotly.graph_objects as go
import networkx as nx
from networkx.readwrite import json_graph
import multiprocessing as mp
import concurrent.futures
from collections import OrderedDict
import csv
import numpy as np
from folium.plugins import HeatMap, Fullscreen
import random
import string

def load_indonesia_data():
    """Load Indonesia province and location data from JSON file"""
    try:
        with open('indonesia_data.json', 'r') as f:
            data = json.load(f)
        return data['provinces'], data['locations']
    except Exception as e:
        st.error(f"Error loading Indonesia data: {str(e)}")
        return [], {}

def create_province_selection():
    """Creates the province selection interface with auto-generated directory name"""
    # Load provinces from JSON
    provinces, _ = load_indonesia_data()
    
    # Initialize session state for directory name if not exists
    if 'directory_name' not in st.session_state:
        st.session_state.directory_name = ""
        
    def update_directory_name():
        """Callback function to update directory name when province changes"""
        province = st.session_state.province_select
        if province and province != "--select--":
            random_digits = ''.join(random.choices(string.digits, k=9))
            st.session_state.directory_name = f"NG{random_digits}_{province}"
        else:
            st.session_state.directory_name = ""
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Create the province dropdown with callback
        selected_province = st.selectbox(
            "Base Area (Province) *",
            options=["--select--"] + provinces,
            key="province_select",
            on_change=update_directory_name
        )
    
    with col2:
        # Display directory name
        st.text_input(
            "Directory Name *",
            value=st.session_state.directory_name,
            disabled=True
        )
    
    # Return None if --select-- is chosen, otherwise return the province
    return (None if selected_province == "--select--" else selected_province), st.session_state.directory_name


def generate_random_id(prefix="NG", length=12):
    """Generate a random ID for project identification."""
    return f"{prefix}{''.join(random.choices(string.digits, k=length))}"

def get_master_locations(selected_province):
    """Returns predefined master location data based on selected province"""
    _, location_data = load_indonesia_data()
    
    # If province exists in data
    if selected_province in location_data:
        return (
            location_data[selected_province]["depot"],
            location_data[selected_province]["shelter"],
            location_data[selected_province]["village"]
        )
    else:
        # Return empty lists if no data for province
        return [], [], []

def create_location_selection(selected_province):
    """Creates the location selection interface with search and filtering"""
    st.subheader("Pilih Lokasi")
    
    # Get master location data based on selected province
    depot_data, shelter_data, village_data = get_master_locations(selected_province)
    
    # Check if there's any data for the selected province
    if not any([depot_data, shelter_data, village_data]):
        st.warning(f"Tidak ada data tersedia untuk provinsi {selected_province}")
        return None
    
    # Create tabs for different location types
    location_tabs = st.tabs(["Depot", "Shelter", "Village"])
    selected_locations = []
    
    # Only show tabs with data
    with location_tabs[0]:
        if depot_data:
            st.write("### Master Depot")
            depot_search = st.text_input("Cari Depot", key="depot_search")
            filtered_depots = [d for d in depot_data if depot_search.lower() in d['name'].lower()] if depot_search else depot_data
            
            col1, col2 = st.columns([0.1, 0.9])
            with col1:
                select_all_depots = st.checkbox("Select All", key="select_all_depots")
            
            for depot in filtered_depots:
                if select_all_depots or st.checkbox(f"{depot['name']}", key=f"depot_{depot['name']}"):
                    selected_locations.append(depot)
        else:
            st.info("Tidak ada data depot tersedia untuk provinsi ini")
    
    with location_tabs[1]:
        if shelter_data:
            st.write("### Master Shelter")
            shelter_search = st.text_input("Cari Shelter", key="shelter_search")
            filtered_shelters = [s for s in shelter_data if shelter_search.lower() in s['name'].lower()] if shelter_search else shelter_data
            
            col1, col2 = st.columns([0.1, 0.9])
            with col1:
                select_all_shelters = st.checkbox("Select All", key="select_all_shelters")
            
            for shelter in filtered_shelters:
                if select_all_shelters or st.checkbox(f"{shelter['name']}", key=f"shelter_{shelter['name']}"):
                    selected_locations.append(shelter)
        else:
            st.info("Tidak ada data shelter tersedia untuk provinsi ini")
    
    with location_tabs[2]:
        if village_data:
            st.write("### Master Village")
            village_search = st.text_input("Cari Village", key="village_search")
            filtered_villages = [v for v in village_data if village_search.lower() in v['name'].lower()] if village_search else village_data
            
            col1, col2 = st.columns([0.1, 0.9])
            with col1:
                select_all_villages = st.checkbox("Select All", key="select_all_villages")
            
            for village in filtered_villages:
                if select_all_villages or st.checkbox(f"{village['name']}", key=f"village_{village['name']}"):
                    selected_locations.append(village)
        else:
            st.info("Tidak ada data village tersedia untuk provinsi ini")
    
    # Process selected locations
    if selected_locations:
        return process_selected_locations(selected_locations)
    return None

def process_selected_locations(selected_locations):
    """Process the selected locations and return as DataFrame"""
    if not selected_locations:
        return None
        
    selected_df = pd.DataFrame(selected_locations)
    
    # Check if at least one location of each type is selected
    type_counts = selected_df['type'].value_counts()
    required_types = {'depot', 'shelter', 'village'}
    missing_types = required_types - set(type_counts.index)
    
    if missing_types:
        missing_types_str = ", ".join(missing_types)
        st.error(f"⚠️ Anda harus memilih minimal 1 lokasi untuk setiap tipe. Tipe yang belum dipilih: {missing_types_str}")
        return None
    
    if st.session_state.nodes_df is not None:
        selected_df['node_id'] = selected_df.apply(
            lambda row: find_nearest_node({
                'lat': row['latitude'], 
                'lon': row['longitude']
            }, st.session_state.nodes_df), 
            axis=1
        )
        
        preview = preview_data(selected_df)
        
        st.success(f"✅ {len(selected_df)} lokasi berhasil dipilih!")
        
        # Display type distribution
        st.write("### Distribusi Tipe Lokasi:")
        for type_name in required_types:
            count = type_counts.get(type_name, 0)
            if count > 0:
                st.success(f"✓ {type_name.title()}: {count} lokasi")
            else:
                st.error(f"✗ {type_name.title()}: Belum dipilih")
        
        preview_tabs = st.tabs(["Data Preview", "Statistik", "Peta"])
        
        with preview_tabs[0]:
            st.write("Sample Data:")
            st.dataframe(selected_df.head())
        
        with preview_tabs[1]:
            col_stat1, col_stat2 = st.columns(2)
            
            with col_stat1:
                st.metric("Total Lokasi", preview['total_locations'])
                st.write("Distribusi Tipe:")
                for type_name, count in preview['types_distribution'].items():
                    st.write(f"- {type_name}: {count}")
            
            with col_stat2:
                if 'population_stats' in preview:
                    st.metric("Total Populasi", f"{preview['population_stats']['total']:,}")
                    st.metric("Rata-rata Populasi", f"{preview['population_stats']['average']:,.0f}")
        
        with preview_tabs[2]:
            m = create_preview_map(selected_df)
            st_folium(m, width=800)
        
        return selected_df
    else:
        st.warning("Harap upload network file terlebih dahulu untuk mendapatkan referensi nodes.")
        return None
    
def create_preview_map(df):
    """Create a preview map with the selected locations"""
    m = folium.Map(location=[-6.21462, 106.84513], zoom_start=11)
    
    color_map = {
        'village': 'blue',
        'shelter': 'red',
        'depot': 'green'
    }
    
    for _, row in df.iterrows():
        color = color_map.get(row['type'], 'gray')
        folium.Marker(
            [row['latitude'], row['longitude']],
            popup=row['name'],
            tooltip=f"{row['name']} ({row['type']})",
            icon=folium.Icon(color=color)
        ).add_to(m)
    
    return m
# Add this to the main function to show selection statistics
def show_selection_stats(selected_df):
    """Shows statistics about selected locations"""
    if selected_df is not None and not selected_df.empty:
        st.write("### Statistik Lokasi Terpilih")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Show total counts
            total_locations = len(selected_df)
            st.metric("Total Lokasi Terpilih", total_locations)
            
            # Show type distribution
            st.write("Distribusi Tipe:")
            type_counts = selected_df['type'].value_counts()
            for type_name, count in type_counts.items():
                st.write(f"- {type_name.title()}: {count}")
        
        with col2:
            # Show map of selected locations
            m = folium.Map(location=[-6.21462, 106.84513], zoom_start=11)
            
            # Color mapping for different types
            color_map = {
                'village': 'blue',
                'shelter': 'red',
                'depot': 'green'
            }
            
            # Add markers for selected locations
            for _, row in selected_df.iterrows():
                folium.Marker(
                    [row['latitude'], row['longitude']],
                    popup=row['name'],
                    tooltip=f"{row['name']} ({row['type']})",
                    icon=folium.Icon(color=color_map.get(row['type'], 'gray'))
                ).add_to(m)
            
            st_folium(m, height=300)


##perubahan

# Tambahkan fungsi validate_and_transform_data di sini
def validate_and_transform_data(df):
    """
    Validate and transform the input data dynamically based on column count
    
    Parameters:
    df (pandas.DataFrame): Input DataFrame
    
    Returns:
    tuple: (is_valid, message, transformed_df)
    """
    try:
        # Get number of columns
        num_columns = len(df.columns)
        
        # Handle different column structures
        if num_columns not in [4, 5]:
            return False, f"Format tidak valid: Data harus memiliki 4 atau 5 kolom, ditemukan {num_columns} kolom", None
            
        # Map location types to standard types
        type_mapping = {
            # Villages and residential
            'village': 'village',
            'residential': 'village',
            'housing': 'village',
            'settlement': 'village',
            'kampung': 'village',
            
            # Shelters
            'shelter': 'shelter',
            'evacuation': 'shelter',
            'camp': 'shelter',
            'deployment': 'shelter',
            'extra': 'shelter',
            
            # Depots and warehouses
            'depot': 'depot',
            'warehouse': 'depot',
            'storage': 'depot',
            'logistics': 'depot',
            'damaged': 'depot',
            'airport': 'depot'
        }
        
        # Create standard column mapping
        df = df.copy()  # Create a copy to avoid modifying original
        
        # Rename columns based on position
        if num_columns == 4:
            df.columns = ['name', 'type', 'latitude', 'longitude']
        else:  # num_columns == 5
            df.columns = ['name', 'type', 'latitude', 'longitude', 'population']
            
        # Convert coordinates to float
        df['latitude'] = df['latitude'].astype(float)
        df['longitude'] = df['longitude'].astype(float)
        
        # If population exists, convert to integer
        if 'population' in df.columns:
            df['population'] = df['population'].astype(int)
        
        # Validate coordinate ranges for Jakarta
        valid_lat = (df['latitude'] >= -6.4) & (df['latitude'] <= -6.0)
        valid_lon = (df['longitude'] >= 106.6) & (df['longitude'] <= 107.0)
        
        if not all(valid_lat):
            return False, "Latitude harus berada dalam range Jakarta (-6.4 sampai -6.0)", None
        if not all(valid_lon):
            return False, "Longitude harus berada dalam range Jakarta (106.6 sampai 107.0)", None
            
        # Standardize type column
        df['type'] = df['type'].str.lower().str.strip()
        
        # Map types using the type_mapping dictionary
        df['type'] = df['type'].map(lambda x: type_mapping.get(x, None))
        
        # Check for invalid types
        invalid_mask = df['type'].isna()
        if invalid_mask.any():
            invalid_types = df[invalid_mask]['type'].unique()
            valid_types_str = ", ".join(sorted(set(type_mapping.values())))
            invalid_types_str = ", ".join(sorted(set(df[invalid_mask]['type'].unique())))
            return False, f"Tipe lokasi tidak dikenali: {invalid_types_str}. Tipe yang diperbolehkan: {valid_types_str}", None
        
        # Add node_id column (will be filled later)
        df['node_id'] = None
        
        # Add metadata about the source format
        df.attrs['source_format'] = f"{num_columns}_columns"
        df.attrs['has_population'] = 'population' in df.columns
        
        return True, "Data valid", df
        
    except Exception as e:
        return False, f"Error validating data: {str(e)}", None

def preview_data(df):
    """
    Generate a preview of the validated data
    
    Parameters:
    df (pandas.DataFrame): Validated DataFrame
    
    Returns:
    dict: Statistics and preview information
    """
    preview = {
        'total_locations': len(df),
        'types_distribution': df['type'].value_counts().to_dict(),
        'coordinate_ranges': {
            'latitude': {
                'min': df['latitude'].min(),
                'max': df['latitude'].max()
            },
            'longitude': {
                'min': df['longitude'].min(),
                'max': df['longitude'].max()
            }
        },
        'sample_data': df.head(5).to_dict('records')
    }
    
    # Add population statistics if available
    if 'population' in df.columns:
        preview['population_stats'] = {
            'total': df['population'].sum(),
            'average': df['population'].mean(),
            'max': df['population'].max(),
            'min': df['population'].min()
        }
        
    return preview
    
def process_network_file(file_path):
    """Process network file and return node and edge information"""
    nodes = []
    edges = []
    total_nodes = None
    total_edges = None
    count_edges = 0
    count_nodes = 0
    
    with open(file_path) as f:
        count = 0
        for line in f:
            if count == 7:
                total_nodes = int(line)
            elif count == 8:
                total_edges = int(line)
            elif count > 8:
                if count_nodes < total_nodes:
                    # Start reading nodes
                    node_id, lat, lon = line.split()
                    nodes.append({
                        'id': int(node_id),
                        'lat': float(lat),
                        'lon': float(lon)
                    })
                    count_nodes += 1
                else:
                    # Read edges
                    source_id, target_id, length, street_type, max_speed, bidirectional = line.split()
                    edges.append({
                        'source': int(source_id),
                        'target': int(target_id),
                        'length': float(length),
                        'type': street_type,
                        'speed': float(max_speed),
                        'bidirectional': bool(int(bidirectional))
                    })
                    count_edges += 1
            count += 1
    
    return pd.DataFrame(nodes), pd.DataFrame(edges)

def find_nearest_node(point, nodes_df):
    """Find nearest node for a given point"""
    distances = np.sqrt(
        (nodes_df['lat'] - point['lat'])**2 + 
        (nodes_df['lon'] - point['lon'])**2
    )
    return nodes_df.iloc[distances.argmin()]['id']

def calculate_risk(nodes_df):
    """Calculate random risk values for nodes"""
    return np.random.uniform(0, 1, size=len(nodes_df))

def create_subgraph(poi_df, nodes_df, edges_df):
    """Create subgraph based on POIs with connections between different types"""
    G = nx.Graph()
    
    # Add nodes
    for _, node in nodes_df.iterrows():
        G.add_node(node['id'], lat=node['lat'], lon=node['lon'])
    
    # Add edges
    for _, edge in edges_df.iterrows():
        G.add_edge(
            edge['source'], 
            edge['target'], 
            weight=edge['length']
        )
    
    try:
        # Debug print
        st.write("Debug: POI DataFrame columns:", poi_df.columns.tolist())
        st.write("Debug: First few rows of POI data:", poi_df.head())
        
        # Get nodes by type (menggunakan indeks kolom)
        village_nodes = poi_df[poi_df.iloc[:, 1].str.lower() == 'village']['node_id'].astype(int).unique()
        shelter_nodes = poi_df[poi_df.iloc[:, 1].str.lower() == 'shelter']['node_id'].astype(int).unique()
        depot_nodes = poi_df[poi_df.iloc[:, 1].str.lower() == 'depot']['node_id'].astype(int).unique()
        
        st.write("Debug: Number of nodes by type:",
                f"Villages: {len(village_nodes)}, "
                f"Shelters: {len(shelter_nodes)}, "
                f"Depots: {len(depot_nodes)}")
        
        # Get shortest paths between all combinations
        paths = []
        
        # Connect villages to nearest shelters
        for v_node in village_nodes:
            min_path = None
            min_length = float('inf')
            for s_node in shelter_nodes:
                try:
                    path = nx.shortest_path(G, int(v_node), int(s_node), weight='weight')
                    path_length = sum(G[path[i]][path[i+1]]['weight'] for i in range(len(path)-1))
                    if path_length < min_length:
                        min_length = path_length
                        min_path = path
                except nx.NetworkXNoPath:
                    continue
                except Exception as e:
                    st.write(f"Debug: Error in path finding: {str(e)}")
                    continue
            if min_path:
                paths.extend(min_path)
        
        # Connect shelters to nearest depots
        for s_node in shelter_nodes:
            min_path = None
            min_length = float('inf')
            for d_node in depot_nodes:
                try:
                    path = nx.shortest_path(G, int(s_node), int(d_node), weight='weight')
                    path_length = sum(G[path[i]][path[i+1]]['weight'] for i in range(len(path)-1))
                    if path_length < min_length:
                        min_length = path_length
                        min_path = path
                except nx.NetworkXNoPath:
                    continue
                except Exception as e:
                    st.write(f"Debug: Error in path finding: {str(e)}")
                    continue
            if min_path:
                paths.extend(min_path)
        
        # Create subgraph with unique nodes from paths
        unique_nodes = list(set(paths)) if paths else []
        
        # Add all POI nodes to ensure they're included
        poi_nodes = poi_df['node_id'].dropna().astype(int).unique()
        unique_nodes.extend(list(set(poi_nodes) - set(unique_nodes)))
        
        st.write("Debug: Number of unique nodes in subgraph:", len(unique_nodes))
        
        return G.subgraph(unique_nodes)
        
    except Exception as e:
        st.error(f"Error in create_subgraph: {str(e)}")
        # Return minimum subgraph if error occurs
        return G.subgraph(poi_df['node_id'].dropna().astype(int).unique())


def get_risk_category(risk_value):
    """Get risk category and color based on risk value"""
    if risk_value > 0.7:
        return "Tinggi", "#ff0000"
    elif risk_value > 0.3:
        return "Sedang", "#ff8080"
    else:
        return "Rendah", "#ffcccc"

def get_gradient_color(risk_value):
    """Get color based on risk value using a gradient"""
    # Define gradient from green to red
    red = int(255 * risk_value)
    green = int(255 * (1 - risk_value))
    blue = 0
    return f"#{red:02x}{green:02x}{blue:02x}"

def add_connection_lines(risk_map, subgraph_nodes, edges_with_risk, poi_df):
    """Add lines connecting nodes based on type with improved styling"""
    # Create feature groups for different connection types
    village_connections = folium.FeatureGroup(name='Village Connections')
    shelter_connections = folium.FeatureGroup(name='Shelter Connections')
    depot_connections = folium.FeatureGroup(name='Depot Connections')

    # Create dictionary of node types
    if 'type' not in poi_df.columns:
        poi_df['type'] = 'unknown'
    node_types = dict(zip(poi_df['node_id'], poi_df['type']))

    # Add edges
    for _, edge in edges_with_risk.iterrows():
        try:
            source_id = int(edge['source'])
            target_id = int(edge['target'])
            source_type = node_types.get(source_id, 'unknown').lower()
            target_type = node_types.get(target_id, 'unknown').lower()

            # Get nodes coordinates
            source_node = subgraph_nodes[subgraph_nodes['id'] == source_id].iloc[0]
            target_node = subgraph_nodes[subgraph_nodes['id'] == target_id].iloc[0]

            # Ensure coordinates are float values
            source_lat = float(source_node['lat'])
            source_lon = float(source_node['lon'])
            target_lat = float(target_node['lat'])
            target_lon = float(target_node['lon'])

            # Create line with gradient color based on average risk
            points = [
                [source_lat, source_lon],
                [target_lat, target_lon]
            ]
            
            line = folium.PolyLine(
                points,
                weight=2,
                color=get_gradient_color((edge['risk_source'] + edge['risk_target']) / 2),
                opacity=0.6,
                tooltip=f"{source_type.title()} to {target_type.title()}"
            )

            # Add to appropriate feature group
            if 'village' in (source_type, target_type):
                line.add_to(village_connections)
            elif 'shelter' in (source_type, target_type):
                line.add_to(shelter_connections)
            else:
                line.add_to(depot_connections)

        except Exception as e:
            st.warning(f"Warning: Error adding connection line: {str(e)}")
            continue

    # Add feature groups to map
    village_connections.add_to(risk_map)
    shelter_connections.add_to(risk_map)
    depot_connections.add_to(risk_map)

    return risk_map

def create_risk_map(subgraph_nodes, edges_with_risk=None, poi_df=None):
    """Create risk map with detailed tooltips, node type information, and custom icons"""
    try:
        # Initialize map
        risk_map = folium.Map(
            location=[-6.21462, 106.84513],
            zoom_start=11,
            tiles='OpenStreetMap',
            prefer_canvas=True
        )
        
        # Add Fullscreen control
        Fullscreen(
            position="topright",
            title="Expand me",
            title_cancel="Exit me",
            force_separate_button=True
        ).add_to(risk_map)

        # Create feature groups for each type
        feature_groups = {
            'village': folium.FeatureGroup(name='Villages'),
            'shelter': folium.FeatureGroup(name='Shelters'),
            'depot': folium.FeatureGroup(name='Depots'),
            'heatmap': folium.FeatureGroup(name='Risk Heatmap')
        }

        # Add connection lines first (if available)
        if edges_with_risk is not None and poi_df is not None:
            risk_map = add_connection_lines(risk_map, subgraph_nodes, edges_with_risk, poi_df)

        # Create a dictionary of node_id to location info from POI data
        location_info = {}
        if poi_df is not None and not poi_df.empty:
            try:
                valid_nodes = poi_df[poi_df['node_id'].notna()]
                for _, row in valid_nodes.iterrows():
                    node_id = int(row['node_id']) if pd.notna(row['node_id']) else None
                    if node_id is not None:
                        location_info[node_id] = {
                            'name': str(row['name']),  # Ensure string
                            'type': str(row.get('type', 'unknown')).lower(),  # Ensure string and lowercase
                            'population': row.get('population', 'N/A')
                        }
            except Exception as e:
                st.warning(f"Warning: Error processing POI data: {str(e)}")

        # Add nodes with enhanced tooltips and custom icons
        heat_data = []
        for _, node in subgraph_nodes.iterrows():
            try:
                # Convert all values to appropriate types explicitly
                node_lat = float(node['lat'])
                node_lon = float(node['lon'])
                node_risk = float(node['risk'])
                node_id = int(node['id'])

                # Get location info
                info = location_info.get(node_id, {})
                location_type = str(info.get('type', 'unknown')).lower()
                location_name = str(info.get('name', f'Node {node_id}'))
                population = info.get('population', 'N/A')

                # Get risk category and color
                risk_category, _ = get_risk_category(node_risk)
                gradient_color = get_gradient_color(node_risk)

                # Create popup HTML with explicit string formatting
                popup_html = f"""
                <div style="font-family: Arial, sans-serif; padding: 10px; min-width: 200px;">
                    <h4 style="margin-top: 0; color: #2c3e50;">{location_name}</h4>
                    <table style="width: 100%; border-collapse: collapse;">
                        <tr>
                            <td style="padding: 3px; color: #7f8c8d;"><strong>Type:</strong></td>
                            <td style="padding: 3px;">{location_type.title()}</td>
                        </tr>
                        <tr>
                            <td style="padding: 3px; color: #7f8c8d;"><strong>Risk Value:</strong></td>
                            <td style="padding: 3px;">{node_risk:.3f}</td>
                        </tr>
                        <tr>
                            <td style="padding: 3px; color: #7f8c8d;"><strong>Risk Level:</strong></td>
                            <td style="padding: 3px;">{risk_category}</td>
                        </tr>
                        <tr>
                            <td style="padding: 3px; color: #7f8c8d;"><strong>Population:</strong></td>
                            <td style="padding: 3px;">{population}</td>
                        </tr>
                        <tr>
                            <td style="padding: 3px; color: #7f8c8d;"><strong>Coordinates:</strong></td>
                            <td style="padding: 3px;">{node_lat:.6f}, {node_lon:.6f}</td>
                        </tr>
                    </table>
                </div>
                """

                # Create marker and add to map
                marker = folium.Marker(
                    location=[node_lat, node_lon],
                    popup=folium.Popup(popup_html, max_width=300),
                    tooltip=f"{location_name} - Risk: {node_risk:.3f}",
                    icon=folium.Icon(color=get_gradient_color(node_risk))
                )

                # Add marker to appropriate feature group
                if location_type in feature_groups:
                    marker.add_to(feature_groups[location_type])
                else:
                    marker.add_to(risk_map)

                # Add data for heatmap
                heat_data.append([node_lat, node_lon, node_risk])

            except Exception as e:
                st.warning(f"Warning: Error adding marker for node {node.get('id', 'unknown')}: {str(e)}")
                continue

        # Add all feature groups to map
        for group in feature_groups.values():
            group.add_to(risk_map)

        return risk_map

    except Exception as e:
        st.error(f"Error creating risk map: {str(e)}")
        return None


def main():
    st.set_page_config(layout="wide", page_title="Integrate Module Network Generation")
    
    # Initialize session state
    if 'nodes_df' not in st.session_state:
        st.session_state.nodes_df = None
    if 'edges_df' not in st.session_state:
        st.session_state.edges_df = None
    if 'poi_df' not in st.session_state:
        st.session_state.poi_df = None
    if 'processed_data' not in st.session_state:
        st.session_state.processed_data = False
    if 'project_random_id' not in st.session_state:
        st.session_state.project_random_id = generate_random_id()

    st.title("Network Generation")
    
    # Add province and directory name selection at the top
    selected_province, directory_name = create_province_selection()
    

    st.write("Pilih lokasi untuk Integrate Module Network Generation")

    # File upload section
    col1, col2 = st.columns(2)
    
    with col1:
        # Create location selection interface with selected province
        selected_df = create_location_selection(selected_province)  # Pass the selected province here
        
        if selected_df is not None and not selected_df.empty:
            st.session_state.poi_df = selected_df
    
    with col2:
        st.subheader("Network Files")
        network_pycgr = st.file_uploader("Upload Network PYCGR File (jakarta.pycgrc)", type=['pycgrc'])
        network_json = st.file_uploader("Upload Network JSON File (jakarta_contracted.json)", type=['json'])
        
        if network_pycgr and network_json:
            try:
                # Save temporary network file
                with tempfile.NamedTemporaryFile(delete=False, suffix='.pycgrc') as tmp_file:
                    tmp_file.write(network_pycgr.getvalue())
                    tmp_path = tmp_file.name
                
                # Process network file
                nodes_df, edges_df = process_network_file(tmp_path)
                
                # Store in session state
                st.session_state.nodes_df = nodes_df
                st.session_state.edges_df = edges_df
                
                st.success("Network files berhasil diproses!")
                
                # Clean up temporary file
                os.unlink(tmp_path)
                
            except Exception as e:
                st.error(f"Error memproses network file: {str(e)}")
    # Check if all required data types are selected
    can_process = False
    if st.session_state.poi_df is not None and not st.session_state.poi_df.empty:
        type_counts = st.session_state.poi_df['type'].value_counts()
        required_types = {'depot', 'shelter', 'village'}
        can_process = all(type_name in type_counts.index for type_name in required_types)

    # Process button
    if st.button("Proses Analisis", disabled=not (st.session_state.poi_df is not None and network_pycgr and network_json)):
        try:
            with st.spinner('Memproses data...'):
                if st.session_state.nodes_df is None or st.session_state.poi_df is None:
                    st.error("Data nodes atau POI belum tersedia. Harap upload semua file yang diperlukan.")
                    return
                    
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                # Processing steps
                steps = ['Loading data', 'Creating subgraph', 'Calculating risks', 'Generating output']
                for idx, step in enumerate(steps):
                    status_text.text(f"Step {idx+1}/{len(steps)}: {step}")
                    progress_bar.progress((idx + 1) * 25)
                    
                    if idx == 0:
                        nodes_df = st.session_state.nodes_df
                        edges_df = st.session_state.edges_df
                        poi_df = st.session_state.poi_df
                        
                    elif idx == 1:
                        subgraph = create_subgraph(poi_df, nodes_df, edges_df)
                        subgraph_nodes = pd.DataFrame([
                            {'id': n, **d} 
                            for n, d in subgraph.nodes(data=True)
                        ])
                        
                    elif idx == 2:
                        risks = calculate_risk(subgraph_nodes)
                        subgraph_nodes['risk'] = risks
                        
                        edges_with_risk = edges_df.merge(
                            subgraph_nodes[['id', 'risk']], 
                            left_on='source', 
                            right_on='id', 
                            suffixes=('', '_source')
                        )
                        edges_with_risk = edges_with_risk.merge(
                            subgraph_nodes[['id', 'risk']], 
                            left_on='target', 
                            right_on='id', 
                            suffixes=('', '_target')
                        )
                        edges_with_risk.rename(columns={
                            'risk': 'risk_source',
                            'risk_target': 'risk_target'
                        }, inplace=True)
                        
                    else:
                        st.session_state.subgraph_nodes = subgraph_nodes
                        st.session_state.edges_with_risk = edges_with_risk
                        st.session_state.processed_data = True

                st.success("Analisis selesai!")
                
        except Exception as e:
            st.error(f"Error during processing: {str(e)}")
            st.write("Debug info:")
            st.write(f"nodes_df available: {st.session_state.nodes_df is not None}")
            st.write(f"poi_df available: {st.session_state.poi_df is not None}")

    # Show results if available
    if st.session_state.get('processed_data', False):
        st.write("### Hasil Analisis")
        
        col_viz1, col_viz2 = st.columns(2)
        
        with col_viz1:
            st.subheader("Distribusi Nilai Risiko")
            fig_hist = px.histogram(
                st.session_state.subgraph_nodes, 
                x='risk',
                title='Distribusi Nilai Risiko',
                labels={'risk': 'Nilai Risiko', 'count': 'Frekuensi'},
                nbins=30
            )
            st.plotly_chart(fig_hist)
            
            st.subheader("Statistik Nilai Risiko")
            risk_stats = st.session_state.subgraph_nodes['risk'].describe()
            st.write(risk_stats)
            
            st.subheader("Distribusi Jenis Jalan berdasarkan Risiko")
            street_type_risk = st.session_state.edges_with_risk.groupby('type')['risk_source'].mean().reset_index()
            fig_pie = px.pie(
                street_type_risk,
                names='type',
                values='risk_source',
                title='Rata-rata Risiko berdasarkan Jenis Jalan'
            )
            st.plotly_chart(fig_pie)
            
        with col_viz2:
            st.subheader("Statistik Nilai Risiko")
            fig_box = px.box(
                st.session_state.subgraph_nodes,
                y='risk',
                title='Statistik Nilai Risiko'
            )
            st.plotly_chart(fig_box)
            
            st.subheader("Rentang Risiko")
            fig_bar = px.bar(
                st.session_state.subgraph_nodes,
                x='id',
                y='risk',
                title='Rentang Risiko untuk Setiap Simpul',
                labels={'id': 'ID Simpul', 'risk': 'Nilai Risiko'},
                color='risk',
                color_continuous_scale='Reds'
            )
            st.plotly_chart(fig_bar)
        
        # Risk map
        st.write("### Peta Risiko")
        try:
            risk_map = create_risk_map(
                st.session_state.subgraph_nodes,
                st.session_state.edges_with_risk,
                st.session_state.poi_df
            )
            
            if risk_map:
                st_folium(risk_map, width=1000, height=600)
                
        except Exception as e:
            st.error(f"Error displaying map: {str(e)}")

        # Summary statistics
        st.write("### Ringkasan Statistik")
        col_sum1, col_sum2, col_sum3 = st.columns(3)
        
        with col_sum1:
            avg_risk = st.session_state.subgraph_nodes['risk'].mean()
            st.metric(
                label="Rata-rata Risiko",
                value=f"{avg_risk:.3f}",
                delta=f"{(avg_risk - 0.5):.3f}",
                delta_color="inverse"
            )
            
        with col_sum2:
            high_risk_count = len(st.session_state.subgraph_nodes[st.session_state.subgraph_nodes['risk'] > 0.7])
            st.metric(
                label="Jumlah Lokasi Risiko Tinggi",
                value=high_risk_count,
                delta=f"{(high_risk_count/len(st.session_state.subgraph_nodes)*100):.1f}%"
            )
            
        with col_sum3:
            edge_count = len(st.session_state.edges_with_risk)
            st.metric(
                label="Jumlah Koneksi",
                value=edge_count
            )

        # Download section
        st.write("### Download Hasil Analisis")
        
        # Prepare download data
        csv_buffer = io.StringIO()
        st.session_state.edges_with_risk.to_csv(csv_buffer, index=False)
        csv_str = csv_buffer.getvalue()
        
        excel_buffer = io.BytesIO()
        with pd.ExcelWriter(excel_buffer, engine='xlsxwriter') as writer:
            st.session_state.edges_with_risk.to_excel(writer, sheet_name='Results', index=False)
            st.session_state.subgraph_nodes.to_excel(writer, sheet_name='Nodes', index=False)
        excel_buffer.seek(0)
        
        # Create download buttons with directory name
        col_down1, col_down2 = st.columns(2)
        
        with col_down1:
            st.download_button(
                label="Download Results (CSV)",
                data=csv_str,
                file_name=f"{directory_name}_results.csv",
                mime="text/csv"
            )
            
        with col_down2:
            st.download_button(
                label="Download Complete Results (Excel)",
                data=excel_buffer,
                file_name=f"{directory_name}_complete_results.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )

if __name__ == "__main__":
    main()
