import streamlit as st
import pandas as pd
import random
import string
import json

# Constants
PROVINSI_INDONESIA = [
    "ACEH", "SUMATERA UTARA", "SUMATERA BARAT", "RIAU", "JAMBI", "SUMATERA SELATAN",
    "BENGKULU", "LAMPUNG", "KEPULAUAN BANGKA BELITUNG", "KEPULAUAN RIAU", "DKI JAKARTA",
    "JAWA BARAT", "JAWA TENGAH", "DI YOGYAKARTA", "JAWA TIMUR", "BANTEN", "BALI",
    "NUSA TENGGARA BARAT", "NUSA TENGGARA TIMUR", "KALIMANTAN BARAT", "KALIMANTAN TENGAH",
    "KALIMANTAN SELATAN", "KALIMANTAN TIMUR", "KALIMANTAN UTARA", "SULAWESI UTARA",
    "SULAWESI TENGAH", "SULAWESI SELATAN", "SULAWESI TENGGARA", "GORONTALO",
    "SULAWESI BARAT", "MALUKU", "MALUKU UTARA", "PAPUA", "PAPUA BARAT"
]

def generate_random_id(prefix="NG", length=12):
    """Generate a random ID for project identification."""
    return f"{prefix}{''.join(random.choices(string.digits, k=length))}"

def display_network_configuration():
    """Display and handle network configuration inputs."""
    st.header("Integrate Module Network Generation")
    # Initialize random number in session state if not exists
    if 'project_random_id' not in st.session_state:
        st.session_state.project_random_id = generate_random_id()
    # Basic Configuration
    col1, col2 = st.columns(2)
    with col1:
        province = st.selectbox(
            "Base Area (Province)",
            ["--select--"] + PROVINSI_INDONESIA,
            help="Select the province for network generation"
        )
    with col2:
        if province != "--select--":
            # Jika provinsi sudah dipilih, output directory disabled dan terisi otomatis
            output_dir = st.text_input(
                "Output Directory",
                value=f"{st.session_state.project_random_id}_{province.upper().replace(' ', '_')}",
                help="Directory where output files will be saved",
                disabled=True
            )
        else:
            # Jika provinsi belum dipilih, output directory kosong dan disabled
            output_dir = st.text_input(
                "Output Directory",
                value="",
                help="Directory where output files will be saved",
                disabled=True,
                placeholder="Please select province first"
            )
    return province, output_dir

def display_file_upload_section():
    """Handle file uploads and related configurations."""
    st.header("Data Input")
    with st.expander("Required Files", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            osm_file = st.file_uploader(
                "OpenStreetMap File (.osm/.pbf)",
                type=["osm", "pbf"],
                help="Upload OSM/PBF file containing road network data"
            )
            poi_file = st.file_uploader(
                "Point of Interest File (CSV)",
                type=["csv"],
                help="Upload CSV containing locations data (name,type,lat,lon)"
            )
        with col2:
            risk_layer = st.file_uploader(
                "Risk Layer Image (PNG)",
                type=["png"],
                help="Upload risk layer image from INARISK"
            )
            st.markdown("##### Pixel-Coordinate Mapping")
            pixel_coords = st.text_area(
                "Input at least 3 pixel-coordinate pairs",
                placeholder="Format: [x,y] => [lat,lon]\nExample:\n[199,151] => [-6.124142,106.656685]\n[72,392] => [-6.288732,106.569526]",
                help="Enter pixel-to-coordinate mapping points"
            )
    return osm_file, poi_file, risk_layer, pixel_coords

# Inisialisasi data lokasi
LOKASI_DATA = {
    "DKI JAKARTA": {
        "Depot": pd.DataFrame({
            'Nama': ['DKI KANTOR SUDIN JAKARTA BARAT', 'DKI KANTOR SUDIN JAKARTA PUSAT', 'DKI KANTOR SUDIN PEMASARAN', 
                     'DKI KODAM JAYA JAKARTA', 'DKI POS MANGGA DUA', 'DKI POS PASAR BARU', 'DKI SENTRAL CIKINI ARENG', 
                     'DKI SENTRAL KEBUN JERUK', 'DKI SUDIN JAKARTA UTARA', 'DKI SUDIN PEMASARAN JAKARTA SELATAN'], 
            'Latitude': ['6.1755498', '6.1675599999999995', '6.2101452', '-6.2351861', '6.1582349', '6.1585127', 
                         '6.1957525', '6.1951185', '6.1277512000000001', '6.2614827'],
            'Longitude': ['106.7800956', '106.8291115', '106.8388871', '106.8040072', '106.8248552', '106.8514085', 
                          '106.8299062', '106.774407', '106.8267167', '106.8109566'],
            'Type': ['Depot'] * 10
        }),
        "Shelter": pd.DataFrame({
            'Nama': ['LT', 'PT DUTA KARYA BUANA', 'SMA MAARIF', 'SPBU BOULEVARD', 'SPBU BUNCIT', 'SPBU GREEN ANDARA', 
                     'SPBU GROGOL', 'SPBU KEDOYA', 'SPBU SETIABUDI', 'SPBU UTAN KAYU'],
            'Latitude': ['6.215213', '6.212561', '6.234245', '6.227517', '6.262056', '6.234725', '-6.16452', 
                         '6.165466', '6.226174', '6.19808'],
            'Longitude': ['106.8194274', '106.7842794', '106.7945289', '106.8048651', '106.8064697', '106.8365462', 
                          '106.7923954', '106.759898', '106.8197527', '106.8743577'],
            'Type': ['Shelter'] * 10
        }),
        "Village": pd.DataFrame({
            'Nama': ['ANCO', 'ANDRE', 'SAMSU ARDI', 'BODAS JAYA', 'BUKIT DURI', 'DKI Cipinang Melayu', 'DKI Dukuh', 
                     'DKI Gunung', 'DKI North Pondok Bambu', 'DKI Pondok Kopi'],
            'Latitude': ['6.142513', '6.1542245', '6.1131556', '6.2210164', '6.225043', '6.232508', '6.227512', 
                         '6.260861', '6.2342239999999995', '6.3374549999999995'],
            'Longitude': ['106.8362474', '106.7942754', '106.7608726', '106.8064951', '106.8504297', '106.9152327', 
                          '106.9132801', '106.8625881', '106.8977387', '106.8927736'],
            'Type': ['Village'] * 10
        })
    }
}

def convert_selections_to_json(selected_locations):
    """Convert selected locations to JSON format."""
    json_data = {
        "selected_locations": {
            "depot": [],
            "shelter": [],
            "village": []
        },
        "summary": {
            "total_depot": len(selected_locations['Depot']),
            "total_shelter": len(selected_locations['Shelter']),
            "total_village": len(selected_locations['Village']),
            "total_locations": sum(len(locations) for locations in selected_locations.values())
        }
    }
    
    # Add locations data
    for location_type in ['Depot', 'Shelter', 'Village']:
        for location in selected_locations[location_type]:
            location_data = {
                "name": location['Nama'],
                "latitude": float(location['Latitude']),
                "longitude": float(location['Longitude']),
                "type": location['Type']
            }
            json_data["selected_locations"][location_type.lower()].append(location_data)
    
    return json_data

def display_location_management(province):
    """Handle location data management interface."""
    st.header("Master Files")
    if province == "--select--":
        st.warning("Please select a province first")
        return
 #Add Evacuation Percentage field
    evac_percentage = st.number_input(
        "Evacuation Percentage (%)", 
        min_value=0, 
        max_value=100, 
        value=0,
        help="Enter the evacuation percentage",
        key="evac_percentage"
    )
    st.markdown("---")
    # Initialize session states
    if 'show_form' not in st.session_state:
        st.session_state.show_form = {
            'Depot': False,
            'Shelter': False,
            'Village': False
        }
    
    if 'temp_data' not in st.session_state:
        st.session_state.temp_data = {
            'Depot': None,
            'Shelter': None,
            'Village': None
        }
    
    # Initialize selected locations session state
    if 'selected_locations' not in st.session_state:
        st.session_state.selected_locations = {
            'Depot': [],
            'Shelter': [],
            'Village': []
        }
    
    # Function to update selected locations
    def update_selection(location_type, selected_indices, data):
        st.session_state.selected_locations[location_type] = [
            data.iloc[idx] for idx in selected_indices
        ]

    tabs = st.tabs(["Depot", "Shelter", "Village"])
    for tab, location_type in zip(tabs, ["Depot", "Shelter", "Village"]):
        with tab:
            st.subheader(f"{location_type} Locations")
            
            if province in LOKASI_DATA and location_type in LOKASI_DATA[province]:
                data = LOKASI_DATA[province][location_type]
                
                # Search and filter
                col1, col2 = st.columns([3, 1])
                with col1:
                    search = st.text_input(
                        "Search locations",
                        key=f"search_{location_type}",
                        placeholder="Enter name to search..."
                    )
                with col2:
                    st.write("")  # Spacing
                    total_locations = len(data)
                    st.info(f"Total {location_type}s: {total_locations}")

                # Filter data based on search
                if search:
                    data = data[data['Nama'].str.contains(search, case=False)]

                # Display data with selection
                st.write("Select locations:")
                selected_rows = []
                for idx, row in data.iterrows():
                    col1, col2, col3, col4 = st.columns([0.5, 2, 1, 1])
                    with col1:
                        is_selected = st.checkbox("", key=f"select_{location_type}_{idx}")
                        if is_selected:
                            selected_rows.append(idx)
                    with col2:
                        st.write(row['Nama'])
                    with col3:
                        st.write(f"Lat: {row['Latitude']}")
                    with col4:
                        st.write(f"Long: {row['Longitude']}")
                
                # Update selected locations
                update_selection(location_type, selected_rows, data)

                # Display selected count
                st.info(f"Selected {location_type}s: {len(selected_rows)}")

                # Action buttons
                col1, col2, col3 = st.columns([1, 1, 1])
                with col1:
                    if st.button(f"Export {location_type}", key=f"export_{location_type}"):
                        st.download_button(
                            label=f"Download {location_type} Data",
                            data=data.to_csv(index=False).encode('utf-8'),
                            file_name=f"{location_type.lower()}_locations.csv",
                            mime='text/csv',
                            key=f"download_{location_type}"
                        )
                with col2:
                    if st.button(f"Add New {location_type}", key=f"add_{location_type}"):
                        st.session_state.show_form[location_type] = True
                with col3:
                    if st.button(f"View Map", key=f"map_{location_type}"):
                        st.info(f"Map view for {location_type} locations will be implemented soon")

                # Form untuk menambah lokasi baru
                if st.session_state.show_form[location_type]:
                    with st.form(key=f'add_{location_type}_form'):
                        st.subheader(f"Add New {location_type}")
                        new_name = st.text_input("Location Name")
                        col1, col2 = st.columns(2)
                        with col1:
                            new_lat = st.text_input("Latitude")
                        with col2:
                            new_lon = st.text_input("Longitude")
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            if st.form_submit_button("Save", type="primary"):
                                if new_name and new_lat and new_lon:
                                    new_row = pd.DataFrame({
                                        'Nama': [new_name],
                                        'Latitude': [new_lat],
                                        'Longitude': [new_lon],
                                        'Type': [location_type]
                                    })
                                    st.session_state.temp_data[location_type] = pd.concat([
                                        LOKASI_DATA[province][location_type],
                                        new_row
                                    ], ignore_index=True)
                                    st.success(f"New {location_type} location saved successfully!")
                                    st.session_state.show_form[location_type] = False
                                    st.rerun()
                                else:
                                    st.error("Please fill in all fields")
                        with col2:
                            if st.form_submit_button("Close"):
                                st.session_state.show_form[location_type] = False
                                st.rerun()

            else:
                st.info(f"No {location_type} data available for {province}")

    # Display summary and JSON export section
    if any(len(locations) > 0 for locations in st.session_state.selected_locations.values()):
        st.header("Selected Locations Summary")
        
        # Convert selections to JSON
        json_data = convert_selections_to_json(st.session_state.selected_locations)
        
        # Display summary
        col1, col2 = st.columns([2, 1])
        with col1:
            st.json(json_data)
        with col2:
            st.download_button(
                label="Download Selections as JSON",
                data=json.dumps(json_data, indent=2),
                file_name=f"selected_locations_{province.lower().replace(' ', '_')}.json",
                mime="application/json",
                help="Download all selected locations in JSON format"
            )
            
            if st.button("Save Selections", type="primary"):
                # Here you can add logic to save the JSON data to a database or file
                st.success("Selections saved successfully!")
                
                # Display summary counts
                st.write("Summary:")
                for key, value in json_data["summary"].items():
                    st.write(f"- {key.replace('_', ' ').title()}: {value}")

    # Display summary of all selected locations
    st.header("Selected Locations Summary")
    for location_type in ["Depot", "Shelter", "Village"]:
        if st.session_state.selected_locations[location_type]:
            st.subheader(f"Selected {location_type}s ({len(st.session_state.selected_locations[location_type])})")
            selected_df = pd.DataFrame(st.session_state.selected_locations[location_type])
            st.dataframe(
                selected_df[['Nama', 'Latitude', 'Longitude', 'Type']],
                hide_index=True
            )

    # Save all selections button
    if any(len(locations) > 0 for locations in st.session_state.selected_locations.values()):
        if st.button("Save All Selections", type="primary"):
            # Here you can implement the save logic
            # For now, we'll just show a success message
            st.success("All selections saved successfully!")
            
            # Create a summary of selections
            summary = {
                'Depot': len(st.session_state.selected_locations['Depot']),
                'Shelter': len(st.session_state.selected_locations['Shelter']),
                'Village': len(st.session_state.selected_locations['Village'])
            }
            
            st.json(summary)
def display_process_control(province, output_dir, osm_file, poi_file, risk_layer, pixel_coords):
    """Display process control and monitoring interface."""
    st.header("Process Control")
    # Configuration preview
    with st.expander("Configuration Preview", expanded=False):
        config = {
            "name": province.lower().replace(' ', '_'),
            "output_dir": output_dir,
            "network_pycgr_file": "path/to/network.pycgrc",
            "poi_file": "path/to/locations.csv",
            "risk_layer_file": "path/to/risk.png",
            "risk_coordinates_samples": [
                [[199, 151], [-6.124142, 106.656685]],
                [[72, 392], [-6.288732, 106.569526]]
            ]
        }
        st.code(json.dumps(config, indent=2))
    # Process execution
    col1, col2 = st.columns([2, 1])
    with col1:
        if st.button("Generate Subnetwork", type="primary"):
            if province == "--select--":
                st.warning("Please select a province first")
            elif not osm_file:
                st.warning("Please upload an OSM file")
            elif not poi_file:
                st.warning("Please upload a POI CSV file")
            elif not risk_layer:
                st.warning("Please upload a risk layer image")
            elif not pixel_coords:
                st.warning("Please input pixel-coordinate pairs")
            else:
                with st.spinner("Generating subnetwork..."):
                    # Add actual process execution
                    success = True  # Replace with actual backend logic
                    if success:
                        st.success("Subnetwork generated successfully!")
                        st.markdown("[View Result](#dashboard)", unsafe_allow_html=True)
                    else:
                        st.error("Failed to generate subnetwork. Please check input files and settings.")
    with col2:
        st.info("Estimated processing time: 5-10 minutes")

def display_status_monitoring():
    """Display process status and monitoring information."""
    st.header("Status Monitor")
    # Status indicators
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(
            label="Processing Status",
            value="Ready",
            delta="Idle"
        )
    with col2:
        st.metric(
            label="Memory Usage",
            value="0 MB",
            delta="0%"
        )
    with col3:
        st.metric(
            label="Processing Time",
            value="0:00",
            delta="0 sec"
        )

def display_network_visualization():
    """Display network visualization interface."""
    # Header with navigation-like display
    col_nav1, col_nav2, col_nav3 = st.columns([0.1, 2, 0.2])
    with col_nav1:
        st.markdown("📊")
    with col_nav2:
        st.markdown('<span style="color: #0D6EFD;">Network Generation</span>', unsafe_allow_html=True)
    with col_nav3:
        st.button("+ New", type="primary", key="new_btn")
    
    # Available Network Generations banner with icon
    st.markdown("""
        <div style='background-color: #4CAF50; color: white; padding: 8px 16px; border-radius: 4px; margin: 10px 0;'>
            <span style="display: flex; align-items: center;">
                <span style="margin-right: 8px;">📊</span>
                Available Network Generations
            </span>
        </div>
    """, unsafe_allow_html=True)
    
    # Initialize session state for data if not exists
    if 'network_data' not in st.session_state:
        st.session_state.network_data = {
            'No.': range(1, 6),
            'Date Created': ['13 Apr 2022', '12 May 2022', '05 Mar 2023', '12 Mar 2023', '31 May 2023'],
            'Province Name': ['NUSA TENGGARA BARAT', 'NUSA TENGGARA BARAT', 'DKI JAKARTA', 
                            'NUSA TENGGARA BARAT', 'NUSA TENGGARA BARAT'],
            'Evacuation Percentage(%)': [100, 15, 90, 30, 100],
            'Number Of POI': [17, 20, 9, 7, 17],
            'Network Code': ['NG1648821496_NUSA_TENGGARA_BARAT', 'NG1652326949_NUSA_TENGGARA_BARAT',
                           'NG1677985532_DKI_JAKARTA', 'NG1678582276_NUSA_TENGGARA_BARAT',
                           'NG23053107134Q_NUSA_TENGGARA_BARAT'],
            'User': ['admin', 'mr_toto', 'admin', 'user07', 'admin']
        }
    
    # Search and filter section with improved styling
    col1, col2, col3, col4 = st.columns([3, 1.2, 1.2, 0.8])
    with col1:
        search = st.text_input("", placeholder="Keyword", key="search_input")
    with col2:
        column_filter = st.selectbox("", [
            "-- Column --",
            "No.",
            "Date Created",
            "Province Name",
            "Evacuation Percentage(%)",
            "Number Of POI",
            "Network Code",
            "User"
        ], key="column_filter")
    with col3:
        direction = st.selectbox("", 
            ["-- Direction --", "Ascending", "Descending"],
            key="direction_filter"
        )
    with col4:
        col4_1, col4_2 = st.columns([1, 1])
        with col4_1:
            search_clicked = st.button("🔍", key="search_btn")
        with col4_2:
            reset_clicked = st.button("✖️", key="reset_btn")

    # Create DataFrame
    df = pd.DataFrame(st.session_state.network_data)

    # Apply filters when search button is clicked
    if search_clicked:
        # Text search
        if search:
            mask = df.astype(str).apply(lambda x: x.str.contains(search, case=False)).any(axis=1)
            df = df[mask]
        
        # Column sorting
        if column_filter != "-- Column --" and direction != "-- Direction --":
            ascending = direction == "Ascending"
            
            # Special handling for date column
            if column_filter == "Date Created":
                df['Date Created'] = pd.to_datetime(df['Date Created'], format='%d %b %Y')
                df = df.sort_values(column_filter, ascending=ascending)
                df['Date Created'] = df['Date Created'].dt.strftime('%d %b %Y')
            else:
                df = df.sort_values(column_filter, ascending=ascending)

    # Reset filters
    if reset_clicked:
        df = pd.DataFrame(st.session_state.network_data)
        search = ""
        column_filter = "-- Column --"
        direction = "-- Direction --"
        st.experimental_rerun()

    # Update Custom CSS untuk struktur vertikal button
    st.markdown("""
        <style>
        .action-container {
            display: flex;
            flex-direction: column;
            gap: 4px;
        }
        .visualization-btn {
            background-color: #FFA500;
            color: black;
            padding: 4px 8px;
            border-radius: 4px;
            text-decoration: none;
            font-size: 12px;
            cursor: pointer;
            display: flex;
            align-items: center;
            border: none;
            width: 100%;
        }
        .download-dropdown {
            position: relative;
            display: block;
            width: 100%;
        }
        .download-btn {
            background-color: white;
            color: #0D6EFD;
            padding: 4px 8px;
            border-radius: 4px;
            text-decoration: none;
            font-size: 12px;
            cursor: pointer;
            display: flex;
            align-items: center;
            border: 1px solid #0D6EFD;
            width: 100%;
            justify-content: space-between;
        }
        .dropdown-content {
            display: none;
            position: absolute;
            background-color: white;
            min-width: 100%;
            box-shadow: 0px 8px 16px 0px rgba(0,0,0,0.2);
            z-index: 1;
            border-radius: 4px;
            margin-top: 4px;
        }
        .download-dropdown:hover .dropdown-content {
            display: block;
        }
        .dropdown-item {
            color: black;
            padding: 8px 12px;
            text-decoration: none;
            display: block;
            font-size: 12px;
        }
        .dropdown-item:hover {
            background-color: #f8f9fa;
        }
        </style>
    """, unsafe_allow_html=True)
    
        # Update fungsi create_action_buttons untuk menghilangkan newlines
    def create_action_buttons(network_code):
        return (
            '<div class="action-container">'
            '<button class="visualization-btn">📊 Visualization</button>'
            '<div class="download-dropdown">'
            '<button class="download-btn">Download Archives <span>▾</span></button>'
            '<div class="dropdown-content">'
            '<a href="#" class="dropdown-item">Input Files</a>'
            '<a href="#" class="dropdown-item">Output Files</a>'
            '</div>'
            '</div>'
            '</div>'
        )
    
    df['Actions'] = df['Network Code'].apply(create_action_buttons)
    
    # Display table
    st.write(df.to_html(
        escape=False,
        index=False,
        classes='stDataFrame',
        table_id='network_table'
        ).replace('\n', ''), # Menghilangkan semua newlines dari output HTML 
        unsafe_allow_html=True)

def main():
    st.set_page_config(
        page_title="RespondOR Network Generation",
        page_icon="🌐",
        layout="wide"
    )  
    st.title("RespondOR Network Generation")
    # Main navigation
    tab_network, tab_visualization = st.tabs(["Network Generation", "Network Visualization"])
    with tab_network:
        # Left column - Main workflow
        col_main, col_status = st.columns([2, 1])
        with col_main:
            province, output_dir = display_network_configuration()
            # Mengubah urutan eksekusi
            display_location_management(province)  # Dipindah ke atas
            osm_file, poi_file, risk_layer, pixel_coords = display_file_upload_section()  # Dipindah ke bawah
            display_process_control(province, output_dir, osm_file, poi_file, risk_layer, pixel_coords)
        with col_status:
            display_status_monitoring()
    with tab_visualization:
        st.header("Network Visualization")
        display_network_visualization()

        #st.components.v1.iframe(
       #     "http://localhost:5000",
       #     height=800,
       #     scrolling=True
      # )

if __name__ == "__main__":
    main()