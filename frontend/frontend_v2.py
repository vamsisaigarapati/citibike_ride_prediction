import sys
from pathlib import Path
import zipfile
import folium
import geopandas as gpd
import pandas as pd
import requests
import streamlit as st
from branca.colormap import LinearColormap
from streamlit_folium import st_folium

parent_dir = str(Path(__file__).parent.parent)
sys.path.append(parent_dir)

from src.config import DATA_DIR
from src.inference import fetch_next_hour_predictions, load_batch_of_features_from_store
from src.plot_utils import plot_prediction

# 👉 Add your station_names dictionary here
station_names = {'': '', 'HB101': 'Hoboken Terminal - Hudson St & Hudson Pl', 'HB102': 'Hoboken Terminal - River St & Hudson Pl', 'HB103': 'South Waterfront Walkway - Sinatra Dr & 1 St', 'HB105': 'City Hall - Washington St & 1 St', 'HB201': '12 St & Sinatra Dr N', 'HB202': '14 St Ferry - 14 St & Shipyard Ln', 'HB203': 'Bloomfield St & 15 St', 'HB301': '4 St & Grand St', 'HB302': '6 St & Grand St', 'HB303': 'Clinton St & 7 St', 'HB304': '7 St & Monroe St', 'HB305': '9 St HBLR - Jackson St & 8 St', 'HB401': 'Southwest Park - Jackson St & Observer Hwy', 'HB402': 'Madison St & 1 St', 'HB404': 'Mama Johnson Field - 4 St & Jackson St', 'HB407': 'Adams St & 2 St', 'HB408': 'Marshall St & 2 St', 'HB409': 'Clinton St & Newark St', 'HB501': 'Columbus Park - Clinton St & 9 St', 'HB502': '11 St & Washington St', 'HB503': 'Madison St & 10 St', 'HB505': 'Willow Ave & 12 St', 'HB506': 'Grand St & 14 St', 'HB601': 'Church Sq Park - 5 St & Park Ave', 'HB602': 'Stevens - River Ter & 6 St', 'HB603': '8 St & Washington St', 'HB608': '2 St & Park Ave', 'HB609': 'River St & 1 St', 'HB610': 'Adams St & 12 St', 'HB611': '4 St & River St', 'JC002': 'Paulus Hook', 'JC003': 'City Hall', 'JC006': 'Warren St', 'JC008': 'Newport Pkwy', 'JC009': 'Hamilton Park', 'JC013': 'Marin Light Rail', 'JC014': 'Columbus Drive', 'JC018': '5 Corners Library', 'JC019': 'Hilltop', 'JC020': 'Baldwin at Montgomery', 'JC022': 'Oakland Ave', 'JC023': 'Brunswick St', 'JC024': 'Pershing Field', 'JC027': 'Jersey & 6th St', 'JC032': 'Newark Ave', 'JC034': 'Christ Hospital', 'JC035': 'Van Vorst Park', 'JC038': 'Essex Light Rail', 'JC051': 'Union St', 'JC052': 'Liberty Light Rail', 'JC053': 'Lincoln Park', 'JC055': 'McGinley Square', 'JC057': 'Riverview Park', 'JC059': 'Heights Elevator', 'JC063': 'Jackson Square', 'JC065': 'Dey St', 'JC066': 'Newport PATH', 'JC072': 'Morris Canal', 'JC074': 'Jersey & 3rd', 'JC075': 'Monmouth and 6th', 'JC076': 'Dixon Mills', 'JC077': 'Astor Place', 'JC078': 'Lafayette Park', 'JC080': 'Leonard Gordon Park', 'JC081': 'Brunswick & 6th', 'JC082': 'Manila & 1st', 'JC084': 'Communipaw & Berry Lane', 'JC093': 'Fairmount Ave', 'JC094': 'Glenwood Ave', 'JC095': 'Bergen Ave', 'JC097': 'York St & Marin Blvd', 'JC098': 'Washington St', 'JC099': 'Montgomery St', 'JC102': 'Grand St', 'JC103': 'Journal Square', 'JC104': 'Harborside', 'JC105': 'Hoboken Ave at Monmouth St', 'JC107': 'Grant Ave & MLK Dr', 'JC108': 'Bergen Ave & Stegman St', 'JC109': 'Bergen Ave & Sip Ave', 'JC110': 'JC Medical Center', 'JC115': 'Grove St PATH', 'JC116': 'Exchange Pl'}  # paste full dictionary

# ------------------ SHAPE FILE ------------------
def load_citibike_shape_file(data_dir, url):
    data_dir = Path(data_dir)
    zip_path = data_dir / "citibike_shape.zip"
    extract_path = data_dir / "citibike_shape"
    if not zip_path.exists():
        response = requests.get(url)
        response.raise_for_status()
        with open(zip_path, "wb") as f:
            f.write(response.content)
    if not any(extract_path.glob("*.shp")):
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(extract_path)
    shapefile = list(extract_path.glob("*.shp"))[0]
    return gpd.read_file(shapefile).to_crs("epsg:4326")

# ------------------ MAPPING ------------------
def create_citibike_map(shapefile_gdf, prediction_data):
    gdf = shapefile_gdf.copy()
    gdf = gdf.merge(
        prediction_data[["pickup_location_id", "predicted_demand"]],
        left_on="stationid",
        right_on="pickup_location_id",
        how="left"
    )
    gdf["predicted_demand"] = gdf["predicted_demand"].fillna(0)
    gdf = gdf.to_crs(epsg=4326)

    m = folium.Map(location=[40.7178, -74.0431], zoom_start=13, tiles="cartodbpositron")

    for _, row in gdf.iterrows():
        lat, lon = row.geometry.centroid.y, row.geometry.centroid.x
        station_id = row["stationid"]
        station_name = station_names.get(station_id, station_id)
        predicted = int(row["predicted_demand"]) if pd.notna(row["predicted_demand"]) else 0

        popup_text = f"🚲 <b>{station_name}</b><br>Predicted Trips: {predicted}"
        folium.CircleMarker(
            location=(lat, lon),
            radius=7,
            color="blue",
            fill=True,
            fill_color="blue",
            fill_opacity=0.7,
            popup=popup_text
        ).add_to(m)

    return m

# ------------------ STREAMLIT APP ------------------
st.set_page_config(page_title="JC Citi Bike Predictions 🚴", page_icon="🚴", layout="wide")

current_date = pd.Timestamp.now(tz="America/New_York")
st.title("🚴 Jersey City Citi Bike Trip Prediction")
st.markdown(f"🕐 **Current Time:** {current_date.strftime('%Y-%m-%d %H:%M:%S')}")

progress = st.sidebar.progress(0)
N_STEPS = 4

# Step 1: Download shapefile
with st.spinner("📥 Downloading Citi Bike station shape file..."):
    shape_url = "https://data.jerseycitynj.gov/api/explore/v2.1/catalog/datasets/citi-bike-locations-phase-1-system-map-3/exports/shp?lang=en&timezone=America%2FNew_York"
    geo_df = load_citibike_shape_file(DATA_DIR, shape_url)
    st.sidebar.success("✅ Shape file downloaded.")
    progress.progress(1 / N_STEPS)

# Step 2: Load features
with st.spinner("📊 Loading feature data from feature store..."):
    features = load_batch_of_features_from_store(current_date)
    st.sidebar.success("✅ Features loaded.")
    progress.progress(2 / N_STEPS)

# Step 3: Load predictions
with st.spinner("🤖 Fetching predictions from latest model..."):
    predictions = fetch_next_hour_predictions()
    st.sidebar.success("✅ Predictions ready.")
    progress.progress(3 / N_STEPS)

# Add station names
predictions["station_name"] = predictions["pickup_location_id"].map(station_names)

# Step 4: Create Tabs
tab1, tab2 = st.tabs(["🗺️ Predicted Demand Map", "📊 Top Stations & Plot"])

# Tab 1 → MAP
with tab1:
    with st.spinner("🗺️ Plotting map..."):
        map_obj = create_citibike_map(geo_df, predictions)
        st_folium(map_obj, width=800, height=600, key="jc_map", returned_objects=[])
    progress.progress(4 / N_STEPS)

# Tab 2 → STATS + PLOT
with tab2:
    st.subheader("📈 Prediction Statistics")
    col1, col2, col3 = st.columns(3)
    col1.metric("Average Trips", f"{predictions['predicted_demand'].mean():.0f}")
    col2.metric("Max Trips", f"{predictions['predicted_demand'].max():.0f}")
    col3.metric("Min Trips", f"{predictions['predicted_demand'].min():.0f}")

    st.markdown("---")
    st.subheader("🏅 Top 10 Stations by Predicted Demand for next 6hours")
    top10 = predictions.sort_values("predicted_demand", ascending=False).head(10)
    st.dataframe(top10[["station_name", "predicted_demand"]])

    st.markdown("---")
    st.subheader("📉 View Time Series Prediction for a Station")
    all_names = predictions["pickup_location_id"].map(station_names).dropna().unique()
    selected_name = st.selectbox("Select Station:", sorted(all_names))
    selected_id = [k for k, v in station_names.items() if v == selected_name][0]

    fig = plot_prediction(
        features=features[features["pickup_location_id"] == selected_id],
        prediction=predictions[predictions["pickup_location_id"] == selected_id],
    )
    st.plotly_chart(fig, theme="streamlit", use_container_width=True)

# Footer
st.markdown("---")
st.caption("Made with ❤️ + 🚴‍♂️ in Jersey City")