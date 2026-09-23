import streamlit as st
import pandas as pd
import joblib
import folium
import requests
import math
import time

from streamlit_folium import st_folium
from pathlib import Path


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="AI Multi-Disaster Response",
    page_icon="🚨",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# LOAD CSS
# ============================================================

css_file = Path("static/style.css")

if css_file.exists():

    css = css_file.read_text(
        encoding="utf-8"
    )

    st.markdown(
        f"<style>{css}</style>",
        unsafe_allow_html=True
    )


# ============================================================
# LOAD MULTI-DISASTER MODELS
# ============================================================

models = joblib.load(
    "model/risk_model.pkl"
)


# ============================================================
# CONSTANTS
# ============================================================

NOMINATIM_URL = (
    "https://nominatim.openstreetmap.org"
)

OSRM_URL = (
    "https://router.project-osrm.org"
)

USER_AGENT = (
    "AI-Multi-Disaster-Response-Project/1.0"
)


# ============================================================
# GEOCODE LOCATION
# ============================================================

def geocode_location(location):

    url = (
        f"{NOMINATIM_URL}/search"
    )

    params = {
        "q": location,
        "format": "json",
        "limit": 1,
        "addressdetails": 1
    }

    headers = {
        "User-Agent": USER_AGENT
    }

    try:

        response = requests.get(
            url,
            params=params,
            headers=headers,
            timeout=10
        )

        response.raise_for_status()

        results = response.json()

        if not results:
            return None

        result = results[0]

        return {
            "lat": float(result["lat"]),
            "lon": float(result["lon"]),
            "display_name": result["display_name"],
            "address": result.get(
                "address",
                {}
            )
        }

    except requests.RequestException:

        return None


# ============================================================
# REVERSE GEOCODE
# ============================================================

def reverse_geocode(lat, lon):

    url = (
        f"{NOMINATIM_URL}/reverse"
    )

    params = {
        "lat": lat,
        "lon": lon,
        "format": "jsonv2",
        "zoom": 10,
        "addressdetails": 1
    }

    headers = {
        "User-Agent": USER_AGENT
    }

    try:

        response = requests.get(
            url,
            params=params,
            headers=headers,
            timeout=10
        )

        response.raise_for_status()

        result = response.json()

        if not result:
            return None

        return {
            "display_name": result.get(
                "display_name",
                ""
            ),
            "address": result.get(
                "address",
                {}
            )
        }

    except requests.RequestException:

        return None


# ============================================================
# DESTINATION POINT
# ============================================================

def destination_point(
    lat,
    lon,
    distance_km,
    bearing
):

    earth_radius_km = 6371.0

    lat1 = math.radians(lat)
    lon1 = math.radians(lon)
    bearing_rad = math.radians(bearing)

    distance_ratio = (
        distance_km /
        earth_radius_km
    )

    lat2 = math.asin(
        math.sin(lat1)
        *
        math.cos(distance_ratio)

        +

        math.cos(lat1)
        *
        math.sin(distance_ratio)
        *
        math.cos(bearing_rad)
    )

    lon2 = (
        lon1

        +

        math.atan2(
            math.sin(bearing_rad)
            *
            math.sin(distance_ratio)
            *
            math.cos(lat1),

            math.cos(distance_ratio)
            -

            math.sin(lat1)
            *
            math.sin(lat2)
        )
    )

    return (
        math.degrees(lat2),
        math.degrees(lon2)
    )


# ============================================================
# CHECK LAND LOCATION
# ============================================================

def is_land_location(
    lat,
    lon
):

    result = reverse_geocode(
        lat,
        lon
    )

    if result is None:
        return False

    address = result.get(
        "address",
        {}
    )

    country_code = (
        address
        .get(
            "country_code",
            ""
        )
        .lower()
    )

    if country_code == "in":
        return True

    return False


# ============================================================
# DETECT INLAND DIRECTION
# ============================================================

def detect_inland_bearing(
    lat,
    lon
):

    directions = {
        "N": 0,
        "NE": 45,
        "E": 90,
        "SE": 135,
        "S": 180,
        "SW": 225,
        "W": 270,
        "NW": 315
    }

    water_directions = []
    land_directions = []

    test_distance = 15

    for name, bearing in directions.items():

        test_lat, test_lon = (
            destination_point(
                lat,
                lon,
                test_distance,
                bearing
            )
        )

        land = is_land_location(
            test_lat,
            test_lon
        )

        if land:

            land_directions.append(
                name
            )

        else:

            water_directions.append(
                name
            )

        time.sleep(0.2)

    if water_directions:

        water_bearings = [
            directions[d]
            for d in water_directions
        ]

        best_inland_bearing = None
        best_score = -1

        for water_bearing in water_bearings:

            inland_bearing = (
                water_bearing + 180
            ) % 360

            inland_lat, inland_lon = (
                destination_point(
                    lat,
                    lon,
                    20,
                    inland_bearing
                )
            )

            if is_land_location(
                inland_lat,
                inland_lon
            ):

                score = 2

            else:

                score = 0

            if score > best_score:

                best_score = score

                best_inland_bearing = (
                    inland_bearing
                )

            time.sleep(0.2)

        if best_inland_bearing is not None:

            return best_inland_bearing

    if land_directions:

        return directions[
            land_directions[0]
        ]

    return 270


# ============================================================
# CREATE EMERGENCY LOCATIONS
# ============================================================

def create_emergency_locations(
    disaster_lat,
    disaster_lon
):

    inland_bearing = (
        detect_inland_bearing(
            disaster_lat,
            disaster_lon
        )
    )

    rescue_lat, rescue_lon = (
        destination_point(
            disaster_lat,
            disaster_lon,
            30,
            inland_bearing
        )
    )

    camp_lat, camp_lon = (
        destination_point(
            disaster_lat,
            disaster_lon,
            35,
            inland_bearing
        )
    )

    hospital_lat, hospital_lon = (
        destination_point(
            disaster_lat,
            disaster_lon,
            38,
            inland_bearing
        )
    )

    if not is_land_location(
        rescue_lat,
        rescue_lon
    ):

        rescue_lat, rescue_lon = (
            destination_point(
                disaster_lat,
                disaster_lon,
                25,
                inland_bearing
            )
        )

    if not is_land_location(
        camp_lat,
        camp_lon
    ):

        camp_lat, camp_lon = (
            destination_point(
                disaster_lat,
                disaster_lon,
                30,
                inland_bearing
            )
        )

    if not is_land_location(
        hospital_lat,
        hospital_lon
    ):

        hospital_lat, hospital_lon = (
            destination_point(
                disaster_lat,
                disaster_lon,
                35,
                inland_bearing
            )
        )

    return {

        "rescue": {
            "lat": rescue_lat,
            "lon": rescue_lon
        },

        "camp": {
            "lat": camp_lat,
            "lon": camp_lon
        },

        "hospital": {
            "lat": hospital_lat,
            "lon": hospital_lon
        },

        "inland_bearing":
            inland_bearing
    }


# ============================================================
# ROAD ROUTING
# ============================================================

def get_route(
    start_lat,
    start_lon,
    end_lat,
    end_lon
):

    url = (
        f"{OSRM_URL}/route/v1/driving/"
        f"{start_lon},{start_lat};"
        f"{end_lon},{end_lat}"
    )

    params = {
        "overview": "full",
        "geometries": "geojson"
    }

    try:

        response = requests.get(
            url,
            params=params,
            timeout=20
        )

        response.raise_for_status()

        data = response.json()

        if data.get("code") != "Ok":
            return None

        if not data.get("routes"):
            return None

        route = data["routes"][0]

        coordinates = (
            route["geometry"]
            ["coordinates"]
        )

        folium_coordinates = [

            [
                coordinate[1],
                coordinate[0]
            ]

            for coordinate in coordinates
        ]

        distance_km = (
            route["distance"]
            / 1000
        )

        duration_minutes = (
            route["duration"]
            / 60
        )

        return {

            "coordinates":
                folium_coordinates,

            "distance_km":
                distance_km,

            "duration_minutes":
                duration_minutes
        }

    except requests.RequestException:

        return None


# ============================================================
# DISASTER-SPECIFIC RECOMMENDATIONS
# ============================================================

def get_recommendations(
    disaster_type,
    prediction
):

    recommendations = {

        "Flood": {

            "High": [
                "Deploy emergency rescue teams immediately.",
                "Prepare nearby medical facilities.",
                "Arrange emergency food and water supplies.",
                "Issue evacuation warnings.",
                "Monitor flood conditions continuously."
            ],

            "Medium": [
                "Keep emergency response teams ready.",
                "Monitor rainfall and water levels closely.",
                "Prepare essential emergency supplies.",
                "Check nearby relief facilities.",
                "Prepare for possible evacuation."
            ],

            "Low": [
                "Continue monitoring rainfall and water levels.",
                "Maintain emergency preparedness.",
                "Keep essential resources available.",
                "Monitor changes in weather conditions."
            ]
        },

        "Cyclone": {

            "High": [
                "Deploy emergency response teams immediately.",
                "Issue evacuation warnings for vulnerable areas.",
                "Prepare cyclone shelters and medical facilities.",
                "Secure emergency food and water supplies.",
                "Continuously monitor wind and weather conditions."
            ],

            "Medium": [
                "Keep emergency teams on standby.",
                "Monitor wind speed and rainfall closely.",
                "Prepare cyclone shelters and essential supplies.",
                "Check evacuation routes.",
                "Prepare for possible evacuation."
            ],

            "Low": [
                "Continue monitoring weather conditions.",
                "Maintain emergency preparedness.",
                "Keep emergency supplies ready.",
                "Monitor changes in wind and rainfall."
            ]
        },

        "Earthquake": {

            "High": [
                "Deploy search and rescue teams immediately.",
                "Prepare emergency medical facilities.",
                "Evacuate unsafe or damaged structures.",
                "Arrange emergency food and water supplies.",
                "Monitor aftershocks and affected areas."
            ],

            "Medium": [
                "Keep rescue teams on standby.",
                "Inspect vulnerable structures.",
                "Prepare medical and emergency supplies.",
                "Identify safe assembly areas.",
                "Prepare for possible evacuation."
            ],

            "Low": [
                "Continue monitoring seismic conditions.",
                "Maintain emergency preparedness.",
                "Identify nearby safe locations.",
                "Keep essential emergency supplies available."
            ]
        }
    }

    return recommendations[
        disaster_type
    ][prediction]


# ============================================================
# RESOURCE ALLOCATION
# ============================================================

def get_resources(
    disaster_type,
    prediction
):

    if prediction == "High":

        rescue_teams = 5
        ambulances = 4
        food_kits = 2500
        water_units = 2000

    elif prediction == "Medium":

        rescue_teams = 3
        ambulances = 2
        food_kits = 1500
        water_units = 1200

    else:

        rescue_teams = 2
        ambulances = 1
        food_kits = 750
        water_units = 600

    return (
        rescue_teams,
        ambulances,
        food_kits,
        water_units
    )


# ============================================================
# SESSION STATE
# ============================================================

if "prediction_done" not in st.session_state:

    st.session_state.prediction_done = False


if "prediction" not in st.session_state:

    st.session_state.prediction = None


if "confidence" not in st.session_state:

    st.session_state.confidence = 0.0


if "disaster_type" not in st.session_state:

    st.session_state.disaster_type = "Flood"


if "location_data" not in st.session_state:

    st.session_state.location_data = None


if "route_data" not in st.session_state:

    st.session_state.route_data = None


if "emergency_locations" not in st.session_state:

    st.session_state.emergency_locations = None


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown(
        """
        <div style="padding: 10px 4px 18px 4px;">
            <div style="font-size: 20px; font-weight: 800; color: #ffffff;">
                🚨 Disaster Response
            </div>
            <div style="font-size: 12px; color: #94a3b8; margin-top: 6px; line-height: 1.5;">
                AI-powered emergency decision support
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.divider()

    st.subheader(
        "🌦️ Disaster Information"
    )

    disaster_type = st.selectbox(
        "Disaster Type",

        [
            "Flood",
            "Cyclone",
            "Earthquake"
        ]
    )


    # ========================================================
    # FLOOD INPUTS
    # ========================================================

    if disaster_type == "Flood":

        rainfall = st.number_input(
            "Rainfall (mm)",
            min_value=0.0,
            max_value=500.0,
            value=180.0
        )

        water_level = st.number_input(
            "Water Level (m)",
            min_value=0.0,
            max_value=15.0,
            value=7.0
        )

        wind_speed = st.number_input(
            "Wind Speed (km/h)",
            min_value=0.0,
            max_value=200.0,
            value=80.0
        )


    # ========================================================
    # CYCLONE INPUTS
    # ========================================================

    elif disaster_type == "Cyclone":

        wind_speed = st.number_input(
            "Wind Speed (km/h)",
            min_value=0.0,
            max_value=250.0,
            value=140.0
        )

        rainfall = st.number_input(
            "Rainfall (mm)",
            min_value=0.0,
            max_value=500.0,
            value=250.0
        )

        atmospheric_pressure = st.number_input(
            "Atmospheric Pressure (hPa)",
            min_value=900.0,
            max_value=1020.0,
            value=970.0
        )


    # ========================================================
    # EARTHQUAKE INPUTS
    # ========================================================

    else:

        magnitude = st.number_input(
            "Earthquake Magnitude",
            min_value=2.0,
            max_value=8.0,
            value=6.5
        )

        depth = st.number_input(
            "Depth (km)",
            min_value=1.0,
            max_value=100.0,
            value=10.0
        )

        seismic_intensity = st.number_input(
            "Seismic Intensity",
            min_value=1.0,
            max_value=10.0,
            value=7.0
        )


    # ========================================================
    # COMMON INPUTS
    # ========================================================

    population_density = st.number_input(
        "Population Density (people/km²)",
        min_value=0,
        max_value=3000,
        value=800
    )

    previous_disasters = st.number_input(
        "Previous Disaster Frequency",
        min_value=0,
        max_value=10,
        value=3
    )


    st.divider()


    # ========================================================
    # LOCATION
    # ========================================================

    st.subheader(
        "📍 Disaster Location"
    )

    location_text = st.text_input(
        "Enter a place or address",
        placeholder="Example: Chennai Railway Station"
    )

    st.caption(
        "Enter the affected location. "
        "The system will automatically "
        "find its coordinates."
    )

    st.write("")


    # ========================================================
    # PREDICT BUTTON
    # ========================================================

    predict_button = st.button(
        "🔍 PREDICT DISASTER RISK",
        type="primary",
        use_container_width=True
    )

    st.divider()

    st.caption(
        "Enter disaster conditions and "
        "location to generate AI-based "
        "risk and emergency response "
        "recommendations."
    )


# ============================================================
# MAIN HEADER
# ============================================================

st.markdown(
    """
    <div class="dashboard-header">
        <div class="dashboard-title">
            AI-ASSISTED DISASTER RESPONSE<br>
            AND RESOURCE RECOMMENDATION SYSTEM
        </div>
        <div class="dashboard-subtitle">
            AI-powered risk assessment, emergency response planning, resource recommendation and interactive evacuation mapping
        </div>
    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# PREDICTION
# ============================================================

if predict_button:

    # --------------------------------------------------------
    # Validate Location
    # --------------------------------------------------------

    if not location_text.strip():

        st.error(
            "Please enter a disaster location "
            "before predicting the risk."
        )

        st.session_state.prediction_done = False


    else:

        # ----------------------------------------------------
        # Geocode Location
        # ----------------------------------------------------

        with st.spinner(
            "📍 Finding disaster location..."
        ):

            location_data = geocode_location(
                location_text.strip()
            )


        if location_data is None:

            st.error(
                "❌ Location could not be found. "
                "Please enter a more specific "
                "place or address."
            )

            st.session_state.prediction_done = False


        else:

            st.session_state.location_data = (
                location_data
            )


            # ------------------------------------------------
            # SELECT CORRECT MODEL
            # ------------------------------------------------

            selected_model = models[
                disaster_type
            ]


            # ------------------------------------------------
            # PREPARE MODEL INPUT
            # ------------------------------------------------

            if disaster_type == "Flood":

                input_data = pd.DataFrame({

                    "rainfall": [
                        rainfall
                    ],

                    "water_level": [
                        water_level
                    ],

                    "wind_speed": [
                        wind_speed
                    ],

                    "population_density": [
                        population_density
                    ],

                    "previous_disasters": [
                        previous_disasters
                    ]
                })


            elif disaster_type == "Cyclone":

                input_data = pd.DataFrame({

                    "wind_speed": [
                        wind_speed
                    ],

                    "rainfall": [
                        rainfall
                    ],

                    "atmospheric_pressure": [
                        atmospheric_pressure
                    ],

                    "population_density": [
                        population_density
                    ],

                    "previous_disasters": [
                        previous_disasters
                    ]
                })


            else:

                input_data = pd.DataFrame({

                    "magnitude": [
                        magnitude
                    ],

                    "depth": [
                        depth
                    ],

                    "seismic_intensity": [
                        seismic_intensity
                    ],

                    "population_density": [
                        population_density
                    ],

                    "previous_disasters": [
                        previous_disasters
                    ]
                })


            # ------------------------------------------------
            # ML PREDICTION
            # ------------------------------------------------

            prediction = selected_model.predict(
                input_data
            )[0]

            probabilities = selected_model.predict_proba(
                input_data
            )[0]

            probability_dict = dict(
                zip(
                    selected_model.classes_,
                    probabilities
                )
            )

            confidence = (
                probability_dict[prediction]
                * 100
            )


            # ------------------------------------------------
            # SAVE PREDICTION
            # ------------------------------------------------

            st.session_state.prediction = (
                prediction
            )

            st.session_state.confidence = (
                confidence
            )

            st.session_state.disaster_type = (
                disaster_type
            )

            st.session_state.prediction_done = (
                True
            )


            # ------------------------------------------------
            # EMERGENCY LOCATIONS
            # ------------------------------------------------

            with st.spinner(
                "🧭 Finding suitable emergency direction..."
            ):

                emergency_locations = (
                    create_emergency_locations(

                        location_data["lat"],
                        location_data["lon"]
                    )
                )

            st.session_state.emergency_locations = (
                emergency_locations
            )


            # ------------------------------------------------
            # ROAD ROUTE
            # ------------------------------------------------

            camp = (
                emergency_locations["camp"]
            )

            with st.spinner(
                "🚗 Generating evacuation route..."
            ):

                route_data = get_route(

                    location_data["lat"],
                    location_data["lon"],

                    camp["lat"],
                    camp["lon"]
                )

            st.session_state.route_data = (
                route_data
            )


# ============================================================
# DISPLAY RESULTS
# ============================================================

if st.session_state.prediction_done:

    prediction = (
        st.session_state.prediction
    )

    confidence = (
        st.session_state.confidence
    )

    disaster_type = (
        st.session_state.disaster_type
    )

    location_data = (
        st.session_state.location_data
    )

    route_data = (
        st.session_state.route_data
    )

    emergency_locations = (
        st.session_state.emergency_locations
    )


    # ========================================================
    # RISK ASSESSMENT
    # ========================================================

    st.markdown(
        '<div class="section-heading">'
        '📊 Disaster Risk Assessment'
        '</div>',
        unsafe_allow_html=True
    )

    risk1, risk2, risk3 = st.columns(3)


    with risk1:

        if prediction == "High":

            st.error(
                "🔴 HIGH RISK"
            )

        elif prediction == "Medium":

            st.warning(
                "🟡 MEDIUM RISK"
            )

        else:

            st.success(
                "🟢 LOW RISK"
            )


    with risk2:

        st.metric(
            label="Prediction Confidence",
            value=f"{confidence:.1f}%"
        )


    with risk3:

        st.metric(
            label="Disaster Type",
            value=disaster_type
        )


    # ========================================================
    # LOCATION INFORMATION
    # ========================================================

    st.divider()

    st.markdown(
        '<div class="section-heading">'
        '📍 Disaster Location'
        '</div>',
        unsafe_allow_html=True
    )

    st.success(
        f"Location found: "
        f"**{location_data['display_name']}**"
    )

    loc1, loc2 = st.columns(2)

    with loc1:

        st.metric(
            "Latitude",
            f"{location_data['lat']:.6f}"
        )

    with loc2:

        st.metric(
            "Longitude",
            f"{location_data['lon']:.6f}"
        )


    # ========================================================
    # EMERGENCY RESPONSE
    # ========================================================

    st.divider()

    st.markdown(
        '<div class="section-heading">'
        '🚑 Emergency Response Recommendations'
        '</div>',
        unsafe_allow_html=True
    )


    if prediction == "High":

        st.error(
            "Immediate emergency response "
            "is recommended."
        )

    elif prediction == "Medium":

        st.warning(
            "Preparedness and continuous monitoring "
            "are recommended."
        )

    else:

        st.success(
            "Current conditions indicate "
            "relatively low risk."
        )


    recommendations = get_recommendations(
        disaster_type,
        prediction
    )


    rec1, rec2 = st.columns(2)


    for i, recommendation in enumerate(
        recommendations
    ):

        if i % 2 == 0:

            with rec1:

                st.markdown(
                    '<div class="recommendation-box">'
                    f'✅ {recommendation}'
                    '</div>',
                    unsafe_allow_html=True
                )

        else:

            with rec2:

                st.markdown(
                    '<div class="recommendation-box">'
                    f'✅ {recommendation}'
                    '</div>',
                    unsafe_allow_html=True
                )


    # ========================================================
    # RESOURCE ALLOCATION
    # ========================================================

    st.divider()

    st.markdown(
        '<div class="section-heading">'
        '📦 Recommended Resource Allocation'
        '</div>',
        unsafe_allow_html=True
    )


    (
        rescue_teams,
        ambulances,
        food_kits,
        water_units
    ) = get_resources(
        disaster_type,
        prediction
    )


    r1, r2, r3, r4 = st.columns(4)


    with r1:

        st.metric(
            "🚑 Rescue Teams",
            rescue_teams
        )


    with r2:

        st.metric(
            "🚗 Ambulances",
            ambulances
        )


    with r3:

        st.metric(
            "🍱 Food Kits",
            food_kits
        )


    with r4:

        st.metric(
            "💧 Water Units",
            water_units
        )


    # ========================================================
    # RELIEF CAMP
    # ========================================================

    st.divider()

    st.markdown(
        '<div class="section-heading">'
        '🏕️ Relief Camp Recommendation'
        '</div>',
        unsafe_allow_html=True
    )


    st.markdown(
        '<div class="camp-box">'
        '<h3>🏕️ Recommended Relief Camp: Camp A</h3>'
        '<p>Designated emergency location '
        'for evacuation support.</p>'
        '</div>',
        unsafe_allow_html=True
    )


    c1, c2, c3 = st.columns(3)


    with c1:

        st.metric(
            "Capacity",
            "1000 people"
        )


    with c2:

        st.metric(
            "Prototype Safety Score",
            "90%"
        )


    with c3:

        if route_data:

            st.metric(
                "Road Distance",
                f"{route_data['distance_km']:.2f} km"
            )

        else:

            st.metric(
                "Road Distance",
                "Unavailable"
            )


    # ========================================================
    # EVACUATION SUPPORT
    # ========================================================

    st.divider()

    st.markdown(
        '<div class="section-heading">'
        '🚨 Evacuation Support'
        '</div>',
        unsafe_allow_html=True
    )


    st.markdown(
        '<div class="evacuation-box">'
        '🔴 Disaster Area '
        '→ 🟠 Rescue Point '
        '→ 🏕️ Relief Camp '
        '→ 🏥 Emergency Hospital'
        '</div>',
        unsafe_allow_html=True
    )


    st.write(
        "Suggested evacuation direction: move affected "
        "people toward the designated emergency facilities."
    )


    # ========================================================
    # EMERGENCY LOCATIONS
    # ========================================================

    rescue = (
        emergency_locations["rescue"]
    )

    camp = (
        emergency_locations["camp"]
    )

    hospital = (
        emergency_locations["hospital"]
    )

    inland_bearing = (
        emergency_locations["inland_bearing"]
    )


    st.info(
        "🧭 Emergency facilities are positioned "
        "approximately away from the entered "
        "disaster location using geographic "
        "direction detection. These are prototype "
        "locations, not actual emergency facilities."
    )


    inland1, inland2, inland3 = st.columns(3)


    with inland1:

        st.metric(
            "🟠 Rescue Point",
            "~30 km away"
        )


    with inland2:

        st.metric(
            "🏕️ Relief Camp",
            "~35 km away"
        )


    with inland3:

        st.metric(
            "🏥 Emergency Hospital",
            "~38 km away"
        )


    # ========================================================
    # ROUTE INFORMATION
    # ========================================================

    if route_data:

        st.divider()

        st.markdown(
            '<div class="section-heading">'
            '🚗 Evacuation Route Information'
            '</div>',
            unsafe_allow_html=True
        )


        route1, route2 = st.columns(2)


        with route1:

            st.metric(
                "🚗 Road Distance",
                f"{route_data['distance_km']:.2f} km"
            )


        with route2:

            st.metric(
                "⏱️ Estimated Travel Time",
                f"{route_data['duration_minutes']:.0f} min"
            )


        st.success(
            "Actual road-network routing generated "
            "between the disaster location and the "
            "recommended relief camp."
        )


    else:

        st.warning(
            "⚠️ Unable to generate a road route "
            "at this time. Emergency locations "
            "are still displayed on the map."
        )


    # ========================================================
    # INTERACTIVE MAP
    # ========================================================

    st.divider()

    st.markdown(
        '<div class="section-heading">'
        '🗺️ Interactive Emergency Map'
        '</div>',
        unsafe_allow_html=True
    )


    disaster_lat = (
        location_data["lat"]
    )

    disaster_lon = (
        location_data["lon"]
    )


    rescue_lat = rescue["lat"]
    rescue_lon = rescue["lon"]

    camp_lat = camp["lat"]
    camp_lon = camp["lon"]

    hospital_lat = hospital["lat"]
    hospital_lon = hospital["lon"]


    # ========================================================
    # MAP CENTER
    # ========================================================

    map_center_lat = (
        disaster_lat
        +
        rescue_lat
        +
        camp_lat
        +
        hospital_lat
    ) / 4


    map_center_lon = (
        disaster_lon
        +
        rescue_lon
        +
        camp_lon
        +
        hospital_lon
    ) / 4


    emergency_map = folium.Map(

        location=[
            map_center_lat,
            map_center_lon
        ],

        zoom_start=10,

        control_scale=True
    )


    # ========================================================
    # DISASTER MARKER
    # ========================================================

    folium.Marker(

        [
            disaster_lat,
            disaster_lon
        ],

        tooltip="🔴 Disaster Location",

        popup=(
            "<b>🔴 Disaster Location</b><br>"
            f"{location_data['display_name']}<br>"
            f"Disaster: {disaster_type}"
        ),

        icon=folium.Icon(
            color="red",
            icon="warning-sign"
        )

    ).add_to(
        emergency_map
    )


    # ========================================================
    # RESCUE POINT
    # ========================================================

    folium.Marker(

        [
            rescue_lat,
            rescue_lon
        ],

        tooltip="🟠 Emergency Rescue Point",

        popup=(
            "<b>🟠 Emergency Rescue Point</b><br>"
            "Prototype emergency location."
        ),

        icon=folium.Icon(
            color="orange",
            icon="flag"
        )

    ).add_to(
        emergency_map
    )


    # ========================================================
    # RELIEF CAMP
    # ========================================================

    folium.Marker(

        [
            camp_lat,
            camp_lon
        ],

        tooltip="🟢 Relief Camp",

        popup=(
            "<b>🟢 Recommended Relief Camp</b><br>"
            "Camp A<br>"
            "Capacity: 1000 people"
        ),

        icon=folium.Icon(
            color="green",
            icon="home"
        )

    ).add_to(
        emergency_map
    )


    # ========================================================
    # HOSPITAL
    # ========================================================

    folium.Marker(

        [
            hospital_lat,
            hospital_lon
        ],

        tooltip="🔵 Emergency Hospital",

        popup=(
            "<b>🔵 Emergency Hospital</b><br>"
            "Prototype emergency medical facility."
        ),

        icon=folium.Icon(
            color="blue",
            icon="plus"
        )

    ).add_to(
        emergency_map
    )


    # ========================================================
    # APPROXIMATE EMERGENCY DIRECTION
    # ========================================================

    folium.PolyLine(

        [
            [
                disaster_lat,
                disaster_lon
            ],

            [
                rescue_lat,
                rescue_lon
            ]
        ],

        tooltip="Approximate Emergency Direction",

        weight=3,

        dash_array="8, 8"

    ).add_to(
        emergency_map
    )


    # ========================================================
    # ACTUAL ROAD ROUTE
    # ========================================================

    if route_data:

        folium.PolyLine(

            route_data["coordinates"],

            tooltip=(
                "🚗 Recommended Road "
                "Evacuation Route"
            ),

            weight=6

        ).add_to(
            emergency_map
        )


    # ========================================================
    # MAP LEGEND
    # ========================================================

    legend_html = """

    <div style="
        position: fixed;
        bottom: 30px;
        left: 30px;
        width: 250px;
        z-index: 9999;

        background: rgba(20, 20, 20, 0.95);

        border: 1px solid
        rgba(255, 255, 255, 0.4);

        padding: 14px;

        font-size: 14px;

        color: #ffffff;

        border-radius: 10px;

        box-shadow:
        0 4px 14px
        rgba(0,0,0,0.5);
    ">

    <b style="
        font-size: 15px;
        color: #ffffff;
    ">

    Emergency Map Legend

    </b>

    <br><br>

    <span style="color:#ff4d4d;">
    🔴 Disaster Location
    </span>

    <br><br>

    <span style="color:#ff9f43;">
    🟠 Emergency Rescue Point
    </span>

    <br><br>

    <span style="color:#2ecc71;">
    🟢 Relief Camp
    </span>

    <br><br>

    <span style="color:#3498db;">
    🔵 Emergency Hospital
    </span>

    <br><br>

    <span style="color:#ffffff;">
    🚗 Recommended Road Route
    </span>

    </div>

    """


    emergency_map.get_root().html.add_child(

        folium.Element(
            legend_html
        )
    )


    # ========================================================
    # DISPLAY MAP
    # ========================================================

    st_folium(

        emergency_map,

        height=600,

        use_container_width=True
    )


    st.caption(
        "Map data © OpenStreetMap contributors. "
        "Road route generated using available "
        "road-network data."
    )


    # ========================================================
    # EMERGENCY FACILITIES
    # ========================================================

    st.divider()

    st.markdown(
        '<div class="section-heading">'
        '🏥 Emergency Facilities'
        '</div>',
        unsafe_allow_html=True
    )


    e1, e2, e3 = st.columns(3)


    with e1:

        st.markdown(

            '<div class="info-card">'
            '<h3>🏕️ Relief Camp</h3>'
            '<p>Camp A</p>'
            '<p>Capacity: 1000 people</p>'
            '<p>Prototype location</p>'
            '</div>',

            unsafe_allow_html=True
        )


    with e2:

        st.markdown(

            '<div class="info-card">'
            '<h3>🏥 Emergency Hospital</h3>'
            '<p>Medical support location</p>'
            '<p>Prototype facility</p>'
            '</div>',

            unsafe_allow_html=True
        )


    with e3:

        st.markdown(

            '<div class="info-card">'
            '<h3>🚑 Rescue Point</h3>'
            '<p>Emergency response support</p>'
            '<p>Prototype location</p>'
            '</div>',

            unsafe_allow_html=True
        )


    # ========================================================
    # COMPLETION
    # ========================================================

    st.divider()

    st.success(
        "✅ Disaster assessment completed successfully."
    )


    st.markdown(

        '<div class="dashboard-footer">'
        'AI-Assisted Disaster Response and Resource Recommendation System'
        '</div>',

        unsafe_allow_html=True
    )


# ============================================================
# INITIAL SCREEN
# ============================================================

else:

    st.info(
        "👈 Enter disaster information and a "
        "disaster location in the sidebar, "
        "then click **PREDICT DISASTER RISK** "
        "to begin."
    )


    st.markdown(
        '<div class="section-heading">🧠 System Workflow</div>',
        unsafe_allow_html=True
    )

    workflow = [
        ("01", "Input", "Disaster-specific environmental conditions are entered."),
        ("02", "Location", "The affected location is geocoded automatically."),
        ("03", "AI Analysis", "The appropriate Random Forest model predicts disaster risk."),
        ("04", "Response Planning", "Emergency actions and resource requirements are recommended."),
        ("05", "Emergency Locations", "Prototype rescue, relief and medical locations are generated."),
        ("06", "Route Planning", "A road-based evacuation route is generated using OSRM."),
        ("07", "Decision Support", "The final plan is presented with an interactive emergency map."),
    ]

    for number, title, description in workflow:
        st.markdown(
            f"""
            <div class="info-card" style="padding: 15px 18px; margin-bottom: 10px;">
                <div style="display:flex; gap:14px; align-items:flex-start;">
                    <div style="font-size:13px; font-weight:800; color:#ef4444; min-width:28px;">{number}</div>
                    <div>
                        <div style="font-size:16px; font-weight:750; color:#ffffff;">{title}</div>
                        <div style="font-size:14px; color:#94a3b8; margin-top:4px;">{description}</div>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )