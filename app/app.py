from pathlib import Path
import hashlib

import geopandas as gpd
import pandas as pd
import streamlit as st
import folium

from folium.plugins import Fullscreen
from streamlit_folium import st_folium


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="London Road Risk Map",
    page_icon="🗺️",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

RISK_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "london_road_risk_scores.gpkg"
)


# ============================================================
# CSS
# ============================================================

st.markdown(
    """
    <style>

    .block-container {
        padding-top: 1.2rem;
        padding-bottom: 1rem;
        max-width: 100%;
    }

    [data-testid="stMetric"] {
        background-color: rgba(120, 120, 120, 0.08);
        border: 1px solid rgba(120, 120, 120, 0.18);
        padding: 15px;
        border-radius: 10px;
    }

    iframe {
        width: 100% !important;
        display: block !important;
    }

    [data-testid="stIFrame"] {
        width: 100% !important;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# LOAD DATA
# ============================================================

@st.cache_data(show_spinner="Loading road risk data...")
def load_data():

    roads = gpd.read_file(
        RISK_FILE,
        layer="road_risk_scores"
    )

    # Leaflet / Folium uses WGS84.
    roads = roads.to_crs(epsg=4326)

    return roads


roads = load_data()


# ============================================================
# HEADER
# ============================================================

st.title("London Road Collision Risk Map")

st.caption(
    "Spatial risk modeling of London road segments using "
    "traffic collision data, road-network characteristics "
    "and machine learning."
)


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header("Map Controls")


risk_options = st.sidebar.multiselect(
    "Risk Category",
    options=[
        "Critical",
        "High",
        "Moderate",
        "Low"
    ],
    default=[
        "Critical",
        "High"
    ]
)


min_risk = st.sidebar.slider(
    "Minimum Risk Percentile",
    min_value=0,
    max_value=100,
    value=80,
    step=1
)


road_types = sorted(
    roads["highway_clean"]
    .dropna()
    .astype(str)
    .unique()
)


selected_road_types = st.sidebar.multiselect(
    "Road Type",
    options=road_types,
    default=[]
)


only_collision_roads = st.sidebar.checkbox(
    "Only roads with observed collisions",
    value=False
)


only_severe_roads = st.sidebar.checkbox(
    "Only roads with severe collisions",
    value=False
)


# ============================================================
# FILTER DATA
# ============================================================

filtered = roads.copy()


if risk_options:

    filtered = filtered[
        filtered["risk_category"]
        .astype(str)
        .isin(risk_options)
    ]

else:

    filtered = filtered.iloc[0:0].copy()


filtered = filtered[
    filtered["risk_percentile"] >= min_risk
]


if selected_road_types:

    filtered = filtered[
        filtered["highway_clean"]
        .astype(str)
        .isin(selected_road_types)
    ]


if only_collision_roads:

    filtered = filtered[
        filtered["collision_count"] > 0
    ]


if only_severe_roads:

    filtered = filtered[
        filtered["severe_collision_count"] > 0
    ]


# ============================================================
# KPI METRICS
# ============================================================

total_segments = len(filtered)

total_collisions = int(
    filtered["collision_count"].sum()
)

severe_collisions = int(
    filtered["severe_collision_count"].sum()
)

fatal_collisions = int(
    filtered["fatal_collision_count"].sum()
)


col1, col2, col3, col4 = st.columns(4)


col1.metric(
    "Road Segments",
    f"{total_segments:,}"
)

col2.metric(
    "Observed Collisions",
    f"{total_collisions:,}"
)

col3.metric(
    "Severe Collisions",
    f"{severe_collisions:,}"
)

col4.metric(
    "Fatal Collisions",
    f"{fatal_collisions:,}"
)


# ============================================================
# MAP SECTION
# ============================================================

st.subheader("Interactive Road Risk Map")


# ============================================================
# DISPLAY LIMIT
# ============================================================

MAX_DISPLAY_SEGMENTS = 5000


if len(filtered) > MAX_DISPLAY_SEGMENTS:

    st.info(
        f"The current filters contain {len(filtered):,} road segments. "
        f"Map rendering is limited to the top "
        f"{MAX_DISPLAY_SEGMENTS:,} highest-risk segments for "
        f"interactive performance. Summary metrics still use "
        f"all filtered segments."
    )

    map_data = (
        filtered
        .nlargest(
            MAX_DISPLAY_SEGMENTS,
            "risk_percentile"
        )
        .copy()
    )

else:

    map_data = filtered.copy()


# ============================================================
# CREATE MAP
# ============================================================

m = folium.Map(
    location=[
        51.5074,
        -0.1278
    ],
    zoom_start=10,
    tiles=None,
    control_scale=True,
    prefer_canvas=True,
    width="100%",
    height="100%"
)


# ============================================================
# BASE MAP 1 - OPENSTREETMAP
# ============================================================

folium.TileLayer(
    tiles="OpenStreetMap",
    name="OpenStreetMap",
    overlay=False,
    control=True,
    show=True
).add_to(m)


# ============================================================
# BASE MAP 2 - OSM HUMANITARIAN
# ============================================================

folium.TileLayer(
    tiles="https://{s}.tile.openstreetmap.fr/hot/{z}/{x}/{y}.png",
    attr="© OpenStreetMap contributors, Tiles style by HOT",
    name="OSM Humanitarian",
    overlay=False,
    control=True,
    show=False,
    max_zoom=19
).add_to(m)


# ============================================================
# FULLSCREEN
# ============================================================

Fullscreen(
    position="topright",
    title="Fullscreen",
    title_cancel="Exit fullscreen",
    force_separate_button=True
).add_to(m)


# ============================================================
# RISK COLORS
# ============================================================

risk_colors = {
    "Critical": "#d73027",
    "High": "#fc8d59",
    "Moderate": "#fee08b",
    "Low": "#91cf60"
}


# ============================================================
# PREPARE GEOJSON
# ============================================================

if not map_data.empty:

    display_columns = [
        "name",
        "highway_clean",
        "risk_category",
        "risk_percentile",
        "risk_score_raw",
        "collision_count",
        "severe_collision_count",
        "fatal_collision_count",
        "geometry"
    ]


    display_columns = [
        column
        for column in display_columns
        if column in map_data.columns
    ]


    map_display = map_data[
        display_columns
    ].copy()


    # ========================================================
    # CLEAN TEXT
    # ========================================================

    if "name" in map_display.columns:

        map_display["name"] = (
            map_display["name"]
            .fillna("Unnamed Road")
            .astype(str)
        )


    if "highway_clean" in map_display.columns:

        map_display["highway_clean"] = (
            map_display["highway_clean"]
            .fillna("Unknown")
            .astype(str)
        )


    if "risk_category" in map_display.columns:

        map_display["risk_category"] = (
            map_display["risk_category"]
            .fillna("Unknown")
            .astype(str)
        )


    # ========================================================
    # CLEAN NUMERIC VALUES
    # ========================================================

    if "risk_percentile" in map_display.columns:

        map_display["risk_percentile"] = (
            pd.to_numeric(
                map_display["risk_percentile"],
                errors="coerce"
            )
            .fillna(0)
            .round(1)
        )


    if "risk_score_raw" in map_display.columns:

        map_display["risk_score_raw"] = (
            pd.to_numeric(
                map_display["risk_score_raw"],
                errors="coerce"
            )
            .fillna(0)
            .round(3)
        )


    for column in [
        "collision_count",
        "severe_collision_count",
        "fatal_collision_count"
    ]:

        if column in map_display.columns:

            map_display[column] = (
                pd.to_numeric(
                    map_display[column],
                    errors="coerce"
                )
                .fillna(0)
                .astype(int)
            )


    # ========================================================
    # ROAD STYLE
    # ========================================================

    def road_style(feature):

        category = (
            feature["properties"]
            .get(
                "risk_category",
                "Unknown"
            )
        )

        color = risk_colors.get(
            category,
            "#808080"
        )


        if category == "Critical":

            weight = 4.0

        elif category == "High":

            weight = 3.0

        elif category == "Moderate":

            weight = 2.5

        else:

            weight = 2.0


        return {
            "color": color,
            "weight": weight,
            "opacity": 0.90
        }


    # ========================================================
    # TOOLTIP
    # ========================================================

    tooltip_fields = [
        column
        for column in [
            "name",
            "highway_clean",
            "risk_category",
            "risk_percentile"
        ]
        if column in map_display.columns
    ]


    tooltip_alias_map = {

        "name":
            "Road:",

        "highway_clean":
            "Road Type:",

        "risk_category":
            "Risk Category:",

        "risk_percentile":
            "Risk Percentile:"
    }


    tooltip_aliases = [
        tooltip_alias_map[column]
        for column in tooltip_fields
    ]


    # ========================================================
    # POPUP
    # ========================================================

    popup_fields = [
        column
        for column in [
            "name",
            "highway_clean",
            "risk_category",
            "risk_percentile",
            "risk_score_raw",
            "collision_count",
            "severe_collision_count",
            "fatal_collision_count"
        ]
        if column in map_display.columns
    ]


    popup_alias_map = {

        "name":
            "Road:",

        "highway_clean":
            "Road Type:",

        "risk_category":
            "Risk Category:",

        "risk_percentile":
            "Risk Percentile:",

        "risk_score_raw":
            "Model Risk Score:",

        "collision_count":
            "Observed Collisions:",

        "severe_collision_count":
            "Severe Collisions:",

        "fatal_collision_count":
            "Fatal Collisions:"
    }


    popup_aliases = [
        popup_alias_map[column]
        for column in popup_fields
    ]


    # ========================================================
    # ROAD LAYER
    # ========================================================

    road_layer = folium.GeoJson(

        data=map_display.to_json(),

        name="Road Risk",

        style_function=road_style,

        highlight_function=lambda feature: {
            "weight": 6,
            "opacity": 1
        },

        tooltip=folium.GeoJsonTooltip(

            fields=tooltip_fields,

            aliases=tooltip_aliases,

            sticky=False,

            labels=True
        ),

        popup=folium.GeoJsonPopup(

            fields=popup_fields,

            aliases=popup_aliases,

            localize=True,

            labels=True
        )
    )


    road_layer.add_to(m)


# ============================================================
# LEGEND
# ============================================================

legend_html = """
<div style="
position: fixed;
bottom: 35px;
left: 35px;
width: 165px;
background-color: rgba(255,255,255,0.94);
border: 1px solid #999;
z-index: 9999;
font-size: 13px;
padding: 12px;
border-radius: 7px;
color: #222;
">

<b>Road Risk Level</b>

<br><br>

<span style="
color:#d73027;
font-size:18px;
">━</span>
Critical

<br>

<span style="
color:#fc8d59;
font-size:18px;
">━</span>
High

<br>

<span style="
color:#fee08b;
font-size:18px;
">━</span>
Moderate

<br>

<span style="
color:#91cf60;
font-size:18px;
">━</span>
Low

</div>
"""


m.get_root().html.add_child(
    folium.Element(
        legend_html
    )
)


# ============================================================
# LAYER CONTROL
# ============================================================

folium.LayerControl(
    position="topright",
    collapsed=True
).add_to(m)


# ============================================================
# LEAFLET RESIZE FIX
# ============================================================

map_name = m.get_name()


resize_script = f"""
<script>

function resizeLeafletMap() {{

    if (
        typeof {map_name} !== "undefined"
    ) {{

        {map_name}.invalidateSize(true);

    }}

}}


window.addEventListener(
    "load",
    function() {{

        setTimeout(
            resizeLeafletMap,
            200
        );

        setTimeout(
            resizeLeafletMap,
            500
        );

        setTimeout(
            resizeLeafletMap,
            1000
        );

        setTimeout(
            resizeLeafletMap,
            2000
        );

    }}
);


window.addEventListener(
    "resize",
    resizeLeafletMap
);

</script>
"""


m.get_root().html.add_child(
    folium.Element(
        resize_script
    )
)


# ============================================================
# DYNAMIC MAP KEY
# ============================================================

filter_signature = str(
    (
        tuple(sorted(risk_options)),
        min_risk,
        tuple(sorted(selected_road_types)),
        only_collision_roads,
        only_severe_roads
    )
)


map_key = hashlib.md5(
    filter_signature.encode("utf-8")
).hexdigest()


# ============================================================
# DISPLAY MAP
# ============================================================

st_folium(
    m,
    width=None,
    height=620,
    key=f"road_risk_map_{map_key}",
    returned_objects=[]
)


# ============================================================
# EMPTY RESULT MESSAGE
# ============================================================

if map_data.empty:

    st.warning(
        "No road segments match the selected filters. "
        "Change the map controls to display road segments."
    )


# ============================================================
# HIGHEST-RISK TABLE
# ============================================================

st.subheader(
    "Highest-Risk Road Segments"
)


table_columns = [
    "name",
    "highway_clean",
    "risk_category",
    "risk_percentile",
    "risk_score_raw",
    "collision_count",
    "severe_collision_count",
    "fatal_collision_count"
]


table_columns = [
    column
    for column in table_columns
    if column in filtered.columns
]


if not filtered.empty:

    top_roads = (
        filtered[
            table_columns
        ]
        .sort_values(
            "risk_percentile",
            ascending=False
        )
        .head(25)
        .copy()
    )


    top_roads = top_roads.rename(
        columns={

            "name":
                "Road",

            "highway_clean":
                "Road Type",

            "risk_category":
                "Risk Category",

            "risk_percentile":
                "Risk Percentile",

            "risk_score_raw":
                "Model Risk Score",

            "collision_count":
                "Collisions",

            "severe_collision_count":
                "Severe",

            "fatal_collision_count":
                "Fatal"
        }
    )


    st.dataframe(
        top_roads,
        width="stretch",
        hide_index=True
    )


else:

    st.info(
        "No road segments are available "
        "for the current filters."
    )


# ============================================================
# MODEL INFORMATION
# ============================================================

with st.expander(
    "About the Risk Model"
):

    st.markdown(
        """
This application displays relative collision risk across
London road segments.

The underlying analytical workflow combines:

- road-network characteristics
- historical collision records
- spatial density analysis
- spatial autocorrelation analysis
- network feature engineering
- machine-learning risk modeling
- spatial model validation

### Risk Percentile

Risk Percentile represents the relative position of a road
segment compared with the rest of the London road network.

The score should therefore be interpreted as a
**relative road-risk indicator**, rather than the literal
probability that a future collision will occur.

### Risk Categories

- **Critical:** Top 5%
- **High:** 80th–95th percentile
- **Moderate:** 50th–80th percentile
- **Low:** Bottom 50%
        """
    )


# ============================================================
# FOOTER
# ============================================================

st.caption(
    "Traffic Accident Spatial Risk Modeling | "
    "London Road Network Analysis"
)