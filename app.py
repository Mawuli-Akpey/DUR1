import streamlit as st
st.set_page_config(page_title="Traffic Analytics Dashboard for DUR", layout="wide")
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt

# --- Login Page ---
def login():
    st.title("Login")
    rerun = False  # Flag to trigger rerun after form submission
    with st.form("login_form"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Login")
        if submitted:
            valid_username = st.secrets["login"]["username"]
            valid_password = st.secrets["login"]["password"]
            if username == valid_username and password == valid_password:
                st.session_state["logged_in"] = True
                st.success("Login successful! Redirecting...")
                rerun = True
            else:
                st.error("Invalid username or password.")
    if rerun:
        st.rerun()

if "logged_in" not in st.session_state:
    st.session_state["logged_in"] = False

if not st.session_state["logged_in"]:
    login()
    st.stop()

# =============================
# Load Data
# =============================
@st.cache_data
def load_data():
    df = pd.read_excel("3hr DUR data.xlsx")
    return df

df = load_data()

# =============================
# Sidebar Filters (Dynamic)
# =============================
st.sidebar.header("Filters")
# Vehicle & Person selector
filter_options = [col for col in ['light', 'medium', 'heavy', 'person'] if (col in df.columns or col == 'person')]
filter_labels = {'light': 'Light', 'medium': 'Medium', 'heavy': 'Heavy', 'person': 'Person'}
selected_filters = st.sidebar.multiselect(
    "Filter for Vehicle & Person",
    options=filter_options,
    format_func=lambda x: filter_labels[x],
    default=[opt for opt in filter_options if opt != 'person']
)

# Map 'person' to 'person_count' for data selection
selected_data_cols = [col if col != 'person' else 'person_count' for col in selected_filters]

# Hour selector
hour_options = sorted(df['hour'].unique())
selected_hours = st.sidebar.multiselect("Select Hours", options=hour_options, default=hour_options)
# Minute range slider
min_minute, max_minute = int(df['minute'].min()), int(df['minute'].max())
minute_range = st.sidebar.slider("Minute Range", min_value=min_minute, max_value=max_minute, value=(min_minute, max_minute))

# Filter data
filtered_df = df[
    df['hour'].isin(selected_hours) &
    df['minute'].between(minute_range[0], minute_range[1])
]

# Only keep selected columns
filtered_df = filtered_df[['hour', 'minute'] + selected_data_cols]

# Data summary
st.sidebar.markdown("**Data Summary:**")
if not filtered_df.empty:
    st.sidebar.text(f"Records: {len(filtered_df)}\nTime Range: {filtered_df['hour'].min()}:{filtered_df['minute'].min():02d} - {filtered_df['hour'].max()}:{filtered_df['minute'].max():02d}")
else:
    st.sidebar.text("No data for selected filters.")

# =============================
# Main Area
# =============================
# Logo
st.image("DUR.png", width=120)

st.title("DUR Traffic Analytics Dashboard")
st.markdown("""
By: **ZoneSense (2025 ©)**
""")

# =============================
# Insert Feed Screenshot (centered)
# =============================
st.markdown("<div style='text-align: center;'>", unsafe_allow_html=True)
st.image("feed.png", caption="Sample Feed Frame Used for Inference", use_container_width=True)
st.markdown("</div>", unsafe_allow_html=True)

# =============================
# KPIs
# =============================
st.markdown("### Summary Statistics")
kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)
# Total vehicles (sum of vehicle columns only)
vehicle_cols = [col for col in selected_data_cols if col in ['light', 'medium', 'heavy']]
total_vehicles = filtered_df[vehicle_cols].sum().sum() if vehicle_cols else 0
kpi1.metric("Total Vehicles", f"{total_vehicles}")
# Total people (sum of person_count)
total_people = filtered_df['person_count'].sum() if 'person_count' in filtered_df.columns else 0
kpi2.metric("Total People", f"{total_people}")
# Average vehicles per minute
if len(filtered_df) > 0 and vehicle_cols:
    avg_vehicles_per_min = filtered_df[vehicle_cols].sum(axis=1).mean()
else:
    avg_vehicles_per_min = 0
kpi3.metric("Avg Vehicles/Min", f"{avg_vehicles_per_min:.2f}")
# Peak minute traffic
if len(filtered_df) > 0 and vehicle_cols:
    peak_minute_traffic = filtered_df[vehicle_cols].sum(axis=1).max()
else:
    peak_minute_traffic = 0
kpi4.metric("Peak Minute Traffic", f"{peak_minute_traffic}")
# Predominant vehicle type
if vehicle_cols:
    predominant_type = pd.Series({col: filtered_df[col].sum() for col in vehicle_cols}).idxmax()
    predominant_type_label = filter_labels.get(predominant_type, predominant_type)
else:
    predominant_type_label = "--"
kpi5.metric("Predominant Vehicle Type", predominant_type_label)

# =============================
# Charts Section
# =============================
st.markdown("### Charts")

# Vehicle counts by time
if vehicle_cols:
    stacked_df = filtered_df.copy()
    stacked_df['time'] = stacked_df['hour'].astype(str) + ":" + stacked_df['minute'].astype(str).str.zfill(2)
    stacked_df = stacked_df.sort_values(['hour', 'minute'])
    st.write("#### Vehicle Counts by Time")
    st.markdown("Shows the count of each selected vehicle type for every minute in the selected time range. X-axis: time (hour:minute), Y-axis: vehicle count.")
    st.bar_chart(stacked_df.set_index('time')[vehicle_cols])
else:
    st.write("No vehicle types selected for stacked bar chart.")

# Line Chart: Traffic volume over time
if selected_data_cols:
    line_df = filtered_df.copy()
    line_df['time'] = line_df['hour'].astype(str) + ":" + line_df['minute'].astype(str).str.zfill(2)
    line_df = line_df.sort_values(['hour', 'minute'])
    st.write("#### Traffic Volume Over Time")
    st.markdown("Shows the trend of selected vehicle/person counts over time. X-axis: time (hour:minute), Y-axis: count.")
    st.line_chart(line_df.set_index('time')[selected_data_cols])
else:
    st.write("No data columns selected for line chart.")

# Donut Chart: Vehicle/person type distribution (Plotly)
if selected_data_cols:
    donut_data = filtered_df[selected_data_cols].sum()
    st.write("#### Vehicle & Person Distribution (Donut Chart)")
    st.markdown("Shows the proportion of each selected vehicle type and person count in the filtered data.")
    donut_fig = go.Figure(data=[go.Pie(
        labels=[filter_labels.get(lbl, lbl) for lbl in donut_data.index],
        values=donut_data,
        hole=0.5,
        marker=dict(colors=px.colors.qualitative.Plotly),
        textinfo='label+percent',
        insidetextorientation='radial',
    )])
    donut_fig.update_layout(margin=dict(t=0, b=0, l=0, r=0))
    st.plotly_chart(donut_fig, use_container_width=True)
else:
    st.write("No data columns selected for donut chart.")

# Heatmap: Traffic intensity (hour vs. minute) (Plotly)
if vehicle_cols:
    heatmap_data = filtered_df.groupby(['hour', 'minute'])[vehicle_cols].sum().sum(axis=1).unstack(fill_value=0)
    st.write("#### Heatmap: Traffic Intensity (Hour vs. Minute)")
    st.markdown("X-axis: minute, Y-axis: hour (always 1, 2, 3), Z (color): total vehicle count for each (hour, minute) cell. Darker color = more vehicles.")
    # Ensure y-axis is always [1, 2, 3]
    all_hours = [1, 2, 3]
    # Reindex to always have 1,2,3 as y-axis
    heatmap_data = heatmap_data.reindex(all_hours, fill_value=0)
    heatmap_fig = go.Figure(data=go.Heatmap(
        z=heatmap_data.values,
        x=heatmap_data.columns,
        y=heatmap_data.index,
        colorscale='YlGnBu',
        colorbar=dict(title='Vehicle Count')
    ))
    heatmap_fig.update_layout(xaxis_title='Minute', yaxis_title='Hour')
    st.plotly_chart(heatmap_fig, use_container_width=True)
else:
    st.write("No vehicle types selected for heatmap.")

# Bar Chart: Total vehicles per hour (annotated)
if vehicle_cols:
    bar_data = filtered_df.groupby('hour')[vehicle_cols].sum().sum(axis=1)
    st.write("#### Total Vehicles per Hour")
    st.markdown("**Bar Chart:** Shows the total number of vehicles for each hour in the selected range. X-axis: hour, Y-axis: total vehicle count. Each bar is annotated with its value.")
    bar_fig = go.Figure(data=[go.Bar(
        x=bar_data.index,
        y=bar_data.values,
        text=bar_data.values,
        textposition='auto',
        marker_color=px.colors.qualitative.Plotly[:len(bar_data)]
    )])
    bar_fig.update_layout(xaxis_title='Hour', yaxis_title='Total Vehicles', showlegend=False)
    st.plotly_chart(bar_fig, use_container_width=True)
else:
    st.write("No vehicle types selected for bar chart.")

# Correlation Heatmap: person_count vs. vehicle types (Plotly)
if 'person_count' in filtered_df.columns and vehicle_cols:
    corr_data = filtered_df[['person_count'] + vehicle_cols].corr()
    st.write("#### Correlation Heatmap")
    st.markdown("Shows the correlation (from -1 to 1) between person count and each vehicle type. X and Y axes: variable names, Z (color): correlation coefficient.")
    corr_fig = go.Figure(data=go.Heatmap(
        z=corr_data.values,
        x=corr_data.columns,
        y=corr_data.index,
        colorscale='RdBu',
        zmin=-1, zmax=1,
        colorbar=dict(title='Correlation')
    ))
    corr_fig.update_layout(xaxis_title='', yaxis_title='')
    st.plotly_chart(corr_fig, use_container_width=True)
else:
    st.write("No persons selected for correlation heatmap.")

# =============================
# Export Section
# =============================
st.markdown("### Export Data")
st.markdown("**To export the filtered data, click the 'Download Filtered Data as CSV' button below. The file will contain only the data currently shown in the dashboard.**")
# Download filtered data as CSV
csv = filtered_df.to_csv(index=False).encode('utf-8')
st.download_button("Download Filtered Data as CSV", csv, "filtered_data.csv", "text/csv")

# =============================
# Footer
# =============================
st.markdown("---")
st.markdown("About ZoneSense | Contact: info@zonesense.com")
