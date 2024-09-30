import os
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from collections import defaultdict
import requests
from flask import Flask, request, render_template, redirect, url_for
import dash
from dash import dcc, html, dash_table
from dash.dependencies import Input, Output, State
import plotly.graph_objs as go
import geopandas as gpd
import folium
from folium.plugins import MarkerCluster
from branca.colormap import LinearColormap
from functools import lru_cache
# Import custom modules
from data_process import process_data
from get_data import fetch_data_from_api

# Configuration
API_URL = "https://database-mango.onrender.com"#"https://mongodb-api-hmeu.onrender.com"
COLUMNS = ["source_pH", "source_TDS", "source_FRC", "source_pressure", "source_flow"]
Y_RANGES = {
    "source_pH": [7, 10],
    "source_TDS": [0, 500],
    "source_FRC": [0, 0.050],
    "source_pressure": [0, 2],
    "source_flow": [0, 15]
}
TIME_DURATIONS = {
    '1 Hour': timedelta(hours=1),
    '3 Hours': timedelta(hours=3),
    '6 Hours': timedelta(hours=6),
    '12 Hours': timedelta(hours=12),
    '24 Hours': timedelta(hours=24),
    '3 Days': timedelta(days=3),
    '1 Week': timedelta(weeks=1)
}
UNITS = {
    "pH": "",
    "TDS": "ppm",
    "FRC": "ppm",
    "pressure": "bar",
    "flow": "kL per 10 mins"
}

# Initialize Flask and Dash
server = Flask(__name__)
server.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', os.urandom(24))
app = dash.Dash(__name__, server=server, url_base_pathname='/dashboard/')

# Analysis functions
def calculate_pumping_time_and_flow(df):
    daily_pumping_times = defaultdict(timedelta)
    daily_flow_sums = defaultdict(float)
    start_time = None
    current_day = None
    
    for _, row in df.iterrows():
        timestamp = row['timestamp']
        source_flow = float(row['source_flow'])
        day = timestamp.date()
        
        if current_day is None:
            current_day = day
        if day != current_day:
            start_time = None
            current_day = day
        
        if source_flow > 0:
            daily_flow_sums[day] += source_flow
            if start_time is None:
                start_time = timestamp
        elif source_flow == 0 and start_time is not None:
            end_time = timestamp
            daily_pumping_times[day] += end_time - start_time
            start_time = None
    
    if start_time is not None:
        end_time = timestamp
        daily_pumping_times[day] += end_time - start_time
    
    return daily_pumping_times, daily_flow_sums

def format_timedelta(td):
    total_seconds = int(td.total_seconds())
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    seconds = total_seconds % 60
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

def calculate_lpcd(df, population=10000):
    daily_flow = df.groupby(df['timestamp'].dt.date)['source_flow'].sum() 
    return daily_flow / population

def analyze_water_quality(df):
    return {
        'avg_tds': df['source_TDS'].mean(),
        'avg_ph': df['source_pH'].mean(),
        'avg_pressure': df['source_pressure'].mean()
    }

# Initialize Flask
server = Flask(__name__)
server.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', os.urandom(24))

# Initialize Dash
app = dash.Dash(__name__, server=server, url_base_pathname='/dashboard/')

# Load GeoJSON data
@lru_cache(maxsize=None)
def load_geojson():
    geojson_path = "Dadapur.geojson"

    return gpd.read_file(geojson_path)

# Load and process Excel data
@lru_cache(maxsize=None)
def load_excel_data():
    df = pd.read_excel('BOTH_WQ.xlsx', sheet_name="Dadupur")
    return gpd.GeoDataFrame(
        df, geometry=gpd.points_from_xy(df.Longitude, df.Latitude), crs="EPSG:4326"
    )

gdf = load_geojson()
excel_gdf = load_excel_data()
def create_map():
    excel_gdf_4326 = excel_gdf.to_crs(epsg=4326)
    map_center = [excel_gdf_4326.geometry.y.mean(), excel_gdf_4326.geometry.x.mean()]
    m = folium.Map(location=map_center, zoom_start=14)
    
    colormap_tds = LinearColormap(
        colors=['green', 'yellow', 'red'],
        vmin=excel_gdf_4326['Total Dissolved Solids (TDS)'].min(),
        vmax=excel_gdf_4326['Total Dissolved Solids (TDS)'].max(),
        caption='Total Dissolved Solids (TDS)'
    )
    
    marker_cluster_tds = MarkerCluster(name="TDS Data").add_to(m)
    
    def create_popup_content(row):
        return f"""
        Village: {row['Village']}<br>
        pH: {row['pH']}<br>
        TDS: {row['Total Dissolved Solids (TDS)']} mg/L<br>
        FRC: {row['Free Residual Chlorine (FRC)']} mg/L<br>
        Altitude: {row['Altitude']} m<br>
        Pressure: {row['Pressure']} (bar)<br>
        Tap Flow Rate: {row['Tap Flow Rate']} (m3)<br>
        """
    
    for idx, row in excel_gdf_4326.iterrows():
        fill_color = colormap_tds(row['Total Dissolved Solids (TDS)'])
        
        folium.CircleMarker(
            location=[row.geometry.y, row.geometry.x],
            radius=3,
            popup=folium.Popup(create_popup_content(row), max_width=300),
            tooltip=row['Village'],
            color=fill_color,
            fillColor=fill_color,
            fillOpacity=1,
            weight=2
        ).add_to(marker_cluster_tds)
    
    folium.GeoJson(
        gdf,
        name="Dadupur",
        style_function=lambda feature: {
            'fillColor': 'blue',
            'color': 'black',
            'weight': 2,
            'fillOpacity': 0.1
        },
        tooltip=folium.GeoJsonTooltip(fields=['Name'], aliases=['Name: '])
    ).add_to(m)
    
    folium.LayerControl().add_to(m)
    colormap_tds.add_to(m)
    
    return m

# Flask routes
@server.route('/')
def home():
    return render_template('login.html')

@server.route('/login', methods=['POST'])
def login():
    if request.form.get('username') == 'JJM_Haridwar' and request.form.get('password') == 'dadupur':
        return redirect(url_for('dash_app'))
    return "Invalid credentials. Please try again."

@server.route('/dashboard/')
def dash_app():
    return app.index()

app.layout = html.Div([
    # Header
    html.Div([
        html.Div([
            html.Img(src="/static/logo.png", style={'height': '80px', 'width': 'auto'}),
            html.Img(src="/static/itc_logo.png", style={'height': '80px', 'width': 'auto', 'marginLeft': '10px'}),
            html.Div([
                html.H1("Water Monitoring Unit", style={'textAlign': 'center', 'color': '#010738', 'margin': '0'}),
                html.H3("Dadupur", style={'textAlign': 'center', 'color': '#010738', 'margin': '8px 0 0 0'}),
            ]),
            html.Div([
                html.Img(src="/static/EyeNet Aqua.png", style={'height': '90px', 'width': 'auto'}),
            ], style={'display': 'flex', 'alignItems': 'center'})
        ], style={'display': 'flex', 'justifyContent': 'space-between', 'alignItems': 'center', 'maxWidth': '1200px', 'margin': '0 auto', 'padding': '0 20px'})
    ], style={'width': '100%', 'backgroundColor': '#f5f5f5', 'padding': '10px 0', 'boxShadow': '0 2px 5px rgba(0,0,0,0.1)', 'borderRadius': '10px', 'marginBottom': '20px'}),
    
    # Main content
    html.Div([
        # Left side - Parameter boxes
        html.Div([
            html.Div([
                html.Span(id='live-date', style={'marginRight': '10px'}),
                html.Span(id='live-time')
            ], style={'textAlign': 'center', 'color': '#010738', 'fontSize': '20px', 'margin':'0'}),
            
            # Water Quality
            html.Div([
                html.H3("Water Quality", style={'color': '#3498db', 'marginBottom': '10px','textAlign': 'center'}),
                html.Div([
                    html.Span(id='water-quality-live-time', style={'fontSize': '18px', 'fontWeight': 'bold'})
                ], style={'textAlign': 'center', 'marginBottom': '10px'}),
                html.Div([
                    html.Div("Source pH", style={'fontWeight': 'bold', 'display': 'inline-block', 'width': '50%'}),
                    html.Div(id='source-ph-value', style={'display': 'inline-block', 'width': '50%'}),
                ], style={'backgroundColor': '#e8f4f8', 'padding': '10px', 'marginBottom': '5px', 'borderRadius': '5px'}),
                html.Div([
                    html.Div("Source TDS", style={'fontWeight': 'bold', 'display': 'inline-block', 'width': '50%'}),
                    html.Div(id='source-tds-value', style={'display': 'inline-block', 'width': '50%'}),
                ], style={'backgroundColor': '#e8f4f8', 'padding': '10px', 'marginBottom': '5px', 'borderRadius': '5px'}),
                html.Div([
                    html.Div("Source FRC", style={'fontWeight': 'bold', 'display': 'inline-block', 'width': '50%'}),
                    html.Div(id='source-frc-value', style={'display': 'inline-block', 'width': '50%'}),
                ], style={'backgroundColor': '#e8f4f8', 'padding': '10px', 'marginBottom': '5px', 'borderRadius': '5px'}),
                html.Div([
                    html.Div("Source Pressure", style={'fontWeight': 'bold', 'display': 'inline-block', 'width': '50%'}),
                    html.Div(id='source-pressure-value', style={'display': 'inline-block', 'width': '50%'}),
                ], style={'backgroundColor': '#e8f4f8', 'padding': '10px', 'marginBottom': '5px', 'borderRadius': '5px'}),
                html.Div([
                    html.Div("Source Flow", style={'fontWeight': 'bold', 'display': 'inline-block', 'width': '50%'}),
                    html.Div(id='source-flow-value', style={'display': 'inline-block', 'width': '50%'}),
                ], style={'backgroundColor': '#e8f4f8', 'padding': '10px', 'marginBottom': '5px', 'borderRadius': '5px'}),
            ], style={'backgroundColor': '#fff', 'padding': '15px', 'borderRadius': '10px', 'boxShadow': '0 2px 5px rgba(0,0,0,0.1)', 'marginBottom': '20px'}),
            
            # Water Quantity
            html.Div([
                html.H3("Water Quantity", style={'color': '#e74c3c', 'marginBottom': '10px','textAlign': 'center'}),
                html.Div([
                    html.Div("Today's Total Flow", style={'fontWeight': 'bold', 'display': 'inline-block', 'width': '50%'}),
                    html.Div(id='total-flow-value', style={'display': 'inline-block', 'width': '50%'}),
                ], style={'backgroundColor': '#fce8e8', 'padding': '10px', 'marginBottom': '5px', 'borderRadius': '5px'}),
                html.Div([
                    html.Div("Today's Pumping Hours", style={'fontWeight': 'bold', 'display': 'inline-block', 'width': '50%'}),
                    html.Div(id='pumping-hours-value', style={'display': 'inline-block', 'width': '50%'}),
                ], style={'backgroundColor': '#fce8e8', 'padding': '10px', 'marginBottom': '5px', 'borderRadius': '5px'}),
                html.Div([
                    html.Div("Today's LPCD", style={'fontWeight': 'bold', 'display': 'inline-block', 'width': '50%'}),
                    html.Div(id='lpcd-value', style={'display': 'inline-block', 'width': '50%'}),
                ], style={'backgroundColor': '#fce8e8', 'padding': '10px', 'marginBottom': '5px', 'borderRadius': '5px'}),
                html.Div([
                    html.Div("Weekly Average LPCD", style={'fontWeight': 'bold', 'display': 'inline-block', 'width': '50%'}),
                    html.Div(id='weekly-lpcd-value', style={'display': 'inline-block', 'width': '50%'}),
                ], style={'backgroundColor': '#fce8e8', 'padding': '10px', 'marginBottom': '5px', 'borderRadius': '5px'}),
            ], style={'backgroundColor': '#fff', 'padding': '15px', 'borderRadius': '10px', 'boxShadow': '0 2px 5px rgba(0,0,0,0.1)', 'marginBottom': '20px'}),
            
            # Water Quantity Data Table
            html.Div([
                html.H3("Water Quantity Data", style={'textAlign': 'center', 'marginBottom': '20px'}),
                 html.Div([
            dcc.DatePickerRange(
                id='historical-date-picker-range',
                start_date=datetime.now().date() - timedelta(days=7),
                end_date=datetime.now().date(),
                display_format='YYYY-MM-DD'
            ),
            html.Button('View Historical Data', id='view-historical-data-button', n_clicks=0, 
                        style={'marginLeft': '10px', 'backgroundColor': '#4CAF50', 'color': 'white', 'border': 'none', 'padding': '10px 20px', 'borderRadius': '5px', 'cursor': 'pointer'}),
        ], style={'marginBottom': '20px', 'display': 'flex', 'justifyContent': 'center', 'alignItems': 'center'}),
        
                dash_table.DataTable(
                    id='water-quantity-table',
                    columns=[
                        {"name": "Date", "id": "date"},
                        {"name": "Total Flow (kL)", "id": "total_flow"},
                        {"name": "Pumping Hours", "id": "pumping_hours"},
                        {"name": "LPCD", "id": "lpcd"},
                    ],
                    data=[],  # This will be populated by a callback
                    style_table={'overflowX': 'auto'},
                    style_cell={'textAlign': 'left', 'padding': '10px'},
                    style_header={
                        'backgroundColor': 'rgb(230, 230, 230)',
                        'fontWeight': 'bold'
                    },
                    page_size=7,
                ),
                html.Button('Download Water Quantity Data', id='download-water-quantity-button', n_clicks=0, 
                            style={'marginTop': '10px', 'backgroundColor': '#008CBA', 'color': 'white', 'border': 'none', 'padding': '10px 20px', 'borderRadius': '5px', 'cursor': 'pointer'}),
                dcc.Download(id="download-water-quantity-csv"),
            ], style={'backgroundColor': '#fff', 'padding': '20px', 'borderRadius': '10px', 'boxShadow': '0 2px 5px rgba(0,0,0,0.1)', 'marginBottom': '20px'}),
        ], style={'width': '48%', 'marginRight': '2%'}),
        
        # Right side - Map
        html.Div([
            html.H3("Dadupur, Haridwar Map", style={'marginBottom': '10px','textAlign': 'center'}),
            html.Iframe(id='map-iframe', srcDoc=create_map().get_root().render(), 
                        style={'width': '100%', 'height': '600px', 'border': 'none', 'borderRadius': '5px'}),
        ], style={'width': '50%', 'height': 'auto','backgroundColor': '#fff', 'padding': '15px', 'borderRadius': '10px', 'boxShadow': '0 2px 5px rgba(0,0,0,0.1)'}),
    ], style={'display': 'flex', 'justifyContent': 'space-between', 'marginBottom': '20px'}),

    # Stationary Data Analysis
    html.Div([
        html.H3("Stationary Sensor Data Analysis", style={'textAlign': 'center', 'marginBottom': '20px'}),
        html.Div([
            dcc.Dropdown(
                id='time-duration-dropdown',
                options=[{'label': k, 'value': k} for k in TIME_DURATIONS.keys()],
                value='3 Hours',
                style={'width': '200px', 'marginBottom': '20px'}
            ),
        ]),
        html.Div([
            dcc.Graph(id=f'{param}-graph', style={'height': '300px', 'width': '48%', 'display': 'inline-block', 'marginBottom': '20px'})
            for param in ['pH', 'TDS', 'FRC', 'pressure', 'flow']
        ]),
    ], style={'backgroundColor': '#fff', 'padding': '20px', 'borderRadius': '10px', 'boxShadow': '0 2px 5px rgba(0,0,0,0.1)', 'marginBottom': '20px'}),
    
    # Mobile Data Analysis
     html.Div([
        html.H3("Mobile Sensor Data Analysis", style={'textAlign': 'center', 'marginBottom': '20px'}),
        html.Div([
            dcc.Dropdown(
                id='bothwq-parameter-dropdown',
                options=[
                    {'label': 'pH', 'value': 'pH'},
                    {'label': 'TDS', 'value': 'Total Dissolved Solids (TDS)'},
                    {'label': 'FRC', 'value': 'Free Residual Chlorine (FRC)'}
                ],
                value='pH',
                style={'width': '200px', 'marginRight': '10px'}
            ),
            dcc.Dropdown(
                id='zone-dropdown',
                options=[],  # This will be populated by the callback
                value=None,
                placeholder="Select a Zone (optional)",
                style={'width': '200px', 'marginRight': '10px'}
            ),
            dcc.DatePickerRange(
                id='date-range-picker',
                display_format='YYYY-MM-DD'
            ),
        ], style={'display': 'flex', 'justifyContent': 'center', 'marginBottom': '20px'}),
        dcc.Graph(id='bothwq-graph'),
    ], style={'backgroundColor': '#fff', 'padding': '20px', 'borderRadius': '10px', 'boxShadow': '0 2px 5px rgba(0,0,0,0.1)', 'marginBottom': '20px'}),
    
    # Historical Data (centered)
    html.Div([
        html.H3("Historical Data", style={'marginBottom': '20px', 'textAlign': 'center'}),
        html.Div([
            dcc.DatePickerRange(
                id='date-picker-range',
                start_date=datetime.now().date() - timedelta(days=7),
                end_date=datetime.now().date(),
                display_format='YYYY-MM-DD'
            ),
            html.Button('View Data', id='view-data-button', n_clicks=0, 
                        style={'marginLeft': '10px', 'backgroundColor': '#4CAF50', 'color': 'white', 'border': 'none', 'padding': '10px 20px', 'borderRadius': '5px', 'cursor': 'pointer'}),
        ], style={'marginBottom': '20px', 'display': 'flex', 'justifyContent': 'center', 'alignItems': 'center'}),
        html.Div(id='data-table-container', style={'display': 'flex', 'justifyContent': 'center'}),
        dash_table.DataTable(id='historical-data-table', columns=[{"name": i, "id": i} for i in COLUMNS + ['timestamp']],
                             page_size=10, style_table={'overflowX': 'auto'}),
        
        html.Button('Download CSV', id='download-csv-button', n_clicks=0, style={'marginTop': '10px', 'textAlign': 'center'}),
        dcc.Download(id="download-dataframe-csv"),
        
    ], style={'backgroundColor': '#fff', 'padding': '20px', 'borderRadius': '10px', 'boxShadow': '0 2px 5px rgba(0,0,0,0.1)', 'marginBottom': '20px', 'alignItems': 'center'}),
    
    # Footer
    html.Footer([
        html.Div([
            html.P('Dashboard - Powered by ICCW', style={'fontSize': '15px', 'margin': '5px 0'}),
            html.P('Technology Implementation Partner - EyeNet Aqua', style={'fontSize': '15px', 'margin': '5px 0'}),
        ], style={'maxWidth': '1200px', 'margin': '0 auto', 'padding': '0 20px', 'textAlign': 'center'})
    ], style={'width': '100%', 'backgroundColor': '#f9f9f9', 'padding': '20px 0', 'marginTop': '20px', 'boxShadow': '0 -2px 5px rgba(0,0,0,0.1)', 'borderRadius': '10px 10px 0 0'}),
    
    # Intervals for updates
    dcc.Interval(id='live-update-interval', interval=1000, n_intervals=0),
    dcc.Interval(id='interval-component', interval=60000, n_intervals=0),
    
    # Store for historical data
    dcc.Store(id='historical-data-store')
], style={'maxWidth': '1200px', 'margin': '0 auto', 'padding': '0 20px'})

# Callbacks
@app.callback(
    [Output('source-ph-value', 'children'),
     Output('source-tds-value', 'children'),
     Output('source-frc-value', 'children'),
     Output('source-pressure-value', 'children'),
     Output('source-flow-value', 'children'),
     Output('total-flow-value', 'children'),
     Output('pumping-hours-value', 'children'),
     Output('lpcd-value', 'children'),
     Output('weekly-lpcd-value', 'children')] +
    [Output(f'{param}-graph', 'figure') for param in ['pH', 'TDS', 'FRC', 'pressure', 'flow']],
    [Input('interval-component', 'n_intervals'),
     Input('time-duration-dropdown', 'value')]
)

def update_dashboard(n, time_duration):
    try:
        data = fetch_data_from_api(API_URL)
        df = process_data(data)
        
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        today = datetime.now().date()
        
        # Filter data for today
        df_today = df[df['timestamp'].dt.date == today]
        if df.empty:
            return ["N/A"] * 9 + [go.Figure()] * 6

        df['timestamp'] = pd.to_datetime(df['timestamp'])
        end_time = df['timestamp'].max()
        start_time = end_time - TIME_DURATIONS[time_duration]
        df_filtered = df[(df['timestamp'] >= start_time) & (df['timestamp'] <= end_time)]

        latest = df.iloc[-1]
        
           # Calculate daily total flow
        daily_flow = df_today['source_flow'].sum()/1000
        
        # Calculate daily pumping hours
        df_today['is_pumping'] = df_today['source_flow'] > 0
        daily_pumping_hours = df_today['is_pumping'].sum() / 6  # Assuming data is collected every 10 minutes
        
        # Calculate LPCD
        population = 10000  # Adjust this value as needed
        daily_flow = df_today['source_flow'].sum()   # Convert to liters
        daily_lpcd = df_today['source_flow'].sum() * 1000 / population if population > 0 else 0
        
        # Calculate weekly LPCD
        one_week_ago = today - timedelta(days=7)
        df_week = df[(df['timestamp'].dt.date > one_week_ago) & (df['timestamp'].dt.date <= today)]
        weekly_flow = df_week.groupby(df_week['timestamp'].dt.date)['source_flow'].sum() * 1000  # Convert to liters
        weekly_lpcd = (weekly_flow / population).mean() if population > 0 else 0

        # Create individual graphs
        individual_graphs = []
        for param in ['pH', 'TDS', 'FRC', 'pressure', 'flow']:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df_filtered['timestamp'], y=df_filtered[f'source_{param}'], mode='lines+markers'))
            fig.update_layout(
                title=f'Source {param}',
                xaxis_title='Time',
                yaxis_title=f'{param} ({UNITS[param]})',
                height=300
            )
            individual_graphs.append(fig)

        # Create periodic analysis chart
        periodic_fig = go.Figure()
        for column in COLUMNS:
            periodic_fig.add_trace(go.Scatter(x=df_filtered['timestamp'], y=df_filtered[column], mode='lines+markers', name=column))
        periodic_fig.update_layout(
            title=f'Periodic Analysis - Last {time_duration}',
            xaxis_title='Time',
            yaxis_title='Value',
            height=600,
            legend_title='Parameters'
        )

        return [
            f"{latest['source_pH']:.2f}",
            f"{latest['source_TDS']:.2f} ppm",
            f"{latest['source_FRC']:.2f} ppm",
            f"{latest['source_pressure']:.2f} bar",
            f"{latest['source_flow']:.2f} kL per 10 mins",
            f"{daily_flow:.2f} kL",
            f"{daily_pumping_hours:.2f} hrs",
            f"{daily_lpcd:.2f}",
            f"{weekly_lpcd:.2f}"
        ] + individual_graphs
    except Exception as e:
        print(f"Error updating dashboard: {str(e)}")
        return ["Error"] * 9 + [go.Figure()] * 5

# Add a new callback for the BOTHWQ Data Analysis graph
@app.callback(
    [Output('bothwq-graph', 'figure'),
     Output('zone-dropdown', 'options'),
     Output('date-range-picker', 'min_date_allowed'),
     Output('date-range-picker', 'max_date_allowed'),
     Output('date-range-picker', 'start_date'),
     Output('date-range-picker', 'end_date')],
    [Input('bothwq-parameter-dropdown', 'value'),
     Input('zone-dropdown', 'value'),
     Input('date-range-picker', 'start_date'),
     Input('date-range-picker', 'end_date')]
)
def update_bothwq_graph(selected_parameter, selected_zone, start_date, end_date):
    # Read the data
    df = pd.read_excel("Uttarakhand.xlsx", sheet_name=0)
    
    # Convert Date to datetime and sort
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date')
    
    # Prepare zone options
    zone_options = [{'label': f'Zone {i}', 'value': i} for i in df['Zone'].unique()]
    
    # Get min and max dates
    min_date = df['Date'].min()
    max_date = df['Date'].max()
    
    # Set default date range if not provided
    if not start_date:
        start_date = min_date
    if not end_date:
        end_date = max_date
    
    # Filter by date range
    df = df[(df['Date'] >= start_date) & (df['Date'] <= end_date)]
    
    # Filter by zone if selected
    if selected_zone:
        df = df[df['Zone'] == selected_zone]
    
    # Create the figure
    fig = go.Figure()
    
    # Add line plot for each zone
    for zone in df['Zone'].unique():
        zone_df = df[df['Zone'] == zone]
        fig.add_trace(go.Scatter(
            x=zone_df['Date'],
            y=zone_df[selected_parameter],
            mode='lines+markers',
            name=f'Zone {zone}',
            text=[f"Date: {d}<br>Zone: {z}<br>ID: {id}<br>{selected_parameter}: {v}" 
                  for d, z, id, v in zip(zone_df['Date'], zone_df['Zone'], zone_df['Assigned ID'], zone_df[selected_parameter])],
            hoverinfo='text'
        ))
    
    # Update layout
    fig.update_layout(
        title=f'{selected_parameter} vs Date by Zone',
        xaxis_title='Date',
        yaxis_title=selected_parameter,
        height=600,
        legend_title='Zone',
        hovermode='closest'
    )
    
    return fig, zone_options, min_date, max_date, start_date, end_date

    
# Update the options in the dropdown to include all relevant parameters
@app.callback(
    Output('bothwq-parameter-dropdown', 'options'),
    Input('interval-component', 'n_intervals')
)
def update_dropdown_options(n):
    df_bothwq = pd.read_excel('Uttarakhand.xlsx', sheet_name=0)
    parameters = ['pH', 'Total Dissolved Solids (TDS)', 'Free Residual Chlorine (FRC)', 
                  'Pressure', 'Tap Flow Rate', 'Altitude']
    return [{'label': param, 'value': param} for param in parameters if param in df_bothwq.columns]

# Other callbacks (historical data, table update, CSV download) remain the same
@app.callback(
    Output('historical-data-store', 'data'),
    [Input('view-data-button', 'n_clicks')],
    [State('date-picker-range', 'start_date'),
     State('date-picker-range', 'end_date')]
)

def fetch_historical_data(n_clicks, start_date, end_date):
    if n_clicks > 0:
        try:
            data = fetch_data_from_api(API_URL)
            df = process_data(data)
            
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            mask = (df['timestamp'].dt.date >= pd.to_datetime(start_date).date()) & \
                   (df['timestamp'].dt.date <= pd.to_datetime(end_date).date())
            filtered_df = df.loc[mask]
            
            return filtered_df.to_dict('records')
        except Exception as e:
            print(f"Error fetching historical data: {str(e)}")
            return []
    return []

@app.callback(
    [Output('live-date', 'children'),
     Output('live-time', 'children')],
    [Input('live-update-interval', 'n_intervals')]
)

def update_live_time(n):
    now = datetime.now()
    date_string = now.strftime("%Y-%m-%d")
    time_string = now.strftime("%H:%M:%S")
    return date_string, time_string

@app.callback(
    Output('historical-data-table', 'data'),
    [Input('historical-data-store', 'data')]
)

def update_table(data):
    return data or []

@app.callback(
    Output("download-dataframe-csv", "data"),
    [Input("download-csv-button", "n_clicks")],
    [State('historical-data-store', 'data')]
)

def download_csv(n_clicks, data):
    if n_clicks > 0 and data:
        df = pd.DataFrame(data)
        return dcc.send_data_frame(df.to_csv, "historical_data.csv", index=False)

# Update the water quantity table callback
@app.callback(
    Output('water-quantity-table', 'data'),
    Input('interval-component', 'n_intervals')
)
def update_water_quantity_table(n):
    # Fetch and process your data here
    data = fetch_data_from_api(API_URL)
    df = process_data(data)
    
    # Debug: Print column names and first few rows
    #print("Columns:", df.columns)
    #print("First few rows:")
    #print(df.head())
    
    # Convert timestamp to datetime
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Create the 'is_pumping' column
    df['is_pumping'] = (df['source_flow'] > 0).astype(int)
    
    # Calculate daily values
    df['date'] = df['timestamp'].dt.date
    daily_data = df.groupby('date').agg({
        'source_flow': 'sum',
        'is_pumping': 'sum'
    }).reset_index()
    
    # Calculate LPCD (assuming population of 10000)
    population = 10000
    daily_data['lpcd'] = daily_data['source_flow'] * 1000 / population
    
    # Prepare data for the table
    table_data = daily_data.rename(columns={
        'source_flow': 'total_flow',
        'is_pumping': 'pumping_hours'
    })
    # Round pumping hours to 2 decimal places
    table_data['pumping_hours'] = (table_data['pumping_hours'] / 6).round(2)  # Assuming data every 10 minutes
    
    # Round other numeric columns
    table_data['total_flow'] = table_data['total_flow'].round(2)
    table_data['lpcd'] = table_data['lpcd'].round(2)
    
    return table_data.to_dict('records')

@app.callback(
    Output("download-water-quantity-csv", "data"),
    Input("download-water-quantity-button", "n_clicks"),
    prevent_initial_call=True)

def download_water_quantity_csv(n_clicks):
    
    # Fetch and process your data here (similar to update_water_quantity_table)
    data = fetch_data_from_api(API_URL)
    df = process_data(data)
    
    # Calculate daily values and LPCD
    df['date'] = pd.to_datetime(df['timestamp']).dt.date
    daily_data = df.groupby('date').agg({
        'source_flow': 'sum',
        'is_pumping': 'sum'
    }).reset_index()
    
    population = 10000
    daily_data['lpcd'] = daily_data['source_flow'] * 1000 / population
    daily_data['pumping_hours'] = daily_data['is_pumping'] / 6
    
    # Prepare data for CSV
    csv_data = daily_data.rename(columns={
        'source_flow': 'total_flow',
        'is_pumping': 'pumping_hours'
    })[['date', 'total_flow', 'pumping_hours', 'lpcd']]
    
    return dcc.send_data_frame(csv_data.to_csv, "water_quantity_data.csv", index=False)

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    debug = os.environ.get('DEBUG', 'False').lower() == 'true'
    server.run(host='0.0.0.0', port=port, debug=debug)