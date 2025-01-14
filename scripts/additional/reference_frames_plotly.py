import numpy as np
import dash
from dash import dcc, html, Output, Input
import plotly.graph_objects as go
from color_palette import get_color_list

# Sample colors list
colors = get_color_list()
# Function to calculate Cartesian coordinates from spherical coordinates
def spherical_to_cartesian(r, azimuth, elevation):
    x = r * np.cos(elevation) * np.cos(azimuth)
    y = r * np.cos(elevation) * np.sin(azimuth)
    z = r * np.sin(elevation)
    return np.array([x, y, z])

# Transformation matrix for AZR frame
def transformation_AZR_from_W(azimuth, elevation):
    phi = azimuth
    beta = elevation
    transformation = np.array([
        [-np.sin(phi), np.cos(phi), 0],
        [-np.sin(beta) * np.cos(phi), -np.sin(beta) * np.sin(phi), np.cos(beta)],
        [np.cos(beta) * np.cos(phi), np.cos(beta) * np.sin(phi), np.sin(beta)]
    ])
    return transformation

def transformation_C_from_AZR(chi):
    transformation = np.array([
        [np.sin(chi), np.cos(chi), 0],
        [-np.cos(chi), np.sin(chi), 0],
        [0, 0, 1],
    ])
    return transformation
def aerodynamic_pitch(velocity_wind_apparent):
        va_tau, va_n, va_r = velocity_wind_apparent
        sqrt_va_tau_n = np.sqrt(va_tau**2 + va_n**2)
        return np.arctan2(va_r, sqrt_va_tau_n)


def aerodynamic_yaw(velocity_wind_apparent):
    va_tau, va_n, _ = velocity_wind_apparent
    return np.arctan(va_n/va_tau)

def transformation_C_from_A(kite_vel, wind_vel, roll=0):
    apparent_vel = wind_vel - kite_vel
    theta_a = aerodynamic_pitch(apparent_vel)
    chi_a = aerodynamic_yaw(apparent_vel)

    Pitch = np.array([
        [np.cos(theta_a), 0, np.sin(theta_a)],
        [0, 1, 0],
        [-np.sin(theta_a), 0, np.cos(theta_a)]
    ])
    Yaw = np.array([
        [np.cos(chi_a), -np.sin(chi_a), 0],
        [np.sin(chi_a), np.cos(chi_a), 0],
        [0, 0, 1]
    ])
    # add roll x-axis rotation
    Roll = np.array([
        [1, 0, 0],
        [0, np.cos(roll), -np.sin(roll)],
        [0, np.sin(roll), np.cos(roll)]
    ])
    T = Yaw @ Pitch @ Roll

    return T

def transformation_C_from_K(pitch, roll):

    Pitch = np.array([
        [np.cos(pitch), 0, np.sin(pitch)],
        [0, 1, 0],
        [-np.sin(pitch), 0, np.cos(pitch)]
    ])
    Roll = np.array([
        [1, 0, 0],
        [0, np.cos(roll), -np.sin(roll)],
        [0, np.sin(roll), np.cos(roll)]
    ])
    T = Pitch @ Roll
    return T

# Initialize the Dash app
app = dash.Dash(__name__)

# App Layout
app.layout = html.Div([
    html.H1("Interactive Kite Frames Visualization", style={'textAlign': 'center'}),
    html.Div([
        # Left panel for inputs
        html.Div([
            html.Label("Azimuth (degrees):"),
            dcc.Input(
                id="azimuth-input", type="number", value=0,
                placeholder="Enter azimuth", step=5,
                style={'margin-bottom': '20px'}
            ),
            html.Label("Elevation (degrees):"),
            dcc.Input(
                id="elevation-input", type="number", value=0,
                placeholder="Enter elevation", step=5,
                style={'margin-bottom': '20px'}
            ),
            html.Label("Course (degrees):"),
            dcc.Input(
                id="course-input", type="number", value=0,
                placeholder="Enter course", step=5,
                style={'margin-bottom': '20px'}
            ),
            html.Label("Wind Speed (m/s):"),
            dcc.Input(
                id="wind-speed-input", type="number", value=10,
                placeholder="Enter wind speed", step=1,
                style={'margin-bottom': '20px'}
            ),
            html.Label("Reeling Speed (m/s):"),
            dcc.Input(
                id="reeling-speed-input", type="number", value=0,
                placeholder="Enter reeling speed", step=1,
                style={'margin-bottom': '20px'}
            ),
            html.Label("Kite Speed (m/s):"),
            dcc.Input(
                id="kite-speed-input", type="number", value=20,
                placeholder="Enter kite speed", step=1,
                style={'margin-bottom': '20px'}
            ),
            html.Label("Roll (degrees):"),
            dcc.Input(
                id="roll-input", type="number", value=0,
                placeholder="Enter roll", step=5,
                style={'margin-bottom': '20px'}
            ),
            html.Label("Pitch (degrees):"),
            dcc.Input(
                id="pitch-input", type="number", value=0,
                placeholder="Enter pitch", step=5,
                style={'margin-bottom': '20px'}
            ),
        ], style={'width': '20%', 'padding': '20px', 'display': 'flex', 'flex-direction': 'column'}),

        # Right panel for the plot
        html.Div([
            dcc.Graph(
                id="3d-plot",
                style={'height': '700px', 'width': '100%'}  # Adjust height and width
            )
        ], style={'width': '80%', 'padding': '20px'})
    ], style={'display': 'flex', 'flex-direction': 'row'})
])

# Callback to update the plot based on input fields
@app.callback(
    Output("3d-plot", "figure"),
    [
        Input("azimuth-input", "value"),
        Input("elevation-input", "value"),
        Input("course-input", "value"),
        Input("wind-speed-input", "value"),
        Input("reeling-speed-input", "value"),
        Input("kite-speed-input", "value"),
        Input("roll-input", "value"),
        Input("pitch-input", "value")
    ]
)
def update_plot(
    azimuth_deg, elevation_deg, course_deg, wind_speed=10, 
    reeling_speed=0, kite_speed=20, roll_deg=0, pitch_deg=0,
    eye_x=1.25, eye_y=1.25, eye_z=1.25
):
    azimuth = np.radians(azimuth_deg)
    elevation = np.radians(elevation_deg)
    course = np.radians(course_deg)
    roll = np.radians(roll_deg)
    pitch = np.radians(pitch_deg)
    r = 5
    origin = spherical_to_cartesian(r, azimuth, elevation)

    # Create 3D plot
    fig = go.Figure()

    # Add a dotted line from (0, 0, 0) to the origin
    fig.add_trace(go.Scatter3d(
        x=[0, origin[0]],
        y=[0, origin[1]],
        z=[0, origin[2]],
        mode="lines",
        line=dict(color="black", width=2, dash="dot"),  # Black dotted line
        name="Tether"
    ))

    # Add origin point
    fig.add_trace(go.Scatter3d(
        x=[origin[0]],
        y=[origin[1]],
        z=[origin[2]],
        mode="markers+text",
        marker=dict(size=5),
        name=f"Tether attachment point"
    ))

    # Plot wind with customized thickness
    axis_thickness = {"x": 8, "y": 5, "z": 2}  # Define thickness for WIND axes
    for axis in ["x", "y", "z"]:
        fig.add_trace(go.Scatter3d(
            x=[0, 1 if axis == "x" else 0],
            y=[0, 1 if axis == "y" else 0],
            z=[0, 1 if axis == "z" else 0],
            mode="lines",
            line=dict(color=colors[0], width=axis_thickness[axis]),
            name=f"WND {axis}-axis"
        ))



    # Get spherical coordinates
    transformation = transformation_AZR_from_W(azimuth, elevation).T

    # Plot AZR frame
    for axis in ["x", "y", "z"]:
        fig.add_trace(go.Scatter3d(
            x=[origin[0], origin[0] + transformation[0, "xyz".index(axis)]],
            y=[origin[1], origin[1] + transformation[1, "xyz".index(axis)]],
            z=[origin[2], origin[2] + transformation[2, "xyz".index(axis)]],
            mode="lines",
            line=dict(color=colors[1], width=axis_thickness[axis]),
            name=f"AZR {axis}-axis"
        ))

    

    # Add C frame
    transformation_C_from_W = np.matmul(transformation_C_from_AZR(course), transformation_AZR_from_W(azimuth, elevation))
    transformation_W_from_C = transformation_C_from_W.T
    for axis in ["x", "y", "z"]:
        fig.add_trace(go.Scatter3d(
            x=[origin[0], origin[0] + transformation_W_from_C[0, "xyz".index(axis)]],
            y=[origin[1], origin[1] + transformation_W_from_C[1, "xyz".index(axis)]],
            z=[origin[2], origin[2] + transformation_W_from_C[2, "xyz".index(axis)]],
            mode="lines",
            line=dict(color=colors[2], width=axis_thickness[axis]),
            name=f"Course    {axis}-axis"
        ))
    wind_vel = transformation_C_from_W @ np.array([wind_speed, 0, 0])
    kite_vel = np.array([kite_speed, 0, reeling_speed])
    transformation_W_from_A = np.matmul(transformation_W_from_C,transformation_C_from_A(kite_vel, wind_vel, roll))

    transformation_W_from_K = np.matmul(transformation_W_from_C, transformation_C_from_K(pitch, roll))
    z_kite = transformation_W_from_K @ np.array([0, 0, 1])
    origin_kite = origin + z_kite

    fig.add_trace(go.Scatter3d(
        x=[origin[0], origin_kite[0]],
        y=[origin[1], origin_kite[1]],
        z=[origin[2], origin_kite[2]],
        mode="lines",
        line=dict(color="black", width=2, dash="dash"),
        name="Bridle"
    ))

    # Add Aero frame
    for axis in ["x", "y", "z"]:
        if axis == "x":
            label = "-drag"
        elif axis == "y":
            label = "sideforce"
        else:
            label = "lift"
        fig.add_trace(go.Scatter3d(
            x=[origin_kite[0], origin_kite[0] + transformation_W_from_A[0, "xyz".index(axis)]],
            y=[origin_kite[1], origin_kite[1] + transformation_W_from_A[1, "xyz".index(axis)]],
            z=[origin_kite[2], origin_kite[2] + transformation_W_from_A[2, "xyz".index(axis)]],
            mode="lines",
            line=dict(color=colors[3], width=axis_thickness[axis]),
            name=f"Aero {axis}-axis -> {label}"
        ))

    # Add kite frame
    for axis in ["x", "y", "z"]:
        fig.add_trace(go.Scatter3d(
            x=[origin_kite[0], origin_kite[0] + transformation_W_from_K[0, "xyz".index(axis)]],
            y=[origin_kite[1], origin_kite[1] + transformation_W_from_K[1, "xyz".index(axis)]],
            z=[origin_kite[2], origin_kite[2] + transformation_W_from_K[2, "xyz".index(axis)]],
            mode="lines",
            line=dict(color=colors[4], width=axis_thickness[axis]),
            name=f"Kite {axis}-axis"
        ))

    # Add quarter sphere for reference
    u = np.linspace(0, np.pi / 2, 50)  # Quarter azimuth range
    v = np.linspace(-np.pi / 2, np.pi / 2, 50)  # Quarter elevation range
    x = r * np.outer(np.cos(v), np.sin(u))  # Radius = 2
    y = r * np.outer(np.sin(v), np.sin(u))
    z = r * np.outer(np.ones(np.size(u)), np.cos(u))

    fig.add_trace(go.Surface(x=x, y=y, z=z, opacity=0.3, showscale=False, colorscale='Viridis'))

    # Layout adjustments with equal axis spacing
    fig.update_layout(
        scene=dict(
            xaxis=dict(title="X", range=[-0.5, 1.5*r]),
            yaxis=dict(title="Y", range=[-1.5*r,1.5*r]),
            zaxis=dict(title="Z", range=[-0.5, 1.5*r]),
            aspectmode='data',  # Ensures equal axis scaling
            camera=dict(
                eye=dict(x=eye_x, y=eye_y, z=eye_z)  # Set camera eye position
            )
        ),
        title=f"Frames on a Quarter Sphere (Azimuth: {azimuth_deg}°, Elevation: {elevation_deg}°, Course: {course_deg}°)",
        margin=dict(l=0, r=0, t=40, b=0)
    )


    return fig

# Run the app
if __name__ == "__main__":
    app.run_server(debug=True)
