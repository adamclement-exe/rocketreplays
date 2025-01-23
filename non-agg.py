import dash
from dash import dcc, html, Input, Output, State, dash_table
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import json
import io
import base64

# Initialize Dash app
app = dash.Dash(__name__, external_stylesheets=["https://cdn.jsdelivr.net/npm/water.css@2/out/water.css"])
app.title = "RL-DASH⚽️🏎️"
app._favicon = "rl.png"

# App layout
app.layout = html.Div([
    html.H1("⚽️🏎️Rocket League Player Cluster App", style={'textAlign': 'center', 'marginTop': '20px'}),
    html.Div([
        html.P("""
            Welcome to the Rocket League Player Cluster App — an interactive platform for analyzing player performance metrics.
            Upload JSON replay files, and explore dynamic visualizations, player clustering, and performance insights.
        """, style={'textAlign': 'center', 'margin': '20px auto', 'maxWidth': '800px', 'fontSize': '18px'}),
        html.P("""
            WARNING - UPLOAD CAN TAKE SEVERAL MINUTES WITH MULTIPLE JSON FILES.
        """, style={'textAlign': 'center', 'margin': '20px auto', 'maxWidth': '800px', 'fontSize': '12px',
                    'color': '#b30000', 'fontWeight': 'bold', 'backgroundColor': '#ffe6e6',
                    'padding': '10px', 'borderRadius': '5px'})
    ]),
    html.Div([
        html.H2("Upload JSON Files"),
        dcc.Upload(
            id='upload-json',
            children=html.Div(['Drag and Drop or ', html.A('Select Files')]),
            style={'width': '100%', 'height': '60px', 'lineHeight': '60px',
                   'borderWidth': '1px', 'borderStyle': 'dashed', 'borderRadius': '5px',
                   'textAlign': 'center', 'margin': '10px', 'backgroundColor': '#f9f9f9'},
            multiple=True
        ),
    ]),
    html.Div(id='upload-status', style={'margin': '10px'}),
    dcc.Loading(
        id="loading-spinner",
        type="circle",
        children=[
            html.Div([
                html.H2("Correlation Heatmap"),
                dcc.Graph(id='heatmap'),
                dash_table.DataTable(id='heatmap-data', style_table={'overflowX': 'auto'}),
            ], id='heatmap-container', style={'display': 'none'}),
            html.Div([
                html.H2("Scree Plot"),
                dcc.Graph(id='scree-plot'),
                dash_table.DataTable(id='scree-data', style_table={'overflowX': 'auto'}),
            ], id='scree-container', style={'display': 'none'}),
            html.Div([
                html.H2("PCA Biplot"),
                dcc.Graph(id='biplot'),
                dash_table.DataTable(id='biplot-data', style_table={'overflowX': 'auto'}),
            ], id='biplot-container', style={'display': 'none'}),
            html.Div([
                html.H2("Player Clusters"),
                dcc.Graph(id='cluster-plot'),
                dash_table.DataTable(id='cluster-data', style_table={'overflowX': 'auto'}),
            ], id='cluster-container', style={'display': 'none'}),
            html.Div([
                html.H2("Match Outcomes by Cluster"),
                dcc.Graph(id='win-ratio-plot'),
                dash_table.DataTable(id='win-ratio-data', style_table={'overflowX': 'auto'}),
            ], id='win-ratio-container', style={'display': 'none'}),
        ]
    )
])

@app.callback(
    Output('upload-status', 'children'),
    Input('upload-json', 'loading_state')
)
def update_upload_status(loading_state):
    if not loading_state:
        return ""
    if loading_state["is_loading"]:
        return "Processing uploaded files..."
    return "Upload complete!"

@app.callback(
    [
        Output('heatmap', 'figure'),
        Output('heatmap-data', 'data'),
        Output('heatmap-data', 'columns'),
        Output('scree-plot', 'figure'),
        Output('scree-data', 'data'),
        Output('scree-data', 'columns'),
        Output('biplot', 'figure'),
        Output('biplot-data', 'data'),
        Output('biplot-data', 'columns'),
        Output('cluster-plot', 'figure'),
        Output('cluster-data', 'data'),
        Output('cluster-data', 'columns'),
        Output('win-ratio-plot', 'figure'),
        Output('win-ratio-data', 'data'),
        Output('win-ratio-data', 'columns'),
        Output('heatmap-container', 'style'),
        Output('scree-container', 'style'),
        Output('biplot-container', 'style'),
        Output('cluster-container', 'style'),
        Output('win-ratio-container', 'style'),
    ],
    [Input('upload-json', 'contents')],
    [State('upload-json', 'filename')]
)
def update_dashboard(contents, filenames):
    if contents is None:
        return (
            {}, [], [], {}, [], [], {}, [], [], {}, [], [], {}, [], [],
            {'display': 'none'}, {'display': 'none'}, {'display': 'none'}, {'display': 'none'}, {'display': 'none'}
        )

    all_player_data = []
    try:
        for content, filename in zip(contents, filenames):
            content_type, content_string = content.split(',')
            decoded = base64.b64decode(content_string)
            match_data = json.load(io.StringIO(decoded.decode('utf-8')))
            player_stats = pd.json_normalize(match_data['properties']['PlayerStats'])
            player_stats['Team0Score'] = match_data['properties'].get('Team0Score', np.nan)
            player_stats['Team1Score'] = match_data['properties'].get('Team1Score', np.nan)
            player_stats['PlayerID'] = player_stats.index
            all_player_data.append(player_stats)
    except Exception as e:
        return {}, [], [], {}, [], [], {}, [], [], {}, [], [], {}, [], [], f"Error processing files: {str(e)}"

    combined_data = pd.concat(all_player_data, ignore_index=True)
    relevant_metrics = ['Score', 'Assists', 'Saves', 'Shots', 'Goals', 'Name', 'PlayerID']
    player_data = combined_data[relevant_metrics]

    correlation_matrix = player_data.drop(columns=['Name', 'PlayerID']).corr()
    heatmap_fig = px.imshow(
        correlation_matrix, text_auto=True, color_continuous_scale='rdbu', title='Correlation Heatmap'
    )
    heatmap_table_data = correlation_matrix.reset_index().to_dict('records')
    heatmap_table_columns = [{'name': col, 'id': col} for col in correlation_matrix.columns]

    scaler = StandardScaler()
    standardized_data = scaler.fit_transform(player_data.drop(columns=['Name', 'PlayerID']))
    pca = PCA()
    pca_components = pca.fit_transform(standardized_data)
    explained_variance = pca.explained_variance_ratio_

    scree_fig = go.Figure()
    scree_fig.add_trace(go.Scatter(
        x=list(range(1, len(explained_variance) + 1)),
        y=explained_variance,
        mode='lines+markers',
        name='Explained Variance'
    ))
    scree_fig.update_layout(title="Scree Plot: Explained Variance", xaxis_title="Principal Component", yaxis_title="Explained Variance Ratio")
    scree_table_data = pd.DataFrame({"Principal Component": range(1, len(explained_variance) + 1), "Explained Variance": explained_variance}).to_dict('records')
    scree_table_columns = [{'name': col, 'id': col} for col in ['Principal Component', 'Explained Variance']]

    loadings = pca.components_.T * np.sqrt(pca.explained_variance_)
    biplot = go.Figure()
    biplot.add_trace(go.Scatter(
        x=pca_components[:, 0],
        y=pca_components[:, 1],
        mode='markers',
        marker=dict(size=10, color='blue'),
        text=player_data['Name'],
        hoverinfo='text',
        name='Players'
    ))
    for i, column in enumerate(player_data.drop(columns=['Name', 'PlayerID']).columns):
        biplot.add_trace(go.Scatter(
            x=[0, loadings[i, 0]],
            y=[0, loadings[i, 1]],
            mode='lines+text',
            line=dict(width=2, color='red'),
            text=[None, column],
            name=f'Loading: {column}',
            hoverinfo='none'
        ))
    biplot.update_layout(
        title='PCA Biplot',
        xaxis_title='Principal Component 1',
        yaxis_title='Principal Component 2',
        showlegend=True
    )
    biplot_table_data = pd.DataFrame(pca.components_, columns=player_data.drop(columns=['Name', 'PlayerID']).columns).to_dict('records')
    biplot_table_columns = [{'name': col, 'id': col} for col in player_data.drop(columns=['Name', 'PlayerID']).columns]

    num_clusters = max(2, min(5, len(pca_components)))
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    clusters = kmeans.fit_predict(pca_components[:, :2])
    player_data['Cluster'] = clusters

    numeric_data = player_data.select_dtypes(include=[np.number])
    numeric_data['Cluster'] = player_data['Cluster']
    cluster_stats = numeric_data.groupby('Cluster').mean()

    cluster_names = {}
    for cluster, stats in cluster_stats.iterrows():
        if stats[['Goals', 'Shots', 'Assists', 'Saves']].sum() < 0.5:
            cluster_names[cluster] = 'Spectator'
        elif stats['Goals'] > stats['Shots'] * 0.6 and stats['Shots'] > 1:
            cluster_names[cluster] = 'Striker'
        elif stats['Shots'] > stats['Assists'] and stats['Assists'] > stats['Saves']:
            cluster_names[cluster] = 'Attacking Midfield'
        elif stats['Assists'] > stats['Shots'] and stats['Assists'] > stats['Saves']:
            cluster_names[cluster] = 'Defensive Midfield'
        elif stats['Saves'] > stats['Goals'] and stats['Assists'] > stats['Shots'] and stats['Saves'] > 1:
            cluster_names[cluster] = 'Creative Defender'
        elif stats['Saves'] > stats['Goals'] and stats['Shots'] > stats['Assists'] and stats['Saves'] > 1:
            cluster_names[cluster] = 'Offensive Defender'
        elif stats['Saves'] > 1:
            cluster_names[cluster] = 'Defender'
        elif stats['Goals'] < stats['Shots'] * 0.7 and stats['Shots'] > 0.7:
            cluster_names[cluster] = 'Attacking Forward'
        else:
            cluster_names[cluster] = 'Spectator'

    player_data['Cluster Name'] = player_data['Cluster'].map(cluster_names)

    cluster_fig = px.scatter(
        x=pca_components[:, 0], y=pca_components[:, 1],
        color=player_data['Cluster Name'],
        hover_data={'Name': player_data['Name']},
        labels={'x': 'Principal Component 1', 'y': 'Principal Component 2', 'color': 'Play-Style'},
        title="Player Clusters Based on PCA"
    )
    cluster_table_data = player_data.to_dict('records')
    cluster_table_columns = [{'name': col, 'id': col} for col in player_data.columns]

    cluster_outcomes = {name: {'Win': 0, 'Loss': 0, 'Draw': 0} for name in cluster_names.values()}
    for match in all_player_data:
        match = match.merge(player_data[['PlayerID', 'Cluster Name']], on='PlayerID', how='left')
        for _, player in match.iterrows():
            result = 'Loss'
            if (player['Team0Score'] > player['Team1Score'] and player.get('Team', 0) == 0) or \
               (player['Team1Score'] > player['Team0Score'] and player.get('Team', 1) == 1):
                result = 'Win'
            elif player['Team0Score'] == player['Team1Score']:
                result = 'Draw'
            if pd.notna(player['Cluster Name']):
                cluster_outcomes[player['Cluster Name']][result] += 1

    cluster_outcomes_df = pd.DataFrame(cluster_outcomes).T.reset_index()
    cluster_outcomes_df.columns = ['Cluster Name', 'Win', 'Loss', 'Draw']

    win_ratio_fig = px.bar(
        cluster_outcomes_df.melt(id_vars='Cluster Name', var_name='Outcome', value_name='Count'),
        x='Cluster Name', y='Count', color='Outcome', title="Match Outcomes by Cluster", barmode='group'
    )
    win_ratio_table_data = cluster_outcomes_df.to_dict('records')
    win_ratio_table_columns = [{'name': col, 'id': col} for col in cluster_outcomes_df.columns]

    return (
        heatmap_fig, heatmap_table_data, heatmap_table_columns,
        scree_fig, scree_table_data, scree_table_columns,
        biplot, biplot_table_data, biplot_table_columns,
        cluster_fig, cluster_table_data, cluster_table_columns,
        win_ratio_fig, win_ratio_table_data, win_ratio_table_columns,
        {'display': 'block'}, {'display': 'block'}, {'display': 'block'}, {'display': 'block'}, {'display': 'block'}
    )

if __name__ == '__main__':
    app.run_server(debug=True)
