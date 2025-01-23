# Rocket League Player Cluster App

## Overview
The Rocket League Player Cluster App is a web application built using Python's Dash framework. It provides advanced analytics and visualization of player data from Rocket League matches. Users can upload JSON files containing match data and explore insights derived from techniques like PCA and clustering.

## Features
- **Data Upload**: Upload JSON files containing player statistics.
- **PCA Analysis**: Understand the variance in player metrics using PCA Scree Plots and Biplots.
- **Clustering**: Identify player roles (e.g., Offensive, Defensive) based on performance stats.
- **Match Outcome Analysis**: Examine the relationship between clusters and match results.
- **Data Tables**: View detailed tables of data used for each visualization.

## Installation

### Prerequisites
- Python 3.7+
- `pip` (Python package manager)

### Dependencies
Install the required Python libraries:

```bash
pip install dash pandas numpy scikit-learn plotly
```

## Usage

### Running the App
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/rocket-league-player-cluster-app.git
   cd rocket-league-player-cluster-app
   ```
2. Run the app:
   ```bash
   python app.py
   ```
3. Open your web browser and navigate to `http://127.0.0.1:8050/`.

### Uploading Data
- JSON files should include structured data for player stats and match outcomes. Example structure:
  ```json
  {
      "properties": {
          "PlayerStats": [
              {"Name": "Player1", "Score": 100, "Goals": 2, "Assists": 1, "Saves": 3, "Shots": 5, "Team": 0},
              {"Name": "Player2", "Score": 200, "Goals": 1, "Assists": 0, "Saves": 5, "Shots": 6, "Team": 1}
          ],
          "Team0Score": 3,
          "Team1Score": 2
      }
  }
  ```

### Features in Detail

#### Correlation Heatmap
- **What it shows**: Relationships between player metrics (e.g., Goals, Saves).
- **How to interpret**: Strong positive or negative correlations reveal which stats influence each other.

#### PCA Scree Plot
- **What it shows**: The variance explained by each principal component.
- **How to interpret**: Components with the highest variance capture the most information.

#### PCA Biplot
- **What it shows**: Player data projected into a reduced 2D space, with loading vectors indicating contributions of each stat.
- **How to interpret**: Clusters of points represent players with similar playstyles.

#### Player Clusters
- **What it shows**: Players grouped into offensive, balanced, or defensive clusters based on performance metrics.
- **How to interpret**: Helps identify player roles or unique playstyles.

#### Match Outcomes by Cluster
- **What it shows**: Win/Loss/Draw outcomes for each player cluster.
- **How to interpret**: Understand which roles contribute most to winning matches.

## Folder Structure
```
rocket-league-player-cluster-app/
├── app.py               # Main Dash app code
├── data/                # Example JSON files
├── assets/              # Custom CSS (if needed)
├── requirements.txt     # Python dependencies
└── README.md            # Project documentation
```

## Contributing
Contributions are welcome! Feel free to submit pull requests or open issues to suggest features or report bugs.

## License
This project is licensed under the MIT License. See the `LICENSE` file for details.

## Contact
For questions or feedback, reach out at [your-email@example.com].
