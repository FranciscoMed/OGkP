# Optimal Goalkeeper Positioning (OGkP) App

This application calculates the **Optimal Goalkeeper Positioning (OGkP)** using data from the StatsBomb API and visualizes the results in an interactive dashboard built with [Streamlit](https://streamlit.io). The app analyses shot data to evaluate how close a goalkeeper's position is to the computed optimal position based on shot geometry and presents the top 10 shot analyses in an intuitive layout.

## Features

- **Dynamic Competition & Team Selection:** Choose from available competitions and teams using data fetched directly from the StatsBomb API.
- **Goalkeeper Filtering:** Select a goalkeeper from the team lineups. The app extracts goalkeeper names from the match lineups.
- **Shot Data Analysis:** Fetch shot events for the selected team and filter by goalkeeper to calculate the OPM.
- **Interactive Visualizations:** Visualize each shot's data, including shot cones, optimal goalkeeper positioning, and dive circles, using Matplotlib and mplsoccer.
- **Dashboard Layout:** Displays the top 10 shot analyses in a 2-column layout for easy comparison.
- **Real-Time Metrics:** Calculates and highlights the season average of the Optimal Positioning Metric.

## Requirements

- Python 3.7+
- [Streamlit](https://streamlit.io)
- [NumPy](https://numpy.org)
- [Pandas](https://pandas.pydata.org)
- [Matplotlib](https://matplotlib.org)
- [mplsoccer](https://mplsoccer.readthedocs.io)
- [statsbombpy](https://github.com/statsbomb/statsbombpy)

## Installation

1. **Clone the repository:**

   ```bash
   git clone https://github.com/yourusername/opm-app.git
   cd opm-app
