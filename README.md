# F1 Race Predictor

A machine learning project that predicts the final top 10 finishers of Formula 1 races using historical championship data from 1950-2020.

## Project Overview

This project implements and compares two different machine learning approaches to forecast F1 race outcomes:
- **Random Forest** - An ensemble method that combines multiple decision trees for robust predictions
- **Neural Network** - A deep learning model that captures complex non-linear patterns in race data

The models analyze driver performance, constructor (team) strength, historical statistics, and circuit-specific characteristics to predict which drivers will finish in the top 10 positions.

## Problem Statement

Formula 1 race outcomes depend on multiple factors including driver skill, team performance, vehicle reliability, weather, and track characteristics. This project explores whether historical data can effectively predict future race outcomes, and compares the effectiveness of traditional machine learning (Random Forest) versus deep learning (Neural Network) approaches.

## Dataset

The project uses comprehensive Formula 1 historical data including:
- 70+ years of race results (1950-2020)
- Driver performance metrics and statistics
- Constructor/team information and performance
- Qualifying results and grid positions
- Pit stop data and race timings
- 13 CSV files with ~500K+ data points total

**Data Loading Strategy:**
- Models first attempt to load CSV files from local `Dataset/` folder (fast, ~1-2 seconds)
- If local files are not found, automatically falls back to downloading from Kaggle API
- This hybrid approach ensures compatibility whether data is pre-downloaded or fetched on-demand

## Setup & Installation

### Step 1: Create Virtual Environment
```bash
# Create virtual environment
python -m venv venv

# Activate on macOS/Linux
source venv/bin/activate

# Activate on Windows
venv\Scripts\activate
```

### Step 2: Install Dependencies
```bash
# Upgrade pip
pip install --upgrade pip

# Install required packages
pip install -r requirements.txt
```

### Step 3: Prepare Data (Optional)
- Download the F1 dataset from [Kaggle](https://www.kaggle.com/datasets/rohanrao/formula-1-world-championship-1950-2020)
- Extract all CSV files into the `Dataset/` folder
- The models will automatically load from there

## Usage

### Running the Random Forest Model
```bash
python F1_RandomForrest.py
```
- Trains on 80% of historical data
- Evaluates on 20% test set
- Outputs accuracy, precision, recall, and F1 scores
- Prompts for race ID to make predictions

### Running the Neural Network Model
```bash
python F1_NeuralNetwork.py
```
- Similar pipeline to Random Forest
- Includes training curves and loss plots
- Provides confidence scores for predictions
- Displays model architecture summary

### Making Predictions
After training, both models will ask you to enter a race ID to predict. The models will:
1. Load pre-race data for that specific race
2. Engineer features for all drivers in the race
3. Generate top 10 predictions with confidence scores
4. Compare predictions to actual results

### Deactivate Virtual Environment
```bash
deactivate
```

## Methodology

Both models follow the same pipeline:
1. **Data Loading & Preprocessing** - Clean and normalize historical race data from local CSV files
2. **Feature Engineering** - Extract meaningful features like average finishing position, pole position rate, DNF rate, etc.
3. **Train/Test Split** - 80/20 split with temporal ordering to prevent data leakage
4. **Model Training** - Train on historical races
5. **Evaluation** - Test on held-out race data with accuracy and classification metrics
6. **Prediction** - Predict top 10 finishers for a specified race using only pre-race information

## Models

### Random Forest (`F1_RandomForrest.py`)
- **Approach**: Ensemble learning that combines multiple decision trees
- **Advantages**: 
  - Robust to outliers and overfitting
  - Provides feature importance rankings
  - Fast prediction time
  - Easy to interpret
- **Implementation**: Scikit-learn RandomForestClassifier with optimized hyperparameters

### Neural Network (`F1_NeuralNetwork.py`)
- **Approach**: Multi-layer perceptron with hidden layers and regularization
- **Architecture**:
  - Input layer with feature normalization
  - Multiple hidden layers with ReLU activation
  - Dropout layers for regularization
  - Output layer with softmax for classification
- **Advantages**:
  - Captures complex non-linear relationships
  - Can learn intricate patterns in data
  - Flexible architecture for optimization
- **Implementation**: PyTorch with early stopping to prevent overfitting
- **Device Support**: Automatically detects and uses GPU if available (faster training)

## Features Used

The models extract meaningful features from the raw data:
- **Driver Statistics**: Average finishing position, pole position rate, DNF (Did Not Finish) rate, points per race
- **Recent Performance**: Performance in last 5-10 races to capture current form
- **Constructor Strength**: Team's average finishing position and consistency
- **Circuit-Specific Data**: Historical performance at each circuit
- **Grid Position**: Qualifying results as a strong predictor of race outcome
- **Experience**: Total races completed, years in F1

## Results & Performance

Both models achieve strong predictive accuracy on unseen test data. The comparison reveals:
- **Random Forest**: Fast, interpretable, excellent for feature importance analysis
- **Neural Network**: Potentially higher accuracy on complex patterns, requires more tuning

Performance metrics evaluated:
- **Accuracy**: Percentage of drivers correctly predicted in top 10
- **Precision**: Of predicted top 10, how many actually finished top 10
- **Recall**: Of actual top 10 finishers, how many were predicted
- **F1 Score**: Harmonic mean of precision and recall

## Technical Details

### Data Loading
- Loads CSV files directly from `Dataset/` folder (fast, ~1-2 seconds)
- Handles missing values and outliers
- Normalizes numerical features (0-1 scaling)
- Removes races with insufficient historical data
- Temporal ordering prevents future data leakage

### Train/Test Split Strategy
- **Temporal split**: Training on races before a cutoff date, testing on races after
- **80/20 ratio**: Maximizes training data while ensuring robust test evaluation
- **Prevents leakage**: Only uses data available before each race prediction

### Hyperparameters
- **Random Forest**: 100 trees, max depth tuning, min samples per leaf
- **Neural Network**: Layer sizes (128-64-32), dropout rate (0.3), learning rate (0.001), batch size (32)

## Dataset Details

Downloaded from Kaggle's F1 World Championship dataset (https://www.kaggle.com/datasets/rohanrao/formula-1-world-championship-1950-2020):
- **Total Records**: 500,000+ data points
- **Time Period**: 1950-2020 (71 seasons)
- **Races**: 1,076 races
- **Drivers**: 850+ unique drivers
- **Teams**: 210+ unique constructors

### CSV Files Included (place in `Dataset/` folder)
- `races.csv` - Race date, location, and details
- `results.csv` - Race results and finishing positions
- `drivers.csv` - Driver information and nationality
- `constructors.csv` - Team/constructor information
- `qualifying.csv` - Qualifying session results
- `driver_standings.csv` - Championship standings
- `constructor_standings.csv` - Team championship standings
- `lap_times.csv` - Individual lap timing data
- `pit_stops.csv` - Pit stop information
- `seasons.csv` - Season-level information
- `circuits.csv` - Track information
- `status.csv` - Race status codes (finished, DNF, etc.)
- `sprint_results.csv` - Sprint race results (newer seasons)

## Limitations & Future Work

### Current Limitations
- Does not account for weather conditions
- No real-time data on vehicle performance/reliability
- Historical bias from older era racing
- Limited ability to predict accidents/unpredictable events

### Future Improvements
- Add weather data integration
- Include real-time telemetry data
- Implement SHAP values for better model explainability
- Add more recent season data (2021+)
- Ensemble predictions from both models