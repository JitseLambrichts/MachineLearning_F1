"""
F1 Race Winner Predictor using Neural Network
Dataset: Formula 1 World Championship (1950-2020) from Kaggle
+ Nieuwe race voorspellingen
"""

import kagglehub
from kagglehub import KaggleDatasetAdapter
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import warnings
import pickle
warnings.filterwarnings('ignore')

# Check voor GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print("=" * 60)
print("F1 RACE WINNER PREDICTOR - NEURAL NETWORK MODEL")
print("=" * 60)
print(f"🖥️  Using device: {device}")

# Laad alle benodigde datasets
print("\n📥 Loading datasets from Kaggle...")

def load_dataset(file_path):
    """Helper functie om dataset te laden"""
    try:
        return pd.read_csv(f"Dataset/{file_path}")
    except:
        return kagglehub.load_dataset(
            KaggleDatasetAdapter.PANDAS,
            "rohanrao/formula-1-world-championship-1950-2020",
            file_path
        )

# Laad alle datasets
races = load_dataset("races.csv")
results = load_dataset("results.csv")
drivers = load_dataset("drivers.csv")
constructors = load_dataset("constructors.csv")
qualifying = load_dataset("qualifying.csv")
driver_standings = load_dataset("driver_standings.csv")
constructor_standings = load_dataset("constructor_standings.csv")

print("✅ Datasets loaded!")
print(f"   - Races: {len(races)} records")
print(f"   - Results: {len(results)} records")

# Feature Engineering
print("\n🔧 Feature Engineering...")

def create_features(races_df, results_df, qualifying_df, driver_standings_df, constructor_standings_df):
    """Creëer features voor het ML model"""
    races_df = races_df.sort_values('date').reset_index(drop=True)
    feature_list = []
    
    for _, race in races_df.iterrows():
        race_id = race['raceId']
        circuit_id = race['circuitId']  # ← Circuit ID voor circuit-specifieke stats
        race_results = results_df[results_df['raceId'] == race_id].copy()
        race_qualifying = qualifying_df[qualifying_df['raceId'] == race_id].copy()
        
        if len(race_results) == 0:
            continue
        
        for _, result in race_results.iterrows():
            driver_id = result['driverId']
            constructor_id = result['constructorId']
            
            previous_races = races_df[races_df['raceId'] < race_id]
            previous_results = results_df[results_df['raceId'].isin(previous_races['raceId'])]
            
            # ========== ALGEMENE STATISTIEKEN ==========
            driver_history = previous_results[previous_results['driverId'] == driver_id]
            driver_wins = len(driver_history[driver_history['positionOrder'] == 1])
            driver_podiums = len(driver_history[driver_history['positionOrder'] <= 3])
            driver_races = len(driver_history)
            driver_dnf = len(driver_history[driver_history['positionOrder'].isna()])
            
            finished_races = driver_history[driver_history['positionOrder'].notna()]
            driver_avg_pos = finished_races['positionOrder'].mean() if len(finished_races) > 0 else 20
            driver_points = driver_history['points'].sum()
            
            # ========== CIRCUIT-SPECIFIEKE STATISTIEKEN ==========
            # Haal races op dit specifieke circuit op (voor deze race)
            circuit_races = previous_races[previous_races['circuitId'] == circuit_id]
            circuit_results = results_df[results_df['raceId'].isin(circuit_races['raceId'])]
            
            # Driver stats op dit circuit
            driver_circuit_history = circuit_results[circuit_results['driverId'] == driver_id]
            driver_circuit_wins = len(driver_circuit_history[driver_circuit_history['positionOrder'] == 1])
            driver_circuit_podiums = len(driver_circuit_history[driver_circuit_history['positionOrder'] <= 3])
            driver_circuit_races = len(driver_circuit_history)
            
            circuit_finished = driver_circuit_history[driver_circuit_history['positionOrder'].notna()]
            driver_circuit_avg_pos = circuit_finished['positionOrder'].mean() if len(circuit_finished) > 0 else 20
            
            # Constructor stats op dit circuit
            constructor_circuit_history = circuit_results[circuit_results['constructorId'] == constructor_id]
            constructor_circuit_wins = len(constructor_circuit_history[constructor_circuit_history['positionOrder'] == 1])
            constructor_circuit_podiums = len(constructor_circuit_history[constructor_circuit_history['positionOrder'] <= 3])
            
            recent_races = driver_history.tail(5)
            recent_avg_pos = recent_races['positionOrder'].mean() if len(recent_races) > 0 else 20
            recent_wins = len(recent_races[recent_races['positionOrder'] == 1])
            
            constructor_history = previous_results[previous_results['constructorId'] == constructor_id]
            constructor_wins = len(constructor_history[constructor_history['positionOrder'] == 1])
            constructor_podiums = len(constructor_history[constructor_history['positionOrder'] <= 3])
            constructor_points = constructor_history['points'].sum()
            
            qual = race_qualifying[race_qualifying['driverId'] == driver_id]
            qualifying_pos = qual['position'].values[0] if len(qual) > 0 and pd.notna(qual['position'].values[0]) else 20
            
            prev_standing = driver_standings_df[
                (driver_standings_df['driverId'] == driver_id) & 
                (driver_standings_df['raceId'] < race_id)
            ].sort_values('raceId', ascending=False)
            
            championship_pos = prev_standing['position'].values[0] if len(prev_standing) > 0 else 20
            championship_points = prev_standing['points'].values[0] if len(prev_standing) > 0 else 0
            
            prev_const_standing = constructor_standings_df[
                (constructor_standings_df['constructorId'] == constructor_id) & 
                (constructor_standings_df['raceId'] < race_id)
            ].sort_values('raceId', ascending=False)
            
            constructor_champ_pos = prev_const_standing['position'].values[0] if len(prev_const_standing) > 0 else 10
            driver_win_rate = (driver_wins / driver_races * 100) if driver_races > 0 else 0
            driver_podium_rate = (driver_podiums / driver_races * 100) if driver_races > 0 else 0
            
            # Circuit-specifieke win rate
            driver_circuit_win_rate = (driver_circuit_wins / driver_circuit_races * 100) if driver_circuit_races > 0 else 0
            
            won_race = 1 if result['positionOrder'] == 1 else 0
            
            features = {
                'qualifying_position': qualifying_pos,
                # Algemene statistieken
                'driver_wins': driver_wins,
                'driver_podiums': driver_podiums,
                'driver_total_races': driver_races,
                'driver_avg_position': driver_avg_pos,
                'driver_total_points': driver_points,
                'driver_win_rate': driver_win_rate,
                'driver_podium_rate': driver_podium_rate,
                'driver_dnf_count': driver_dnf,
                'recent_avg_position': recent_avg_pos,
                'recent_wins': recent_wins,
                # Circuit-specifieke statistieken
                'driver_circuit_wins': driver_circuit_wins,
                'driver_circuit_podiums': driver_circuit_podiums,
                'driver_circuit_races': driver_circuit_races,
                'driver_circuit_avg_pos': driver_circuit_avg_pos,
                'driver_circuit_win_rate': driver_circuit_win_rate,
                'constructor_circuit_wins': constructor_circuit_wins,
                'constructor_circuit_podiums': constructor_circuit_podiums,
                # Constructor algemeen
                'constructor_wins': constructor_wins,
                'constructor_podiums': constructor_podiums,
                'constructor_points': constructor_points,
                # Championship standings
                'championship_position': championship_pos,
                'championship_points': championship_points,
                'constructor_championship_pos': constructor_champ_pos,
                # Target
                'won_race': won_race,
                'driver_id': driver_id,
                'race_id': race_id
            }
            
            feature_list.append(features)
    
    return pd.DataFrame(feature_list)

df_features = create_features(races, results, qualifying, driver_standings, constructor_standings)

print(f"✅ Features made: {len(df_features)} samples")

df_features = df_features.dropna()
print(f"   - After cleaning: {len(df_features)} samples")

# ============================================
# TIJDGEBASEERDE SPLIT
# ============================================
print("\n🕐 Timebased train/test split... (80% for training data)")

# Sorteer op race_id (chronologisch)
df_features = df_features.sort_values('race_id').reset_index(drop=True)

# Gebruik laatste 20% als test set (meest recente races)
split_idx = int(len(df_features) * 0.8)
train_data = df_features.iloc[:split_idx]
test_data = df_features.iloc[split_idx:]

# Bepaal cutoff race voor voorspelling
cutoff_race_id = train_data['race_id'].max()
prediction_race_id = cutoff_race_id + 1  # Voorspel de VOLGENDE race

print(f"   - Training until race_id: {cutoff_race_id}")
print(f"   - Voorspelling until race_id: {prediction_race_id}")

# Toon welke races NIET in training zitten
test_race_ids = test_data['race_id'].unique()
test_races_info = races[races['raceId'].isin(test_race_ids)].sort_values('date')

print(f"\n📋 Races NOT included in the trainingsdata ({len(test_races_info)} races):")
print("=" * 60)
for _, race in test_races_info.iterrows():
    print(f"   Race ID {race['raceId']:4d}: {race['name']:30s} - {race['date']}")
print("=" * 60)

# Split features en labels
X_train = train_data.drop(['won_race', 'driver_id', 'race_id'], axis=1)
y_train = train_data['won_race']
X_test = test_data.drop(['won_race', 'driver_id', 'race_id'], axis=1)
y_test = test_data['won_race']

print(f"\n📊 Dataset splits:")
print(f"   - Training set: {len(X_train)} samples")
print(f"   - Test set: {len(X_test)} samples")
print(f"   ⚠️  Test set only containins UPCOMING races!")

# Feature Scaling (belangrijk voor neural networks)
print("\n📏 Scaling features...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Converteer naar PyTorch tensors
X_train_tensor = torch.FloatTensor(X_train_scaled).to(device)
y_train_tensor = torch.FloatTensor(y_train.values).to(device)
X_test_tensor = torch.FloatTensor(X_test_scaled).to(device)
y_test_tensor = torch.FloatTensor(y_test.values).to(device)

# DataLoader
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

# Neural Network Model
class F1WinnerPredictor(nn.Module):
    def __init__(self, input_size):
        super(F1WinnerPredictor, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.network(x)

# Initialiseer model
print("\n🧠 Training Neural Network model...")
input_size = X_train.shape[1]
nn_model = F1WinnerPredictor(input_size).to(device)

# Loss en optimizer
# Gebruik weighted loss voor imbalanced data
pos_weight = torch.tensor([(len(y_train) - y_train.sum()) / y_train.sum()]).to(device)
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
optimizer = optim.Adam(nn_model.parameters(), lr=0.001, weight_decay=1e-5)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=10, factor=0.5)

# Training loop
num_epochs = 100
best_loss = float('inf')
patience_counter = 0
early_stop_patience = 20

# Pas model aan voor BCEWithLogitsLoss (verwijder sigmoid uit forward)
class F1WinnerPredictorV2(nn.Module):
    def __init__(self, input_size):
        super(F1WinnerPredictorV2, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(32, 1)
        )
    
    def forward(self, x):
        return self.network(x)

nn_model = F1WinnerPredictorV2(input_size).to(device)
optimizer = optim.Adam(nn_model.parameters(), lr=0.001, weight_decay=1e-5)

for epoch in range(num_epochs):
    nn_model.train()
    total_loss = 0
    
    for batch_X, batch_y in train_loader:
        optimizer.zero_grad()
        outputs = nn_model(batch_X).squeeze()
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    avg_loss = total_loss / len(train_loader)
    scheduler.step(avg_loss)
    
    # Early stopping
    if avg_loss < best_loss:
        best_loss = avg_loss
        patience_counter = 0
        # Save best model
        best_model_state = nn_model.state_dict().copy()
    else:
        patience_counter += 1
    
    if patience_counter >= early_stop_patience:
        print(f"   Early stopping at epoch {epoch+1}")
        break
    
    if (epoch + 1) % 20 == 0:
        print(f"   Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")

# Laad beste model
nn_model.load_state_dict(best_model_state)
print("✅ Model trained!")

# Evaluatie
nn_model.eval()
with torch.no_grad():
    y_pred_logits = nn_model(X_test_tensor).squeeze()
    y_pred_proba = torch.sigmoid(y_pred_logits)
    y_pred = (y_pred_proba > 0.5).float()
    accuracy = (y_pred == y_test_tensor).float().mean().item()

print("\n" + "=" * 60)
print("MODEL EVALUATIE")
print("=" * 60)
print(f"\n🎯 Accuracy: {accuracy * 100:.2f}%")

# Save model en scaler
torch.save(nn_model.state_dict(), 'f1_nn_model.pth')
with open('f1_scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)
with open('f1_feature_cols.pkl', 'wb') as f:
    pickle.dump(X_train.columns.tolist(), f)
print("\n💾 Model opgeslagen als 'f1_nn_model.pth'")
print("💾 Scaler opgeslagen als 'f1_scaler.pkl'")

# ============================================
# VOORSPEL NIEUWE RACE (zonder data leakage)
# ============================================

def predict_race_winner(model, scaler, new_race_data, feature_cols):
    """
    Voorspel winnaar voor nieuwe race met Neural Network
    
    Parameters:
    - model: getraind neural network model
    - scaler: fitted StandardScaler
    - new_race_data: DataFrame met driver features
    - feature_cols: lijst van feature kolommen
    
    Returns:
    - DataFrame met voorspellingen gesorteerd op kans
    """
    # Verwijder extra kolommen
    X_new = new_race_data[feature_cols]
    
    # Scale features
    X_new_scaled = scaler.transform(X_new)
    X_new_tensor = torch.FloatTensor(X_new_scaled).to(device)
    
    # Voorspel kansen
    model.eval()
    with torch.no_grad():
        logits = model(X_new_tensor).squeeze()
        win_probabilities = torch.sigmoid(logits).cpu().numpy()
    
    # Maak resultaat DataFrame
    predictions = new_race_data.copy()
    predictions['win_probability'] = win_probabilities * 100
    predictions = predictions.sort_values('win_probability', ascending=False)
    
    return predictions

print("\n" + "=" * 60)
print("🏁 PREDICT NEW RACE")
print("=" * 60)

# Toon beschikbare races (focus op 2023-2024)
available_races = races[races['raceId'] > cutoff_race_id].sort_values('date')
recent_races = available_races[available_races['year'] >= 2023]

print(f"\n📋 Available races for prediction (2023-2024 season):")
print("=" * 85)
if len(recent_races) > 0:
    print("🏁 2023-2024 SEASON (Most Recent Data):")
    print("-" * 85)
    for _, race in recent_races.iterrows():
        print(f"   ID {race['raceId']:4d}: {race['name']:40s} | {race['date']}")
    print("=" * 85)
else:
    print("   Geen 2023-2024 races beschikbaar na cutoff")
    print("\n📋 Oudere races beschikbaar:")
    print("=" * 85)
    for _, race in available_races.head(20).iterrows():
        print(f"   ID {race['raceId']:4d}: {race['name']:30s} - {race['date']}")
    print("=" * 85)

target_race_id = int(input("🏎️  Give the RACE ID for a prediction of that race: "))

# Valideer dat race NA cutoff ligt
if target_race_id <= cutoff_race_id:
    print(f"\n⚠️  WAARSCHUWING: Race {target_race_id} zit in trainingsdata!")
    print(f"   Dit veroorzaakt data leakage. Kies een race > {cutoff_race_id}")
    exit()

# Check of deze race bestaat
if target_race_id in races['raceId'].values:
    target_race = races[races['raceId'] == target_race_id].iloc[0]
    
    print(f"\n📍 Race: {target_race['name']} - {target_race['date']}")
    print(f"   Circuit: {target_race['circuitId']}")
    print(f"   ⚠️  This race was NOT included in trainging data!")
    
    # Haal stats op TOT (maar niet inclusief) deze race
    race_results = results[results['raceId'] == target_race_id]
    race_qualifying = qualifying[qualifying['raceId'] == target_race_id]
    
    # Bereken features ALLEEN met data van VOOR deze race
    new_race_features = []
    
    print("\n🏎️  Calculating driver's chances...")
    
    circuit_id = target_race['circuitId']  # Circuit ID voor circuit-specifieke stats
    
    for driver_id in race_results['driverId'].unique():
        driver_result = race_results[race_results['driverId'] == driver_id].iloc[0]
        constructor_id = driver_result['constructorId']
        
        # BELANGRIJ: Gebruik alleen data van VOOR deze race (< target_race_id)
        driver_history = results[
            (results['driverId'] == driver_id) & 
            (results['raceId'] < target_race_id)  # ← Strikte ongelijkheid!
        ]
        
        # ========== ALGEMENE STATISTIEKEN ==========
        driver_wins = len(driver_history[driver_history['positionOrder'] == 1])
        driver_podiums = len(driver_history[driver_history['positionOrder'] <= 3])
        driver_races = len(driver_history)
        
        finished = driver_history[driver_history['positionOrder'].notna()]
        driver_avg_pos = finished['positionOrder'].mean() if len(finished) > 0 else 20
        driver_points = driver_history['points'].sum()
        
        recent = driver_history.tail(5)
        recent_avg_pos = recent['positionOrder'].mean() if len(recent) > 0 else 20
        recent_wins = len(recent[recent['positionOrder'] == 1])
        
        constructor_history = results[
            (results['constructorId'] == constructor_id) & 
            (results['raceId'] < target_race_id)  # ← Strikte ongelijkheid!
        ]
        constructor_wins = len(constructor_history[constructor_history['positionOrder'] == 1])
        constructor_podiums = len(constructor_history[constructor_history['positionOrder'] <= 3])
        constructor_points = constructor_history['points'].sum()
        
        # ========== CIRCUIT-SPECIFIEKE STATISTIEKEN ==========
        # Haal races op dit circuit op (voor deze race)
        circuit_races = races[
            (races['circuitId'] == circuit_id) & 
            (races['raceId'] < target_race_id)
        ]
        circuit_results = results[results['raceId'].isin(circuit_races['raceId'])]
        
        # Driver stats op dit circuit
        driver_circuit_history = circuit_results[circuit_results['driverId'] == driver_id]
        driver_circuit_wins = len(driver_circuit_history[driver_circuit_history['positionOrder'] == 1])
        driver_circuit_podiums = len(driver_circuit_history[driver_circuit_history['positionOrder'] <= 3])
        driver_circuit_races = len(driver_circuit_history)
        
        circuit_finished = driver_circuit_history[driver_circuit_history['positionOrder'].notna()]
        driver_circuit_avg_pos = circuit_finished['positionOrder'].mean() if len(circuit_finished) > 0 else 20
        
        # Constructor stats op dit circuit
        constructor_circuit_history = circuit_results[circuit_results['constructorId'] == constructor_id]
        constructor_circuit_wins = len(constructor_circuit_history[constructor_circuit_history['positionOrder'] == 1])
        constructor_circuit_podiums = len(constructor_circuit_history[constructor_circuit_history['positionOrder'] <= 3])
        
        qual_pos = race_qualifying[race_qualifying['driverId'] == driver_id]['position'].values
        qualifying_pos = qual_pos[0] if len(qual_pos) > 0 else 15
        
        # Standings van VOOR deze race
        standing = driver_standings[
            (driver_standings['driverId'] == driver_id) & 
            (driver_standings['raceId'] < target_race_id)
        ].sort_values('raceId', ascending=False)
        
        championship_pos = standing['position'].values[0] if len(standing) > 0 else 20
        championship_points = standing['points'].values[0] if len(standing) > 0 else 0
        
        const_standing = constructor_standings[
            (constructor_standings['constructorId'] == constructor_id) & 
            (constructor_standings['raceId'] < target_race_id)
        ].sort_values('raceId', ascending=False)
        
        constructor_champ_pos = const_standing['position'].values[0] if len(const_standing) > 0 else 10
        
        driver_win_rate = (driver_wins / driver_races * 100) if driver_races > 0 else 0
        driver_podium_rate = (driver_podiums / driver_races * 100) if driver_races > 0 else 0
        driver_dnf = len(driver_history[driver_history['positionOrder'].isna()])
        
        # Circuit-specifieke win rate
        driver_circuit_win_rate = (driver_circuit_wins / driver_circuit_races * 100) if driver_circuit_races > 0 else 0
        
        # Driver info
        driver_info = drivers[drivers['driverId'] == driver_id].iloc[0]
        constructor_info = constructors[constructors['constructorId'] == constructor_id].iloc[0]
        
        new_race_features.append({
            'driver_name': f"{driver_info['forename']} {driver_info['surname']}",
            'constructor': constructor_info['name'],
            'qualifying_position': qualifying_pos,
            # Algemene statistieken
            'driver_wins': driver_wins,
            'driver_podiums': driver_podiums,
            'driver_total_races': driver_races,
            'driver_avg_position': driver_avg_pos,
            'driver_total_points': driver_points,
            'driver_win_rate': driver_win_rate,
            'driver_podium_rate': driver_podium_rate,
            'driver_dnf_count': driver_dnf,
            'recent_avg_position': recent_avg_pos,
            'recent_wins': recent_wins,
            # Circuit-specifieke statistieken
            'driver_circuit_wins': driver_circuit_wins,
            'driver_circuit_podiums': driver_circuit_podiums,
            'driver_circuit_races': driver_circuit_races,
            'driver_circuit_avg_pos': driver_circuit_avg_pos,
            'driver_circuit_win_rate': driver_circuit_win_rate,
            'constructor_circuit_wins': constructor_circuit_wins,
            'constructor_circuit_podiums': constructor_circuit_podiums,
            # Constructor algemeen
            'constructor_wins': constructor_wins,
            'constructor_podiums': constructor_podiums,
            'constructor_points': constructor_points,
            # Championship standings
            'championship_position': championship_pos,
            'championship_points': championship_points,
            'constructor_championship_pos': constructor_champ_pos
        })
    
    new_race_df = pd.DataFrame(new_race_features)
    
    # Voorspel met Neural Network!
    feature_cols = X_train.columns.tolist()
    predictions = predict_race_winner(nn_model, scaler, new_race_df, feature_cols)
    
    print("\n" + "=" * 60)
    print("🏆 RACE PREDICTION - TOP 10")
    print("=" * 60)
    
    # Toon ook de werkelijke winnaar als beschikbaar
    actual_winner = results[
        (results['raceId'] == target_race_id) & 
        (results['positionOrder'] == 1)
    ]
    
    if len(actual_winner) > 0:
        winner_id = actual_winner.iloc[0]['driverId']
        winner_info = drivers[drivers['driverId'] == winner_id].iloc[0]
        print(f"\n✅ Werkelijke winnaar: {winner_info['forename']} {winner_info['surname']}")
        print("=" * 60)
    
    for idx, (_, row) in enumerate(predictions.head(10).iterrows(), 1):
        marker = "🏆" if len(actual_winner) > 0 and winner_info['surname'] in row['driver_name'] else "  "
        print(f"\n{marker} {idx}. {row['driver_name']:25s} ({row['constructor']})")
        print(f"   Win Probability: {row['win_probability']:.2f}%")
        print(f"   Qualifying: P{int(row['qualifying_position'])} | Championship: P{int(row['championship_position'])}")
        print(f"   Career: {int(row['driver_wins'])} wins, {int(row['driver_podiums'])} podiums")
        print(f"   🏁 This circuit: {int(row['driver_circuit_wins'])} wins, {int(row['driver_circuit_podiums'])} podiums in {int(row['driver_circuit_races'])} races (avg pos: {row['driver_circuit_avg_pos']:.1f})")
else:
    print(f"\n⚠️  Race {target_race_id} niet gevonden in dataset")
    print("   Probeer een andere race_id of train op meer recente data")