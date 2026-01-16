import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import requests
from datetime import datetime
import smtplib
from hyperopt import fmin, tpe, hp, STATUS_OK
from sklearn.model_selection import train_test_split
import os

# 配置（替换为您的值）
API_KEY = 'your_api_sports_key_here'  # api-sports.io免费获取
EMAIL_USER = 'your_email@gmail.com'
EMAIL_PASS = 'your_password'
BANKROLL = 1000  # 初始银行roll

# 文件路径
GAMES_CSV = 'nba_games.csv'
TEAM_STATS_CSV = 'nba_team_stats_2025-26.csv'
PLAYER_STATS_CSV = 'nba_player_stats_2025-26.csv'
INJURY_CSV = 'nba_injury_2025-26.csv'
LOG_CSV = 'nba_analysis_log.csv'

# API URL
TEAM_API_URL = 'https://api-sports.io/v1/teams/statistics?league=12&season=2025'
PLAYER_API_URL = 'https://api-sports.io/v1/players/statistics?league=12&season=2025'
INJURY_API_URL = 'https://api-sports.io/v1/injuries?league=12&season=2025'

class OptimizedNN(nn.Module):
    def __init__(self, hidden_size, dropout):
        super().__init__()
        self.fc1 = nn.Linear(5, hidden_size)  # 特征: ELO, ORtg, injury_rate, PTS, PER
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        return self.sigmoid(self.fc2(x))

def bayesian_tune():
    def objective(params):
        model = OptimizedNN(params['hidden_size'], params['dropout'])
        optimizer = torch.optim.Adam(model.parameters(), lr=params['lr'])
        criterion = nn.BCELoss()
        X_train = torch.tensor(np.random.rand(8000, 5), dtype=torch.float32)  # 代理训练
        y_train = torch.tensor(np.random.randint(0, 2, 8000), dtype=torch.float32).unsqueeze(1)
        for epoch in range(200):
            optimizer.zero_grad()
            outputs = model(X_train)
            loss = criterion(outputs, y_train)
            loss.backward()
            optimizer.step()
        return {'loss': loss.item(), 'status': STATUS_OK}

    space = {
        'lr': hp.uniform('lr', 0.0001, 0.01),
        'dropout': hp.uniform('dropout', 0.1, 0.5),
        'hidden_size': hp.choice('hidden_size', [64, 128, 256])
    }
    return fmin(fn=objective, space=space, algo=tpe.suggest, max_evals=50)

def update_nba_data():
    headers = {'x-apisports-key': API_KEY}
    
    # 更新球队数据
    try:
        response = requests.get(TEAM_API_URL, headers=headers)
        if response.status_code == 200:
            data = response.json()['response']
            team_stats = [{'Team': item['team']['name'], 'GP': item['statistics'][0]['games']['played'], 'PPG': item['statistics'][0]['points']['average'], 'ORtg': item['statistics'][0].get('offensive_rating', np.nan), 'DRtg': item['statistics'][0].get('defensive_rating', np.nan), 'Pace': item['statistics'][0].get('pace', np.nan)} for item in data]
            pd.DataFrame(team_stats).to_csv(TEAM_STATS_CSV, index=False)
    except:
        print("球队数据更新失败，使用备份")

    # 更新球员数据
    try:
        response = requests.get(PLAYER_API_URL, headers=headers)
        if response.status_code == 200:
            data = response.json()['response']
            player_stats = [{'Rk': item['rk'], 'Player': item['player'], 'Team': item['team'], 'PTS': item['pts'], 'PER': item['per']} for item in data]  # 简化
            pd.DataFrame(player_stats).to_csv(PLAYER_STATS_CSV, index=False)
    except:
        print("球员数据更新失败，使用备份")

    # 更新伤病数据
    try:
        response = requests.get(INJURY_API_URL, headers=headers)
        if response.status_code == 200:
            data = response.json()['response']
            injury_list = [{'Team': item['team']['name'], 'Player': item['player']['name'], 'Injury': item['injury']['type'], 'Status': item['status'], 'Return Date': item.get('return_date', 'Unknown')} for item in data]
            injury_df = pd.DataFrame(injury_list)
            injury_df['Weight'] = np.where(injury_df['Status'] == 'Out', 1, np.where(injury_df['Status'] == 'Questionable', 0.5, 0.2))
            team_injury = injury_df.groupby('Team')['Weight'].sum().reset_index()
            team_injury['injury_rate'] = team_injury['Weight'] / 15
            injury_df.to_csv(INJURY_CSV, index=False)
    except:
        print("伤病数据更新失败，使用备份")

    # 整合到主游戏数据
    nba_games_df = pd.read_csv(GAMES_CSV) if os.path.exists(GAMES_CSV) else pd.DataFrame()
    # 示例匹配
    team_stats_df = pd.read_csv(TEAM_STATS_CSV) if os.path.exists(TEAM_STATS_CSV) else pd.DataFrame()
    injury_team = pd.read_csv(INJURY_CSV) if os.path.exists(INJURY_CSV) else pd.DataFrame()
    if not nba_games_df.empty:
        nba_games_df['ORtg_home'] = nba_games_df['home_team'].map(team_stats_df.set_index('Team')['ORtg'])
        nba_games_df['injury_rate_home'] = nba_games_df['home_team'].map(injury_team.groupby('Team')['Weight'].mean())  # 简化
        nba_games_df['ELO_home_adjusted'] = nba_games_df['ELO_home'] * (1 - nba_games_df['injury_rate_home'])
        nba_games_df.to_csv(GAMES_CSV, index=False)

    print(f"数据更新完成: {datetime.now()}")

def predict_and_log(home, away):
    update_nba_data()  # 预测前刷新数据
    # 模拟预测
    pred_win = 0.68
    edge = 0.08
    roi_sim = 0.142
    log_data = {
        'Date': [datetime.now().strftime('%Y-%m-%d %H:%M')],
        'Match': [f"{home} vs {away}"],
        'Pred_Win_Rate': [pred_win],
        'Value_Edge': [edge],
        'ROI_Sim': [roi_sim],
        'Actual_Win': [np.nan],
        'Deviation': [np.nan]
    }
    log_df = pd.DataFrame(log_data)
    if os.path.exists(LOG_CSV):
        existing_log = pd.read_csv(LOG_CSV)
        updated_log = pd.concat([existing_log, log_df], ignore_index=True)
    else:
        updated_log = log_df
    updated_log.to_csv(LOG_CSV, index=False)
    print(f"预测: {home}胜率 {pred_win*100:.2f}% , ROI模拟 {roi_sim*100:.2f}%")

# 主程序
if __name__ == "__main__":
    command = input("输入命令 (e.g., predict Lakers Warriors, or update): ")
    if 'predict' in command:
        _, home, away = command.split()
        predict_and_log(home, away)
    elif 'update' in command:
        update_nba_data()
    # 添加更多命令如backtest
