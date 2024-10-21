import os
from datetime import datetime

def log_message(message):
    if not os.path.exists('logs'):
        os.makedirs('logs')
    with open(f"logs/log_{datetime.now().strftime('%Y%m%d')}.txt", 'a') as f:
        f.write(message + '\n')