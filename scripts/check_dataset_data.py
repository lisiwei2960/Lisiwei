import os
import sqlite3
import pandas as pd

# 数据库路径
DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'power_forecast.db')

conn = sqlite3.connect(DB_PATH)
cursor = conn.cursor()

cursor.execute('SELECT id, filename, data FROM dataset')
rows = cursor.fetchall()

print('检查所有数据集的data字段...')
for row in rows:
    dataset_id, filename, data = row
    if not data:
        print(f'数据集ID={dataset_id}, 文件名={filename}，data字段为空')
        continue
    try:
        df = pd.read_json(data)
    except Exception as e:
        print(f'数据集ID={dataset_id}, 文件名={filename}，data字段解析失败: {e}')

print('检查完成。')
conn.close() 