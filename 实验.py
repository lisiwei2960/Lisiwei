import subprocess

def run_experiment(command):
    """执行实验命令"""
    print(f"Running: {command}")
    result = subprocess.run(command, shell=True, text=True)
    if result.returncode != 0:
        print(f"Error occurred while running: {command}")
    else:
        print(f"Completed: {command}")

def main():
    # 实验 1: 普通 TimeXer
    # run_experiment('python run.py --task_name long_term_forecast --model_id ETTh1_6h --model TimeXer --data ETTh1 --root_path ./data/ETT/ --data_path ETTh1.csv --seq_len 96 --label_len 48 --pred_len 96 --train_epochs 50 --batch_size 32 --pred_len 6')
    # run_experiment('python run.py --task_name long_term_forecast --model_id ETTh1_12h --model TimeXer --data ETTh1 --root_path ./data/ETT/ --data_path ETTh1.csv --seq_len 96 --label_len 48 --pred_len 96 --train_epochs 50 --batch_size 32 --pred_len 12')
    # run_experiment('python run.py --task_name long_term_forecast --model_id ETTh1_24h --model TimeXer --data ETTh1 --root_path ./data/ETT/ --data_path ETTh1.csv --seq_len 96 --label_len 48 --pred_len 96 --train_epochs 50 --batch_size 32 --pred_len 24')

    # 实验 2: 只加 SE
    run_experiment('python run.py --task_name long_term_forecast --model_id ETTh1_SE_6h --model TimeXer --data ETTh1 --root_path ./data/ETT/ --data_path ETTh1.csv --seq_len 96 --label_len 48 --pred_len 96 --train_epochs 50 --batch_size 32 --use_se --pred_len 6')
    run_experiment('python run.py --task_name long_term_forecast --model_id ETTh1_SE_12h --model TimeXer --data ETTh1 --root_path ./data/ETT/ --data_path ETTh1.csv --seq_len 96 --label_len 48 --pred_len 96 --train_epochs 50 --batch_size 32 --use_se --pred_len 12')
    run_experiment('python run.py --task_name long_term_forecast --model_id ETTh1_SE_24h --model TimeXer --data ETTh1 --root_path ./data/ETT/ --data_path ETTh1.csv --seq_len 96 --label_len 48 --pred_len 96 --train_epochs 50 --batch_size 32 --use_se --pred_len 24')

    # 实验 3: 只加混合卷积-注意力
    run_experiment('python run.py --task_name long_term_forecast --model_id ETTh1_Hybrid_6h --model TimeXer --data ETTh1 --root_path ./data/ETT/ --data_path ETTh1.csv --seq_len 96 --label_len 48 --pred_len 96 --train_epochs 50 --batch_size 32 --use_hybrid --pred_len 6')
    run_experiment('python run.py --task_name long_term_forecast --model_id ETTh1_Hybrid_12h --model TimeXer --data ETTh1 --root_path ./data/ETT/ --data_path ETTh1.csv --seq_len 96 --label_len 48 --pred_len 96 --train_epochs 50 --batch_size 32 --use_hybrid --pred_len 12')
    run_experiment('python run.py --task_name long_term_forecast --model_id ETTh1_Hybrid_24h --model TimeXer --data ETTh1 --root_path ./data/ETT/ --data_path ETTh1.csv --seq_len 96 --label_len 48 --pred_len 96 --train_epochs 50 --batch_size 32 --use_hybrid --pred_len 24')

    # 实验 4: 同时加 SE + 混合卷积-注意力
    run_experiment('python run.py --task_name long_term_forecast --model_id ETTh1_SE_Hybrid_6h --model TimeXer --data ETTh1 --root_path ./data/ETT/ --data_path ETTh1.csv --seq_len 96 --label_len 48 --pred_len 96 --train_epochs 50 --batch_size 32 --use_se --use_hybrid --pred_len 6')
    run_experiment('python run.py --task_name long_term_forecast --model_id ETTh1_SE_Hybrid_12h --model TimeXer --data ETTh1 --root_path ./data/ETT/ --data_path ETTh1.csv --seq_len 96 --label_len 48 --pred_len 96 --train_epochs 50 --batch_size 32 --use_se --use_hybrid --pred_len 12')
    run_experiment('python run.py --task_name long_term_forecast --model_id ETTh1_SE_Hybrid_24h --model TimeXer --data ETTh1 --root_path ./data/ETT/ --data_path ETTh1.csv --seq_len 96 --label_len 48 --pred_len 96 --train_epochs 50 --batch_size 32 --use_se --use_hybrid --pred_len 24')

if __name__ == "__main__":
    main()
