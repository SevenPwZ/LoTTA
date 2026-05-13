import torch
import numpy as np
import itertools
from collections import defaultdict
from Model.Trainer import Trainer

base_config = {

    'epochs': 200,
    'sche_gamma': 0.98,
    'mask_num': 15,
    'gamma': 1,
    'K': 3,

    'device': 'cuda:6',
    'data_dir': 'Data/',
    'runs': 1,
    'num_workers': 0,
    'batch_size': 512,
    'en_nlayers': 3,
    'de_nlayers': 3,
    'hidden_dim': 256,
    'z_dim': 128,
    'mask_nlayers': 3,

}
dataset_fixed_params = {
    'breastw': {
        'data_dim': 9,
        'learning_rate': [0.01],
        'lambda': [10],
        'normal_tuning_epochs': [10],
        'abnormal_tuning_epochs': [10],
        'r': [64],
        'random_seed': [123, 3407, 3507]
    }
}

log_file = "output/results_log.txt"

if __name__ == "__main__":
    all_dataset_results = defaultdict(list)
    all_metrics = []

    datasets_to_run = ['breastw']
    for dataset_name in datasets_to_run:
        print(f"\n=== Running experiments for dataset: {dataset_name} ===")
        results = []
        count = 0
        if dataset_name in dataset_fixed_params:
            fixed = dataset_fixed_params[dataset_name]
            base_config['data_dim'] = fixed['data_dim']
            param_grid = {
                'learning_rate': fixed['learning_rate'],
                'lambda': fixed['lambda'],
                'normal_tuning_epochs': fixed['normal_tuning_epochs'],
                'abnormal_tuning_epochs': fixed['abnormal_tuning_epochs'],
                'r': fixed['r'],
                'random_seed': fixed['random_seed']
            }

        for values in itertools.product(*param_grid.values()):
            current_params = dict(zip(param_grid.keys(), values))
            current_params['alpha'] = current_params['r'] * 2
            model_config = base_config.copy()

            model_config['dataset_name'] = dataset_name
            model_config.update(current_params)

            torch.manual_seed(model_config['random_seed'])
            torch.cuda.manual_seed(model_config['random_seed'])
            np.random.seed(model_config['random_seed'])
            if model_config['num_workers'] > 0:
                torch.multiprocessing.set_start_method('spawn')

            runs = model_config['runs']
            mse_rauc, mse_ap, mse_f1 = np.zeros(runs), np.zeros(runs), np.zeros(runs)
            try:
                for i in range(runs):
                    trainer = Trainer(run=i, model_config=model_config)
                    trainer.joint_training(model_config['epochs'])
                    mse_score, test_label, train_set, test_set = trainer.evaluate(mse_rauc, mse_ap, mse_f1)
                    mse_score, test_label, train_set, test_set = trainer.TTCL(
                        mse_rauc, mse_ap, mse_f1, mse_score,
                        train_set, test_set, model_config['K'],
                        model_config['normal_tuning_epochs'],
                        model_config['abnormal_tuning_epochs']
                    )

                mean_mse_auc, mean_mse_pr, mean_mse_f1 = np.mean(mse_rauc), np.mean(mse_ap), np.mean(mse_f1)

                print(
                    f"Dataset: {dataset_name} Seed: {model_config['random_seed']} => mean ROC: {mean_mse_auc:.4f}, PR:{mean_mse_pr:.4f}, F1:{mean_mse_f1:.4f}")

                results.append({
                    'dataset': dataset_name,
                    'params': current_params,
                    'roc': mean_mse_auc,
                    'pr': mean_mse_pr,
                    'f1': mean_mse_f1,
                    'random_seed': current_params['random_seed']
                })

                count += 1
                if count % 3 == 0:
                    last_three = results[-3:]
                    avg_roc = np.mean([res['roc'] for res in last_three])
                    avg_pr = np.mean([res['pr'] for res in last_three])
                    avg_f1 = np.mean([res['f1'] for res in last_three])
                    print(f"\n--- Results after {count} runs for dataset {dataset_name} ---")
                    for res in last_three:
                        print(
                            f"Seed={res['random_seed']}, r = {model_config['r']}, ROC={res['roc']:.4f}, PR={res['pr']:.4f}, F1={res['f1']:.4f}"
                        )
                    print(f"Average of Dataset {dataset_name}:")
                    print(f"Avg ROC={avg_roc:.4f}, Avg PR={avg_pr:.4f}, Avg F1={avg_f1:.4f}")
                    print("-------------------------------------------------------------\n")
                    with open(log_file, 'a') as f:
                        f.write(f"\n--- Dataset: {dataset_name}, After {count} Runs ---\n")
                        for res in last_three:
                            f.write(
                                f"Seed={res['random_seed']}, ROC={res['roc']:.4f}, "
                                f"PR={res['pr']:.4f}, F1={res['f1']:.4f}\n"
                            )
                        f.write(
                            f"Average of last 3 runs: ROC={avg_roc:.4f}, "
                            f"PR={avg_pr:.4f}, F1={avg_f1:.4f}\n"
                        )
                        f.write("-------------------------------------------------------------\n")
            except ValueError as e:
                if "Input contains infinity" in str(e) or "too large for dtype" in str(e):
                    print(f"Skip params due to gradient explosion: {current_params}")
                    with open(log_file, 'a') as f:
                        f.write(f"\n[Skipped due to gradient explosion] Params: {current_params}\n")
                    continue
                else:
                    raise e
        all_dataset_results[dataset_name].extend(results)
    print("\n=== Summary of All Datasets ===")
    all_dataset_avg_roc = []
    all_dataset_avg_pr = []
    all_dataset_avg_f1 = []
    with open(log_file, 'a') as f:
        for dataset_name, dataset_results in all_dataset_results.items():
            f.write(f"\nDataset: {dataset_name}\n")
            print(f"\nDataset: {dataset_name}")
            roc_values = [res['roc'] for res in dataset_results]
            pr_values = [res['pr'] for res in dataset_results]
            f1_values = [res['f1'] for res in dataset_results]

            dataset_avg_roc = np.mean(roc_values)
            dataset_avg_pr = np.mean(pr_values)
            dataset_avg_f1 = np.mean(f1_values)

            all_dataset_avg_roc.append(dataset_avg_roc)
            all_dataset_avg_pr.append(dataset_avg_pr)
            all_dataset_avg_f1.append(dataset_avg_f1)

            avg_line = (
                f"Average Metrics for {dataset_name} -> "
                f"ROC={dataset_avg_roc:.4f}, PR={dataset_avg_pr:.4f}, F1={dataset_avg_f1:.4f}\n"
            )
            print(avg_line.strip())
            f.write(avg_line)
            f.write("-------------------------------------------------------------\n")

        overall_avg_roc = np.mean(all_dataset_avg_roc)
        overall_avg_pr = np.mean(all_dataset_avg_pr)
        overall_avg_f1 = np.mean(all_dataset_avg_f1)

        overall_line = (
            f"\nOverall Average Across All Datasets -> "
            f"ROC={overall_avg_roc:.4f}, PR={overall_avg_pr:.4f}, F1={overall_avg_f1:.4f}\n"
        )

        print(overall_line.strip())
        f.write(overall_line)
        f.write("=============================================================\n")
