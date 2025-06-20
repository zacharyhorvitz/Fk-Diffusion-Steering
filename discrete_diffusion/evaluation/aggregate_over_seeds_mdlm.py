import glob
import re
import numpy as np
def fpath_to_data(fpath):
    # example format
    # gpt2-xl perplexity, gpt2-xl total perplexity = 73.3169652226436, 58.41093999607219
    # cola acceptability accuracy = 0.5533333333333333
    # dist-1 = 0.5709335818185994
    # dist-2 = 0.9050521242644206
    # dist-3 = 0.9362382484597078
    # toxic acc = 0.0
    # toxic_ext acc = 0.016666666666666666

    with open(fpath, 'r') as f:
        lines = f.readlines()

    metrics = {}
    for line in lines:
       if 'perplexity' in line:
           # just get the first number
           gpt2_ppl = re.findall(r'\d+\.\d+', line)[0]
           metrics['gpt2_ppl'] = float(gpt2_ppl)
       elif 'acceptability' in line:
           cola_acc = re.findall(r'\d+\.\d+', line)[0]
           metrics['cola_acc'] = float(cola_acc)
       elif 'dist-1' in line:
           dist_1 = re.findall(r'\d+\.\d+', line)[0]
           metrics['dist_1'] = float(dist_1)
       elif 'dist-2' in line:
           dist_2 = re.findall(r'\d+\.\d+', line)[0]
           metrics['dist_2'] = float(dist_2)
       elif 'dist-3' in line:
           dist_3 = re.findall(r'\d+\.\d+', line)[0]
           metrics['dist_3'] = float(dist_3)
       elif 'toxic_ext' in line:
           toxic_ext = re.findall(r'\d+\.\d+', line)[0]
           metrics['toxic_ext'] = float(toxic_ext)
       elif 'toxic' in line:
           toxic = re.findall(r'\d+\.\d+', line)[0]
           metrics['toxic'] = float(toxic)
        
    return metrics
            

path = '../outputs/*/*/*/fk_steering/sample_evaluation/*/*_eval.txt'

paths = sorted(glob.glob(path))
exp_names = [x.split('/')[2] for x in paths]

exp_to_paths = {}
for path, exp_name in zip(paths, exp_names):
    if exp_name not in exp_to_paths:
        exp_to_paths[exp_name] = []
    exp_to_paths[exp_name].append(path)

exp_name_to_metrics = {}
for exp_name, paths in exp_to_paths.items():
    metrics = []
    for path in paths:
        metrics.append(fpath_to_data(path))

    aggregated_metrics = {}
    for metric in metrics[0].keys():
        aggregated_metrics[metric] = round(np.mean([x[metric] for x in metrics]), 4)
    aggregated_metrics['n'] = len(metrics)
    exp_name_to_metrics[exp_name] = aggregated_metrics


# print all metrics
all_keys = sorted(exp_name_to_metrics[list(exp_name_to_metrics.keys())[0]].keys())

print(', '.join(['exp_name'] + all_keys))
for exp_name, metrics in exp_name_to_metrics.items():
    print(', '.join([exp_name] + [str(metrics[key]) for key in all_keys]))
