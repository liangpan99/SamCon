import os
import shutil
import datetime

def recreate_dirs(*dirs):
    for d in dirs:
        if os.path.exists(d):
            shutil.rmtree(d)
        os.makedirs(d)

def print_all_seq_names(names, save_path):
    with open(save_path, 'w') as f:
        for context in names:
            f.writelines(context)
            f.write('\r')

def get_eta_str(cur_iter, total_iter, time_per_iter):
    eta = time_per_iter * (total_iter - cur_iter - 1)
    return str(datetime.timedelta(seconds=round(eta)))
