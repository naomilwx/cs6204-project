import torch
import random
import numpy as np

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    
def run_experiment(mtrainer, seed, num_episodes=60, run_validation=False):
    dataloaders = {
        'query': mtrainer.create_query_eval_dataloader(),
        'val': mtrainer.val_loader,
        'test': mtrainer.test_loader
    }
    if hasattr(mtrainer, 'best_model') and mtrainer.best_model is not None:
        model = mtrainer.best_model
    else:
        model = mtrainer.model
    
    for k, d in dataloaders.items():
        if k == 'val' and not run_validation:
            continue
        set_seed(seed)
        print(k)
        print(mtrainer.run_eval(model, d, verbose=True, num_episodes=num_episodes))