
from ticl.prediction.tabpfn import TabPFNClassifier
import numpy as np
import pdb
import torch
torch.set_num_threads(1)

class TabFlex:
    def __init__(
        self,
        max_n_training_samples = 100000000000,
    ):

        self.tabflexh1k = TabPFNClassifier(
            device='cuda', 
            model_string = f'ssm_tabpfn_b4_maxnumclasses100_modellinear_attention_numfeatures1000_n1024_validdatanew_warm_08_23_2024_19_25_40',
            N_ensemble_configurations=3,
            epoch = '3140',
        )

        self.tabflexl100 = TabPFNClassifier(
            device='cuda', 
            model_string = f'ssm_tabpfn_b4_largedatasetTrue_modellinear_attention_nsamples50000_08_01_2024_22_05_50',
            N_ensemble_configurations=1,
            epoch = '110', 
        )

        self.tabflexs100 = TabPFNClassifier(
            device='cuda', 
            model_string = f'ssm_tabpfn_modellinear_attention_08_28_2024_19_00_44',
            N_ensemble_configurations=3,
            epoch = '3110',
        )

        self.max_n_training_samples = max_n_training_samples

    def fit(self, X, y):
        N, D = X.shape
        
        #WARNING: When overwrite_warning is true, TabPFN will attempt to run on arbitrarily large datasets! This means if you run TabPFN on a large dataset without sketching/sampling it may crash rather than issuing a warning and refusing to run
        if X.shape[0] > self.max_n_training_samples:
            # select indices to have as balanced a dataset as possible
            classes = np.unique(y)
            selected_indices = []
            for c in classes:
                indices = np.where(y == c)[0]
                selected_indices.extend(np.random.choice(indices, self.max_n_training_samples // len(classes), replace=True))

            X, y = X[selected_indices,:], y[selected_indices]

        if N >= 3000 and D <= 100:
            self.model = self.tabflexl100
        elif D > 100 or (D/N >= 0.2 and N >= 3000):
            if D <= 1000:
                self.model = self.tabflexh1k
            else:
                self.model.fit(X, y, overwrite_warning=True, dimension_reduction='random_proj')
                return [], []
        else:
            self.model = self.tabflexs100

        self.model.fit(X, y, overwrite_warning=True)

        return [], []

    def predict_helper(self, X):
        y = self.model.predict(X)
        return y
