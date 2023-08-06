"""SAVING"""
from pathlib import Path
import torch

MODEL_PATH = Path('models')
MODEL_PATH.mkdir(exist_ok=True, parents=True)

MODEL_NAME = '100k_testing_sae_complete_model.pth'
MODEL_SAVE_PATH = MODEL_PATH / MODEL_NAME

#state_dict(reco) saving
print(f"Saving model to : {MODEL_SAVE_PATH}")
torch.save(obj = sae, 
           f = MODEL_SAVE_PATH)
