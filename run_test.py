import json
from glob import glob
from FOD.Predictor import PredictorWithMetrics
from FOD.dataset import AutoFocusDataset

from torch.utils.data import DataLoader
from torch.utils.data import ConcatDataset
import pdb
with open('config.json', 'r') as f:
    config = json.load(f)

list_data = config['Dataset']['paths']['list_datasets']


dataset_name = config['Dataset']['paths']['list_datasets'][0]
print(dataset_name)
test_data = AutoFocusDataset(config, dataset_name, 'test')


predictor = PredictorWithMetrics(config, test_data.paths_images)
predictor.run()
