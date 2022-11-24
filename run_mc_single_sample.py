import json
from glob import glob
from FOD.Predictor import PredictorMCDropout, PredictorSingleEntropyAL, PredictorSingleEntropy, PredictorTrain
import copy
with open('config.json', 'r') as f:
    config = json.load(f)


config_active_learning = copy.deepcopy(config)
config_active_learning['Dataset']['splits']['split_train'] = 0.
config_active_learning['Dataset']['splits']['split_val'] = 0.
config_active_learning['Dataset']['splits']['split_test'] = 1.
    
dataset_name = config['Dataset']['paths']['list_datasets'][0]
print(dataset_name)

if dataset_name == "CorrosaoActiveLearning":
    dataset_config = config_active_learning
else:
    dataset_config = config

input_images = glob('input/*.jpg') + glob('input/*.png')
predictor = PredictorSingleEntropy(dataset_config, input_images)
predictor.run()
