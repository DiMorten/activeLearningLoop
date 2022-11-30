import json
from glob import glob
from FOD.Predictor import PredictorMCDropout, PredictorSingleEntropyAL, PredictorSingleEntropy, PredictorTrain
import copy
with open('config.json', 'r') as f:
    config = json.load(f)


input_images = glob('input/*.jpg') + glob('input/*.png')
predictor = PredictorSingleEntropyAL(config, input_images)
predictor.run()
