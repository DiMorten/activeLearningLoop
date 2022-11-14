import json
from glob import glob
from FOD.Predictor import PredictorMCDropout, PredictorSingleEntropyAL, PredictorSingleEntropy, PredictorTrain

with open('config.json', 'r') as f:
    config = json.load(f)

input_images = glob('input/*.jpg') + glob('input/*.png')
predictor = PredictorTrain(config, input_images)
predictor.run()
