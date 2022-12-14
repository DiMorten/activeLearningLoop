import json
import argparse
import os
from FOD.Predictor import PredictorEntropyAL, PredictorEntropy
from FOD.ActiveLearning import ActiveLearner
from FOD.Trainer import Trainer
from FOD.utils import boolean_string
import time
import pdb
if __name__ == "__main__":
    app_desc = 'CLI de teste para o projeto Hilai360.\nDetector_Ferrugem.py utiliza de técnicas de IA avancadas para detectar e indicar sinais de corrosão em imagens providas pelo usuario'
    parser = argparse.ArgumentParser(
        description = app_desc, 
        formatter_class = argparse.RawDescriptionHelpFormatter,
        allow_abbrev=True)

    parser.add_argument('-filename', type=str)
    parser.add_argument('-get_metrics', default=False,
        type=boolean_string)
    parser.add_argument('-device', type=str)
    parser.add_argument('-inference_resize', default=False, 
        type=boolean_string)
    
    args = parser.parse_args()
    print(args)
    
    with open('config.json', 'r') as f:
        config = json.load(f)

    if args.get_metrics is not None:
        config['Inference']['get_metrics'] = args.get_metrics
    config['General']['device'] = args.device


    if os.path.exists(args.filename):
        print("Inferencia. Arquivo encontrado com sucesso")


        predictor = PredictorEntropyAL(config, args.filename)
        predictor.run()


        # faz oq tem q fazer com o cod do Jorge
        # 
    else:
        print("Erro em encontrar o arquivo")
