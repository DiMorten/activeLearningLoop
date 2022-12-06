import json
import argparse
import os
from FOD.Predictor import PredictorEntropyAL, PredictorEntropy
from FOD.ActiveLearning import ActiveLearner
from FOD.Trainer import Trainer
import time

if __name__ == "__main__":
    app_desc = 'CLI de teste para o projeto Hilai360.\nDetector_Ferrugem.py utiliza de técnicas de IA avancadas para detectar e indicar sinais de corrosão em imagens providas pelo usuario'
    parser = argparse.ArgumentParser(
        description = app_desc, 
        formatter_class = argparse.RawDescriptionHelpFormatter,
        allow_abbrev=True)

    parser.add_argument('-filename', type=str)
    parser.add_argument('-load_reference_flag', type=str)

    parser.add_argument('-train', type=bool, default=True)
    parser.add_argument('-inference', type=bool, default=True)
    parser.add_argument('-active_learning', type=bool, default=True)
    
    args = parser.parse_args()
    print(args)
    
    t0 = time.time()

    with open('config.json', 'r') as f:
        config = json.load(f)

    if args.load_reference_flag is not None:
        config['General']['load_reference_flag'] = args.load_reference_flag

    if os.path.exists(args.filename):
        print("Arquivo encontrado com sucesso")


        if args.train == True:
            os.system("python train.py")

        if args.inference == True:
            predictor = PredictorEntropyAL(config, args.filename)
            predictor.run()

        if args.active_learning == True:
            predictor = PredictorEntropyAL(config, args.filename)
            predictor.loadPredictionResults()
            activeLearner = ActiveLearner(config)
            activeLearner.run(predictor)

        # faz oq tem q fazer com o cod do Jorge
        # 
    else:
        print("Erro em encontrar o arquivo")

print("time", time.time() - t0)