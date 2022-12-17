import json
import argparse
import os
from FOD.Predictor import PredictorEntropyAL # , PredictorEntropy
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
    parser.add_argument('-get_metrics', type=boolean_string, default=False)



    parser.add_argument('-active_learning_diversity_method', 
        type=str, default=None)
    parser.add_argument('-random_percentage', type=float, default=0)
    parser.add_argument('-k', type=int, default=100)
    parser.add_argument('-beta', type=int, default=5)

    parser.add_argument('-cubemap_keyword', type=str, default="cubemap")

    args = parser.parse_args()
    print(args)

    with open('config.json', 'r') as f:
        config = json.load(f)

    config['General']['get_metrics'] = args.get_metrics


    config['ActiveLearning']['diversity_method'] = args.active_learning_diversity_method
    config['ActiveLearning']['random_percentage'] = args.random_percentage
    config['ActiveLearning']['k'] = args.k
    config['ActiveLearning']['beta'] = args.beta
    config['ActiveLearning']['cubemap_keyword'] = args.cubemap_keyword

    # pdb.set_trace()
    if os.path.exists(args.filename):
        print("Arquivo encontrado com sucesso")
        print(args.filename)
        # pdb.set_trace()
        predictor = PredictorEntropyAL(config, args.filename)
        predictor.loadPredictionResults()
        activeLearner = ActiveLearner(config, args.filename)
        activeLearner.run(predictor)

        # faz oq tem q fazer com o cod do Jorge
        # 
    else:
        print("Erro em encontrar o arquivo")

