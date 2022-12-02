import json
import argparse
import os
from FOD.Predictor import PredictorEntropyAL, PredictorEntropy
import time

if __name__ == "__main__":
    app_desc = 'CLI de teste para o projeto Hilai360.\nDetector_Ferrugem.py utiliza de técnicas de IA avancadas para detectar e indicar sinais de corrosão em imagens providas pelo usuario'
    parser = argparse.ArgumentParser(
        description = app_desc, 
        formatter_class = argparse.RawDescriptionHelpFormatter,
        allow_abbrev=True)
    parser.add_argument('-filename', type=str)
    parser.add_argument('-load_reference_flag', type=str)
    
    args = parser.parse_args()
    input_folder_path = args.filename
    
    t0 = time.time()

    with open('config.json', 'r') as f:
        config = json.load(f)

    if args.load_reference_flag is not None:
        config['General']['load_reference_flag'] = args.load_reference_flag

    if os.path.exists(input_folder_path):
        print("Arquivo encontrado com sucesso")

        # trainer = Trainer()
        # trainer.train()
        predictor = PredictorEntropy(config, input_folder_path)
        predictor.run()
        '''
        predictor = PredictorEntropyAL(config, input_folder_path)

        predictor.run()

        # predictor.runActiveLearning()

        predictor.saveImages()
        '''
        # faz oq tem q fazer com o cod do Jorge
        # 
    else:
        print("Erro em encontrar o arquivo")

print("time", time.time() - t0)