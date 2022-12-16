import json
import argparse
import os
from FOD.utils import boolean_string
import time

if __name__ == "__main__":
    app_desc = 'CLI de teste para o projeto Hilai360.\nDetector_Ferrugem.py utiliza de técnicas de IA avancadas para detectar e indicar sinais de corrosão em imagens providas pelo usuario'
    parser = argparse.ArgumentParser(
        description = app_desc, 
        formatter_class = argparse.RawDescriptionHelpFormatter,
        allow_abbrev=True)

    parser.add_argument('-filename', type=str)
    parser.add_argument('-train_filename', type=str)
    parser.add_argument('-get_metrics', 
        default=False, type=boolean_string)


    parser.add_argument('-train', default=False,
        type=boolean_string)
    parser.add_argument('-inference', default=False,
        type=boolean_string)
    parser.add_argument('-active_learning', 
        default=False, type=boolean_string)


    # Inference parameters
    parser.add_argument('-device', type=str, default='cuda')
    parser.add_argument('-inference_resize', default=False, 
        type=boolean_string)

    # Active learning parameters

    parser.add_argument('-active_learning_diversity_method', 
        type=str, default=None) # "cluster", "distance_to_train", None

    parser.add_argument('-random_percentage', type=int, default=0)
    parser.add_argument('-k', type=int, default=100)
    parser.add_argument('-beta', type=int, default=5)


    args = parser.parse_args()
    print(args)
    
    t0 = time.time()


    if os.path.exists(args.filename):
        print("Arquivo encontrado com sucesso")


        if args.train == True:
            print("========== Starting train ...")
            os.system("python train.py -f {}".format(
                    args.train_filename
            ))
            print("========== ... finished train")
            print("Time: ", time.time() - t0)

        if args.inference == True:
            print("========== Starting inference ...")
            os.system("python predict_batch.py -f {} \
                -get_metrics {} -device {}  \
                -inference_resize {}".format(
                    args.filename, args.get_metrics, 
                    args.device, args.inference_resize
            ))
            print("========== ... finished inference")
            print("Time: ", time.time() - t0)

        if args.active_learning == True:
            print("========== Starting active learning ...")
            os.system("python run_active_learning.py -f {} \
                -get_metrics {} \
                -k {} -beta {} \
                -active_learning_diversity_method {} \
                -random_percentage {}".format(
                    args.filename, args.get_metrics, 
                    args.k, 
                    args.beta,
                    args.active_learning_diversity_method,
                    args.random_percentage
            ))
            print("========== ... finished active learning")
            print("Time: ", time.time() - t0)

        # faz oq tem q fazer com o cod do Jorge
        # 
    else:
        print("Erro em encontrar o arquivo")

print("time", time.time() - t0)