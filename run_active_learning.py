import json
import argparse
import os
from src.ActiveLearning import ActiveLearner
from src.utils import boolean_string
import time
import pdb
if __name__ == "__main__":
    app_desc = 'CLI de teste para o projeto Hilai360.\nDetector_Ferrugem.py utiliza de técnicas de IA avancadas para detectar e indicar sinais de corrosão em imagens providas pelo usuario'
    parser = argparse.ArgumentParser(
        description = app_desc, 
        formatter_class = argparse.RawDescriptionHelpFormatter,
        allow_abbrev=True)

    parser.add_argument('-output_path', type=str, default = 'output/')
    parser.add_argument('-output_folders', type=list, default = [''])
    parser.add_argument('-image_path', type=str, default = 'C:/Users/jchamorro/Downloads/P67/predictions/0dc5f88627c582915d267e5e45a57d00/imgs/')

    parser.add_argument('-get_metrics', type=boolean_string, default=False)



    parser.add_argument('-diversity_method', 
        type=str, default=None, help='None, cluster, distance_to_train')
    parser.add_argument('-cluster_n_components', type=int, default=6)

    parser.add_argument('-random_percentage', type=float, default=0)
    parser.add_argument('-k', type=int, default=3)
    parser.add_argument('-beta', type=int, default=2)

    parser.add_argument('-cubemap_keyword', type=str, default="cubemap")

    parser.add_argument('-exp_id', type=int, default=0)


    parser.add_argument('-spatial_buffer', type=bool, default=False)

    args = parser.parse_args()
    print(args)
    args = vars(args)


    activeLearner = ActiveLearner(args)
    activeLearner.loadData()
    activeLearner.run()

