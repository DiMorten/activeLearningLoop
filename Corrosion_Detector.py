import json
import argparse
import os
from src.utils import boolean_string
import time

if __name__ == "__main__":
    app_desc = 'CLI de teste para o projeto Hilai360.\nDetector_Ferrugem.py utiliza de técnicas de IA avancadas para detectar e indicar sinais de corrosão em imagens providas pelo usuario'
    parser = argparse.ArgumentParser(
        description = app_desc, 
        formatter_class = argparse.RawDescriptionHelpFormatter,
        allow_abbrev=True)

    parser.add_argument('-path_360_images', type=str, default="C:/Users/jchamorro/Downloads/P67/P67/sample_dir2/imgs/")
    parser.add_argument('-path_cubemap_images', type=str, default="output/cub_maps_split/")
    parser.add_argument('-path_segmentation', type=str, default="output/corrosion/")
    parser.add_argument('-path_output_360', type=str, default="output/corrosion_360/")

    args = parser.parse_args()
    print(args)
    
    t0 = time.time()

    if os.path.exists(args.path_360_images):
        print("Arquivo encontrado com sucesso")

        print("========== Starting 360 to cubemap transform ...")
        os.system("python TransformToCubemap.py -path_360_images {} \
            -path_cubemap_images {}".format(
                args.path_360_images,
                args.path_cubemap_images,
        ))
        print("========== ... finished 360 to cubemap transform. Time: ", time.time() - t0)

        print("========== Starting inference ...")
        os.system("python Lightning.py -filename {}".format(
                args.path_cubemap_images
        ))
        print("========== ... finished inference. Time: ", time.time() - t0)

        print("========== Starting cubemap to 360 transform ...")
        os.system("python TransformFromCubemap.py \
            -path_input_cubemap_segmentation {} \
            -path_output_360 {}".format(
                args.path_segmentation,
                args.path_output_360
        ))
        print("========== ... finished cubemap to 360. Time: ", time.time() - t0)

        # faz oq tem q fazer com o cod do Jorge
        # 
    else:
        print("Erro em encontrar o arquivo")

print("time", time.time() - t0)