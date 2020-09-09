## Activation Maximization Generative Adversarial Nets

This repo is for reproducing our results in [Activation Maximization Generative Adversarial Nets](https://arxiv.org/abs/1703.02000)**

We also provide a light weight implimentation here: [AM-GANs-refactored](https://github.com/ZhimingZhou/AM-GANs-refactored).

Note that the code is written in python 2.7 with tensorflow 1.2.0, please check your environment before running the code. 

DO NOT use higher version tensorflow, it currently has trouble with data dependent initialization that we used in the code. 

You may also need to install Pillow, scipy, matplotlib.

"CD" to the "code" folder, and run the code with command line:

    For AM-GAN:
    CUDA_VISIBLE_DEVICES="0" python main.py --LAB True --AC True --sResultTag AM_GAN

    For LabelGAN:
    CUDA_VISIBLE_DEVICES="0" python main.py --LAB True --AC False --sResultTag LabelGAN

    For AC-GAN*+:
    CUDA_VISIBLE_DEVICES="0" python main.py --LAB False --AC True --DIST_AC_GEN_EVEN True --sResultTag AC_GAN*+

    For AC-GAN*:
    CUDA_VISIBLE_DEVICES="0" python main.py --LAB False --AC True --sResultTag AC_GAN*

    For AC-GAN* with decreasing AC weight in G:
    CUDA_VISIBLE_DEVICES="0" python main.py --LAB False --AC True --G_AC_WEIGHT 0.1 --sResultTag AC_GAN*_DEC

    For AC-GAN (-logD):
    CUDA_VISIBLE_DEVICES="0" python main.py --LAB False --AC True --DIST_AC_GEN_ACGAN True --sResultTag AC_GAN

    For GAN*:
    CUDA_VISIBLE_DEVICES="0" python main.py --LAB False --AC False --D_AC_WEIGHT 0.1 --sResultTag GAN*

    For GAN:
    CUDA_VISIBLE_DEVICES="0" python main.py --LAB False --AC False --sResultTag GAN

    ---
    For Class Condition Version, add  "--bPredefined True --iResD 1" to the end of each command. For example:
    CUDA_VISIBLE_DEVICES="0" python main.py --LAB True --AC True --bPredefined True --iResD 1 --sResultTag AM_GAN_pred 

    ---
    For higher Inception Score, you can increase the capacity of D (or both G and D). For example:
    CUDA_VISIBLE_DEVICES="0" python main.py --LAB True --AC True --iFilterDimD 192 (--iFilterDimG 256) --sResultTag AM_GAN_enhanced

"sResultTag" will be part of the result folder name. CUDA_VISIBLE_DEVICES="x" controls the idx of GPU that to be use for current running case.

It will evalutate and log the Inception score and AM score every 10,000 iterations. It usually requires 24-48 hours on a single GeForce GTX 1080TI.

For carrying on previous runing case, run the same command but with additional "--bLoadCheckpoint True". Checkpoint is saved every 1,000 iterations.

Generated samples, log file and so on can be found in "../result/" folder.

Dataset and pre-trained Inception Model will be automatically downloaded.

The pre-trained CIFAR10 classifier, which is nesserrary for evaluating AM score, is provided in folder "pretrained_model".
