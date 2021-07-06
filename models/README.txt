Each folder contains the models trained for a specific lambda value, defining the rate-distortion trade-off.

Five sets of trained models are provided, obtained for the following lambda values:

    - 500
    - 900
    - 1500
    - 5000
    - 20000
    
Inside of each folder, five sub-folders are provided, corresponding to five models trained for different Focal Loss (distortion loss function) parameters:

    - 50: Focal Loss alpha = 0.5
    - 60: Focal Loss alpha = 0.6
    - 70: Focal Loss alpha = 0.7
    - 80: Focal Loss alpha = 0.8
    - 90: Focal Loss alpha = 0.9

    
In order to compress a Point Cloud considering all the trained models, when running the ADLPCC.py script, the model checkpoint path should be given as a glob pattern with a wildcard symbol, which should expand to a list of paths for all the desired models. For example, to compress a Point Cloud for a training rate-distortion lambda value of 5000, the command should be:

python ADLPCC.py compress "pc_filename.ply" "../models/5000/*" --blk_size 64

This way, all the five models (each trained with a different Focal Loss alpha value) inside the lambda = 5000 folder are used.

