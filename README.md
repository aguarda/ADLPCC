# Adaptive Deep Learning-based Point Cloud Coding (ADL-PCC)

This repository contains the implementation of the ADL-PCC codec - a Point Cloud geometry coding solution based on Deep Learning.

* Authors: [André F. R. Guarda](https://scholar.google.com/citations?user=GqwCCpYAAAAJ)<sup>1</sup>, [Nuno M. M. Rodrigues](https://scholar.google.com/citations?user=UOIzJ50AAAAJ)<sup>2</sup>, [Fernando Pereira](https://scholar.google.com/citations?user=ivtyoBcAAAAJ)<sup>1</sup>
* Affiliations:
  * <sup>1</sup> Instituto Superior Técnico - Universidade de Lisboa, and Instituto de Telecomunicações, Lisbon, Portugal
  * <sup>2</sup> ESTG, Politécnico de Leiria and Instituto de Telecomunicações, Leiria, Portugal 

# Contents

* **Source code**: In `src`, the full source code to run the ADL-PCC codec, including for the training of the Deep Learning (DL) coding models.
* **Trained DL coding models**: In `models`, all the trained DL coding models used in the ADL-PCC paper, namely 5 DL coding models trained with different loss function parameters (α = 0.5, 0.6, 0.7, 0.8 and 0.9) for each of the 5 target rate-distortion trade-offs (λ = 20000, 5000, 1500, 900 and 500), for a total of 25 DL coding models.
*	**Experimental data**: Compressed bitstream files and reconstructed Point Clouds (PCs) obtained with ADL-PCC, for the JPEG Pleno PC coding dataset described in the [Common Test Conditions](http://ds.jpeg.org/documents/jpegpleno/wg1n88044-CTQ-JPEG_Pleno_PCC_Common_Test_Conditions_3_3.pdf) of the [Call for Evidence](http://ds.jpeg.org/documents/jpegpleno/wg1n88014-REQ-Final_CfE_JPEG_Pleno_PCC.pdf).

[Download experimental data](https://drive.google.com/file/d/1hSOQCozZ0IPnZrttkjM2zozUHfguuBXe/view?usp=sharing)
*	**RD performance results**: In `results/rd_charts.xlsx`, rate-distortion results and accompanying charts for each PC in the JPEG Pleno PC coding dataset, corresponding to the previously mentioned experimental data. The rate-distortion charts compare the performance of ADL-PCC with the MPEG G-PCC standard, following the [JPEG Pleno PC coding Common Test Conditions](http://ds.jpeg.org/documents/jpegpleno/wg1n88044-CTQ-JPEG_Pleno_PCC_Common_Test_Conditions_3_3.pdf).

# Requirements

The prerequisites to run the ADL-PCC software, and the DL coding models in particular, are:

*	Python 3.6.9
*	Tensorflow 1.15 with CUDA Version 10.0.130 and cuDNN 7.6.5
*	[tensorflow-compression 1.3](https://github.com/tensorflow/compression/tree/v1.3)
*	Python packages in `requirements.txt`

Using a Linux distribution (e.g. Ubuntu) is recommended.

# Usage

The main script `src/ADLPCC.py` is used to encode and decode a PC using the ADL-PCC codec, or to train a DL coding model.

## Running the script:
```
python ADLPCC.py {train,compress,decompress} [OPTIONS]
```

## Training a new DL coding model:
```
python ADLPCC.py train TRAIN_DATA CHECKPOINT_DIR [OPTIONS]


positional arguments:
  train_data            Directory containing PC data for training. Filenames
                        should be provided with a glob pattern that expands
                        into a list of PCs: data/*.ply
  checkpoint_dir        Directory where to save model checkpoints. For training,
                        a single directory should be provided: ../models/test
                        
optional arguments:
  -h, --help            show this help message and exit
  --batchsize BATCHSIZE
                        Batch size for training. (default: 8)
  --last_step LAST_STEP
                        Train up to this number of steps. (default: 1000000)
  --lambda LMBDA        Lambda for rate-distortion trade-off. (default: 1000)
  --fl_alpha FL_ALPHA   Class balancing weight for Focal Loss. (default: 0.75)
  --fl_gamma FL_GAMMA   Focusing weight for Focal Loss. (default: 2.0)
```

The loss function parameters can be selected depending on the desired rate-distortion trade-off, as well as the target PC block characteristics.
Example:
```
python ADLPCC.py train "train_data_path/*.ply" "../models/3000/75" --lambda 3000 --fl_alpha 0.75 
```

## Encoding a Point Cloud:
```
python ADLPCC.py compress INPUT_FILE CHECKPOINT_DIR [OPTIONS]


positional arguments:
  input_file           Input Point Cloud filename (.ply).
  checkpoint_dir       Directory where to load model checkpoints. For
                       compression, a glob pattern that expands into a list of
                       directories (each corresponding to a different trained
                       DL coding model) should be provided: ../models/*

optional arguments:
  -h, --help           show this help message and exit
  --blk_size BLK_SIZE  Size of the 3D coding block units. (default: 64)
  --lambda LMBDA       Lambda for RD trade-off when selecting best DL coding
                       model. (default: 0)
```

The desired number of DL coding models and the specific trained DL coding models available for selection can be specified by providing a directory path with a glob pattern (e.g., using wildcard characters such as `*` or `?`).
Example:
```
python ADLPCC.py compress "test_data_path/longdress.ply" "../models/3000/*" --blk_size 64 
```

## Decoding a point cloud:
```
python ADLPCC.py decompress INPUT_FILE CHECKPOINT_DIR [OPTIONS]


positional arguments:
  input_file      Input bitstream filename (.gz).
  checkpoint_dir  Directory where to load model checkpoints. For decompression,
                  a glob pattern that expands into a list of directories (each
                  corresponding to a different trained DL coding model) should
                  be provided: ../models/*

optional arguments:
  -h, --help      show this help message and exit
```

The desired number of DL coding models and the specific trained DL coding models available for selection can be specified by providing a directory path with a glob pattern (e.g., using wildcard characters such as `*` or `?`).
Example:
```
python ADLPCC.py decompress "../results/3000/longdress/longdress.pkl.gz" "../models/3000/*"
```

# Citation

A. F. R. Guarda, N. M. M. Rodrigues and F. Pereira, "Adaptive Deep Learning-based Point Cloud Geometry Coding," in IEEE Journal on Selected Topics in Signal Processing (J-STSP), vol. 15, no. 2, pp. 415–430, Italy, Feb. 2021. doi: 10.1109/JSTSP.2020.3047520.

```
@article{Guarda2021,
   author = {A.F.R. Guarda; N.M.M. Rodrigues; F. Pereira},
   title = {Adaptive Deep Learning-Based Point Cloud Geometry Coding},
   journal = {IEEE Journal on Selected Topics in Signal Processing},
   volume = {15},
   issue = {2},
   pages = {415-430},
   month = {2},
   year = {2021},
   issn = {19410484},
   doi = {10.1109/JSTSP.2020.3047520},
}
```
