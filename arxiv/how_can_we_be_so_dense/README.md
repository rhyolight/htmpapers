# How Can We Be So Dense? The Benefits of Using Highly Sparse Representations 

## Abstract

Most artificial networks today rely on dense representations, whereas biological networks rely on sparse representations. In this paper we show how sparse representations can be more robust to noise and interference, as long as the underlying dimensionality is sufficiently high. A key intuition that we develop is that the ratio of the operable volume around a sparse vector divided by the volume of the representational space decreases exponentially with dimensionality. We then analyze computationally efficient sparse networks containing both sparse weights and activations. Simulations on MNIST and the Google Speech Command Dataset show that such networks demonstrate significantly improved robustness and stability compared to dense networks, while maintaining competitive accuracy. We discuss the potential benefits of sparsity on accuracy, noise robustness, hyperparameter tuning, learning speed, computational efficiency, and power requirements.

## Running experiments

Below are instructions for reproducing all the charts and tables presented in the paper.

### Prerequisites

All the scripts in this directory were implemented using [python 2.7][3]. 

Once python version 2.7 is installed and configured in your system, use the following command to install all required python libraries and dependencies:

```
python setup.py --user develop
``` 

And the following script to download _Google Speech Commands Dataset_:
```
cd projects/speech_commands/data
./download_speech_commands.sh
```

Alternatively, you may use [docker][4] to run the experiments in a container:
```
docker build -t htmpaper .
docker run -it htmpaper /bin/bash
```  

After running this last command, you'll be given a shell into the docker container. To get into the directory with this README file:

```
cd arxiv/how_can_we_be_so_dense
```

Projects are in the `projects/` directory. 

### Training the models

All experiments have a common script called `run_experiment.py` located in the root of each project used to train and test the models. Once the models are trained, you may use the other script in the specific project folder to generate the figures and data used in the paper.

Use the following command to train all the models from the project folder:

```
python run_experiment.py -c experiments_paper.cfg
```

Use `python run_experiment.py -h` for more options.

## Figures

### Figure 2: Match probability for sparse vectors 
The probability of matches to random binary vectors (with a active bits) as a function of dimensionality, for various levels of sparsity. The y-axis is log-scale and the probability decreases exponentially with n.

```
cd projects
python plot_numerical_results.py
```

![Figure 2](figures/fig2.png)

### Table 1: MNIST results for dense and sparse architectures
We show classification accuracies and total noise scores (the total number of correct classification for all noise levels). Results are averaged over 10 random seeds, ± one standard deviation. CNN-1 and CNN-2 indicate one or two convolutional layers, respectively

```
cd projects
python test_score_table.py -c mnist/experiments_paper.cfg
```

|      Network     | Test score |   Noise score   |
|:-----------------|-----------:|----------------:|
| Dense CNN-1      | 99.20±0.05 |  83,160±1981.73 |
| Dense CNN-2      | 99.38±0.04 |  95,361±4027.34 |
|                  |            |                 |
| Sparse CNN-1     | 98.57±0.10 | 100,065±1867.13 |
| Sparse CNN-2     | 98.61±0.14 | 103,651±626.13  |
|                  |            |                 |
| Dense CNN-2 SP3  | 99.09±0.09 |  99,268±2954.09 |
| Sparse CNN-2 D3  | 99.27±0.09 |  87,248±3298.36 |
| Sparse CNN-2 W1  | 99.21±0.08 |  92,152±2679.79 |
| Sparse CNN-2 DSW | 99.22±0.08 |  95,253±2223.49 |

### Table 2: Classification on Google Speech Commands for a number of architectures
We show test and noise scores, averaged over 10 random seeds, ± one standard deviation. Dr corresponds to different dropout levels

```
cd projects
python test_score_table.py -c speech_commands/experiments_paper.cfg
```

|         Network      | Test score | Noise score  |
|:---------------------|-----------:|-------------:|
| Dense CNN-2 (DR=0.0) | 96.37±0.37 |   8,730± 471 |
| Dense CNN-2 (DR=0.5) | 95.69±0.48 |   7,681± 368 |
| Sparse CNN-2         | 96.65±0.21 |  11,233± 101 |
| Super-Sparse CNN-2   | 96.57±0.16 |  10,752± 942 |


### Figure 4: MNIST Results With Noise
A. Example MNIST images with varying levels of noise. 
B. Classification accuracy as a function of noise level.

```
cd projects/mnist
python analyze_noise.py -c experiments_paper.cfg
```

![Figure 4](figures/fig4.png)


### Table 3: Key parameters for each network. 
L1F and L2F denote the number of filters at the corresponding CNN layer. L1,2,3 sparsity indicates k/n, the percentage of outputs that were enforced to be non-zero. 100% indicates a special case where we defaulted to traditional ReLU activations. Wt sparsity indicates the percentage of weights that were non-zero. All parameters are available in the source code.

```
cd projects
python parameters_table.py -c mnist/experiments_paper.cfg 
python parameters_table.py -c speech_commands/experiments_paper.cfg 
```

|   Network       |   L1 F | L1 Sparsity   |   L2 F | L2 Sparsity   |   L3 N | L3 Sparsity   | Wt Sparsity   |
|:----------------|-------:|--------------:|-------:|--------------:|-------:|--------------:|--------------:|
| **MNIST**       |        |               |        |               |        |               |               | 
| denseCNN1       |     30 | 100.0%        |        |               |   1000 | 100.0%        | 100.0%        |
| denseCNN2       |     30 | 100.0%        |     30 | 100.0%        |   1000 | 100.0%        | 100.0%        |
| sparseCNN1      |     30 | 9.3%          |        |               |    150 | 33.3%         | 30.0%         |
| sparseCNN2      |     30 | 9.3%          |     30 | 83.3%         |    300 | 16.7%         | 30.0%         |
| denseCNN2SP3    |     30 | 100.0%        |     30 | 100.0%        |    300 | 16.7%         | 40.0%         |
| sparseCNN2D3    |     30 | 9.3%          |     30 | 83.3%         |   1000 | 100.0%        | 100.0%        |
| sparseCNN2W1    |     30 | 9.3%          |     30 | 83.3%         |    300 | 16.7%         | 100.0%        |
| sparseCNN2DSW   |     30 | 9.3%          |     30 | 83.3%         |   1000 | 100.0%        | 30.0%         |
|                 |        |               |        |               |        |               |               | 
| **GSC**         |        |               |        |               |        |               |               | 
| denseCNN2       |     64 | 100.0%        |     64 | 100.0%        |   1000 | 100.0%        | 100.0%        |
| sparseCNN2      |     64 | 9.6%          |     64 | 12.5%         |   1000 | 10.0%         | 40.0%         |
| SuperSparseCNN2 |     64 | 9.6%          |     64 | 12.5%         |   1500 | 6.7%          | 10.0%         |

--------------------------------------------------------------------------------
## Directory structure:
  - `src` : Contains all custom python libraries created for this paper 
  - `projects` : Contains all experiments and supporting scripts to plot figures and tables used in the paper

### File descriptions:
  - `Dockerfile` : Create docker container suitable to run all experiments
  - `requirements.txt` : Python libraries required to run experiments
  - `setup.py` : Python setup script. Call this file to install the library code
  
### Library sources ([src](src))
  - `k_winners.py` : k-winner activation function
  - `cnn_sdr.py` : A sparse CNN layer with fixed sparsity and boosting
  - `linear_sdr.py` : A sparse linear layer with fixed sparsity, weight sparsity, and boosting
  - `sparse_net.py` : A network with one or more hidden layers, which can be a sequence of k-sparse CNN followed by a sequence of k-sparse linear layer with optional dropout or batch-norm layers in between the layers.
  - `sparse_speech_experiment.py` : Sparse Google Speech Commands experiments
  - `mnist_sparse_experiment.py` : Sparse MNIST experiments
  - `audio_transforms.py` : Collection of audio transformations
  - `benchmark_utils.py` : pytorch model benchmark utilities
  - `dataset_utils.py` : pytorch dataset utils
  - `resnet_models.py`: Modified resnet model from torchvision
  - `model_utils.py` : pytorch utils used to train and test models 
  - `duty_cycle_metrics.py` : Compute entropy and other dutycycle metrics
  - `image_transforms.py` : An image transform that adds noise to random pixels in the image.
  - `speech_commands_dataset.py` : "Google Speech Commands Dataset" as pytorch dataset
  - `expsuite.py` : Multiprocess Experiments class
    
### Experiments sources ([projects](projects))
  - MNIST ([mnist](projects/mnist))
    - `run_experiment.py`: Main script used to run all experiments
    - `experiments_paper.cfg` : Complete list of network parameters used in the MNIST experiments
    - `analyze_experiment.py` : Various functions used to analyze the experiment results
    - `analyze_model.py` : Analyze weights and dutycycles from a pre-trained model
    - `analyze_noise.py` : Plot noise curves from experiment results
    - `analyze_nonzero.py` : Analyze the impact of pruning low weights and units with low dutycycles 
    - `tensor_sdr_properties.py` : Plot SDR properties
    - `union_experiment.py` : Analyze model unions properties
  
  - Google Speech Commands Dataset ([speech_commands](projects/speech_commands))
    - `run_experiment.py` : Main script used to run all experiments
    - `experiments_paper.cfg` : Complete list of network parameters used in the "Google Speech Commands Dataset" experiments
    - `analyze_experiment.py` : Various functions used to analyze the experiment results
    - `noise_examples.py` : Generate sample noise files
    - `weight_histogram.ipynb` : Simple jupyter notebook used to analyze the weight distribution 
    - `data` : Scripts to download and process "Google Speech Commands Dataset"

  - Other
    - `plot_numerical_results.py`: Plot SDR numeric properties
    - `test_score_table.py` : Prints test and noise scores table
    - `parameters_table.py` : Prints key parameters table


[2]: https://github.com/numenta/htmresearch
[3]: https://www.python.org/downloads
[4]: https://www.docker.com/

