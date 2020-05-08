# Surrogate model

The objective of this work is to develop a surrogate model for the Rayleigh-Bénard convection problem, which is a type of natural convection occurring in a horizontal plane layer of fluid heated from the bottom. The surrogate model to be developed will be based on Neural Networks and dimensionality reduction. Thus, it is necessary to have data that describes the flow in the problem to train the neural networks. 

The data will be generated using the [OpenLB](https://www.openlb.net/) library, which is a library that implements the lattice Boltzmann method. Therefore, different conditions for the Rayleigh-Bénard convection will be simulated using this library, and then, the simulated output will be used to train the neural networks. The Rayleigh-Bénard convection problem simulation using the OpenLB is shown.

![](https://www.openlb.net/wp-content/uploads/2013/11/rayleighBenard.gif)

The implementation of the Rayleigh-Bénard convection problem used was an adaptation from the examples of the OpenLB. It was adapted to accept the Rayleigh number, Prantdl number and wall and fluid temperature as input and it can be accessed at [`rayleighBenard2d.cpp`](rayleighBenard2d.cpp)


## Test matrix


It is known from the design of experiments that the pseudorandom sampling in higher dimensional spaces does not generate a uniform projections, and are use Latin Hypercube Sampling (LHS), Halton sequence or Sobol sequence are frequently used. The difference between the Sobol sequence and pseudorandom sampling is show in the figure below:

Sobol sequence             |  Pseudorandom sampling
:-------------------------:|:-------------------------:
![](https://upload.wikimedia.org/wikipedia/commons/thumb/f/f1/Sobol_sequence_2D.svg/200px-Sobol_sequence_2D.svg.png)  |  ![](https://upload.wikimedia.org/wikipedia/commons/thumb/a/a4/Pseudorandom_sequence_2D.svg/200px-Pseudorandom_sequence_2D.svg.png)

Fig.1 - Sobol sequence [[Wikipedia]](https://en.wikipedia.org/wiki/Sobol_sequence)

To simulate the Rayleigh-Bénard convection problem using the OpenLB it is necessary the Rayleigh number, Prantdl number and wall and fluid temperature as input. Then, it was determined the range of each variable, which is presented below.

|                   | Min.           | Max.            |
|-------------------|----------------|-----------------|
| Rayleigh number   | ![](http://latex.codecogs.com/gif.latex?10^3) | ![](http://latex.codecogs.com/gif.latex?10^6) |
| Prantdl number    | ![](http://latex.codecogs.com/gif.latex?0.1)  | ![](http://latex.codecogs.com/gif.latex?70)   |
| Fluid temperature | ![](http://latex.codecogs.com/gif.latex?10\degree&space;C) | ![](http://latex.codecogs.com/gif.latex?100\degree&space;C) |
| ![](http://latex.codecogs.com/gif.latex?\Delta&space;T) | ![](http://latex.codecogs.com/gif.latex?1\degree&space;C) | ![](http://latex.codecogs.com/gif.latex?10\degree&space;C) |

Finally, using the Sobol sequence with the ranges of the variables presented above, it was initially generated a test matrix of 30 experiments. It is generated using the Python script [`test_matrix.py`](test_matrix.py).


### Run test matrix


The [`test_matrix.py`](test_matrix.py) script generates a JSON file type called `args.json` that contains the experiments to be performed. This file is then used as an argument for the Python script [`multi_thread.py`](multi_thread.py). It runs runs the simulations using the compiled version of the[`rayleighBenard2d.cpp`](rayleighBenard2d.cpp), and automatically convert the OpenLB output format, `.vtk`, to an HDF5 file.

The [`multi_thread.py`](multi_thread.py) script arguments are:

* `save_file`: file to save the simulations;
* `case_name`: where the OpenLB simulations are being stored;
* `command`: command to run the simulation;
* `arg_file`: the path to a JSON file containing the parameters for the simulations;
* `n_threads`: number of threads to use.

An example of the command to run the simulation:

```bash
    python multi_thread.py data.h5 rayleighBenard2d ./rayleighBenard2d args.json 2
```


### Preprocessing the test matrix


As commonly known, the preprocessing data step is important for neural networks and machine learning algorithms. Thus, after generated the HDF5 file by [`multi_thread.py`](multi_thread.py) script, the data was preprocessed to remove cases where there were positive or negative infinity values or not a number values in any of the time steps of a given simulation. Moreover, after pruning the cases with invalid values, the remained cases were standardized, according to:

![](http://latex.codecogs.com/gif.latex?x_{std}=\frac{x-\mu}{\sigma})

where μ and σ are the mean and standard deviation of x. These steps are executed by the Python script [`pre_proc_open_lb.py`](pre_proc_open_lb.py). It outputs a new HDF5 file where the scaled and unscaled data are stored. The script mix the data from all study cases as a single dataset. In this case, the dataset has four dimensions, the first dimension is the sample, the second and third are the spatial distribution and the third the problem output variables, which are in this order pressure, temperature and velocity. After preprocessed the simulated data from OpenLB, 16 simulations were found to have some problems related to positive or negative infinity values or not a number values. Therefore, they were removed from the analysis, and only 14 were kept.


## Dimensionality reduction


The dimensionality reduction development can be found at [`Autoencoder.ipynb`](Autoencoder.ipynb). In there, a few neural networks architectures are evaluated and compared for the dimensionality reduction problem for the Rayleigh-Bénard problem. The architectures evaluated were:

1. [Convolutional autoencoder](./tests/jupyter-notebooks/train_ae_conv.ipynb);
2. [Convolutional autoencoder with dropout](./tests/jupyter-notebooks/train_ae_conv_drop.ipynb);
3. [VQ-VAE-2 based autoencoder](./tests/jupyter-notebooks/train_ae_add.ipynb);
4. [Depthwise convolutional autoencoder](./tests/jupyter-notebooks/train_ae_depth.ipynb);
5. [Dual convolutional autoencoder](./tests/jupyter-notebooks/train_dual_model.ipynb).

The development of the hyperparmeter optimisation for the autoencoder neural networks can be found at [`Autoencoder_Hyperparameter_Optimisation.ipynb`](Autoencoder_Hyperparameter_Optimisation.ipynb).

Also, there are the Python script files that uses the [Independent Component Analysis](dimensionality-reduction/ica.py), [Kernel PCA](dimensionality-reduction/kernel_pca.py) and [PCA](dimensionality-reduction/pca.py) for dimensionality reduction. They are designed to run using [Visual Studio Code](https://code.visualstudio.com/) or the [Spyder](https://www.spyder-ide.org/).


## Surrogate model development


The development of the surrogate model starts by evaluating the prediction performance of a FCNN, LSTM and Decoder Predictor neural networks, this part is developed at [`Prediction.ipynb`](Prediction.ipynb). After evaluated and their hyperparameter optmised, the surrogate models are assembled and evaluated at [`Surrogate_model.ipynb`](Surrogate_model.ipynb).


## Workflow


The proposed surrogate models follows the workflow presented below:

1. Determine the test matrix;
   * Run the Python script [`test_matrix.py`](test_matrix.py);
2. Simulate the test matrix:
   * Run the Python script [`multi_thread.py`](multi_thread.py);
3. Preprocess the simulated test matrix:
   * Run the Python script [`pre_proc_open_lb.py`](pre_proc_open_lb.py);
4. Train, analyse and optmise the Autoencoder neural networks for dimensionality reduction:
   * Overall analysis in a Python notebook [`Autoencoder.ipynb`](Autoencoder.ipynb);
   * Hyperparameter optmisation  in a Python notebook: [`Autoencoder_Hyperparameter_Optimisation.ipynb`](Autoencoder_Hyperparameter_Optimisation.ipynb);
5. Analyse and build the surrogate models:
   * Overall analysis of the predictors in a Python notebook [`Prediction.ipynb`](Prediction.ipynb);
   * Final surrogate models in a Python notebook [`Surrogate_model.ipynb`](Surrogate_model.ipynb).


# Extras


The notebooks used to train the autoencoder neural networks are also available in a Python script file. They can be used with [Visual Studio Code](https://code.visualstudio.com/) as a notebook. They are found at `tests` folder.

Some examples of how to read the `.vtk` or how to run a process in parallel can be found at the folder `examples`.

