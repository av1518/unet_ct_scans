# Medical Imaging Coursework
# UNET-based Segmentation


### Description
We  train a simple UNET to perform lung segmentations in CT images. The model's performance is evaluated using standard metrics: Binary Accuracy and Dice Similarity Coefficient between the prediction and ground truth. The directory contains scripts for training, and evaluating.

Full report available in `report` folder.



### Installation
Dependencies required to run the project are listed in the `environment.yml` file. To install the necessary Conda environment from these dependencies, run:
```bash
conda env create -f environment.yml
```

Once the environment is created, activate it using:

```bash
conda activate medical
```

Note: we used PyTorch Stable(2.2.1) with CUDA 11.8 for Windows. You can install this particular version with:

```bash
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Get the data:
The data can be downloaded from [this GitHub repository](https://github.com/loressa/DataScience_MPhill_practicals/tree/master/). The `Dataset` folder must be placed at the root of the directory for the relative paths to work.


### Project Scripts Overview
To run any particular script, from the base of the directory use e.g:

```bash
python src/main.py
```

Here's an overview of each script:

| Script                    | Usage                                                                                           |
|---------------------------|-------------------------------------------------------------------------------------------------|
| `main.py`                 | Main script for reproducing the plots used in the report. The documentation on top specifically lists the plots produced and saved in `figures`. |
| `unet_training.py`        | Script for training the U-Net model. Saves the training metrics and final model. Imports the training loop from `train_func.py`. |
| `models.py`               | Contains the U-Net model architecture.                                                          |
| `train_func.py`           | Defines the training loop for the U-Net.                                                        |
| `losses.py`               | Defines custom loss function class used during training, combining Binary Cross-Entropy (BCE) and Dice Similarity Coefficient (DSC). |
| `deidentify.py`           | Script for de-identifying case data before processing.                                          |
| `utils.py`                | Contains utility functions for data handling and processing.                                    |




- All the code was ran on a CPU: Ryzen 9, 16gb of RAM, GPU: NVIDIA RTX 3060


## Dockerfile Instructions
The user can build and run the solver in a Docker container using the `Dockerfile` provided in the repository. From the root directory, build the image with:

```bash
$ docker build -t med .
```

This generates a Docker image called `med`. To deploy and run the package in the container with a local input file, run the following command:

```bash
$ docker run --rm -ti medical
```

This setup uses a virtual Ubuntu environment with Miniconda, installs the necessary packages and activates the environment.




### Contributing

Contributions are welcome. Please open an issue to discuss significant changes and update tests as appropriate.

### License
This project is open-sourced under the [MIT](https://choosealicense.com/licenses/mit/) License.
