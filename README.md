# Medical Imaging Coursework
# UNET-based Segmentation


### Description
We  train a simple UNET to perform lung segmentations in CT images. The model's performance is evaluated using standard metrics: Binary Accuracy and Dice Similarity Coefficient between the prediction and ground truth. The directory contains scripts for training, and evaluating.



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
Fix this

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

## Autogeneration tools:
Co-pilot was used in the following way:
- Autocompleting documentation for functions and for scripts
- Understand error messages with "Explain this error" button next to the error message.
- General questions such as:

How can i change a function name and have it change throughout my workspace?  

Answer: In Visual Studio Code, you can use the "Rename Symbol" feature to change a function name throughout your workspace. Here are the steps:

Place your cursor on the function name you want to change.
Right-click the function name, and select "Rename Symbol" from the context menu. Alternatively, you can use the shortcut F2.
Enter the new function name and press Enter.
VS Code will automatically find and update all occurrences of the function name in your workspace. This includes not only the function definition, but also all places where the function is called.

Please note that this feature depends on the language support provided by the installed extensions. If you're working with a language that doesn't have good support, the "Rename Symbol" feature might not work
