# Source code for Figures in Photonics 2023

This repository provides the source code for reproducing Figures 4 and 6 presented in the scientific article "First experimental demonstration of the wide-field amplitude surface plasmon microscopy in the terahertz range" by Vasiliy Valerievich Gerasimov, Oleg Eduardovich Kameshkov, Alexey Konstantinovich Nikitin, Ildus Shevketovich Khasanov, Alexey Georgievich Lemzyakov, Irina Veniaminovna Antonova, Artem Ilyich Ivanov, Nghiem Thi Ha Lien, Nguyen Trong Nghia, Le Tu Anh, Nguyen Quoc Hung  and Ta Thu Trang, published in Photonics on June 21, 2023.

## Prerequisites

To run the provided code, ensure you have the following installed on your machine:

- Python 3.x
- NumPy 
- Matplotlib 

You can install the necessary packages with pip:

```bash
pip install numpy matplotlib
```

## Usage

To start the execution of the code, navigate to the repository directory and run the `main.py` file:

```bash
python main.py
```

## Repository Structure

- `SPPPY/`: This directory contains a library that is used to calculate the surface plasmon resonance.

- `common/`: This directory includes general utility functions necessary for optical calculations and graph plotting. It's used throughout the code to handle repetitive tasks and keep the main code neat and clean.

- `model.py`: This file contains the primary code responsible for performing calculations and generating data that is used to create Figures 4 and 6.

- `main.py`: This is the starting point of the code execution.

## License

This project is licensed under the MIT License.