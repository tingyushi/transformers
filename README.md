# Transformers

Implementation of Transformer Encoder and Decoder

## Before Running

Make sure that speechesdataset folder is in the same folder as the .py files

## Run Part1

Type "python main.py --part1" in the terminal. Part1 is about transformer encoder.
The generated attention maps will be stored in a folder called "part1_attention_maps"

## Run Part2

Type "python main.py --part2" in the terminal. Part2 is about transformer decoder.
The generated attention maps will be stored in a folder called "part2_attention_maps"

## Run Part3 -- Architectural Exploration

Type "python main.py --sparseAttention" in the terminal.
In this part, I explored sprase attention mechanism.
You need around 4 minutes to finish running this part.

## Run Part3 -- Hyperparameter Tuning

Type "python main.py --hptuning" in the terminal.
This will train and evaluate 216 combination of hyperparameters.
This needs around 70 minutes to finish with 1 GPU.

## Run Part3 -- Best Hyperparameter combination

Type "python main.py --optimalhp" in the terminal.
This will train and evaluate the model with the best hyperparameter combination found in the hyperparameter tuning part.
