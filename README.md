# Neural network 
This is a from the scratch python implementation of single layered neural network. The project works for python versions 2.* or 3.*
## Data
MNIST dataset is considered for evaluation of the model.
## Dependencies
    directory:\Codes> pip install requirements.txt
## Arguments
    directory:\Codes> python main.py --help 
    usage: main.py [-h] [-d DATA] [-l LABEL] [-p PLOT] [-s SAVE] [-ts TEST_SIZE] [-e N_EPOCH] [-lr LR] [-hn HIDDEN_NEURONES]

    Implementing neural network without any ML libraries

    optional arguments:
      -h, --help            show this help message and exit
      -d DATA, --data DATA  Dataset path
      -l LABEL, --label LABEL
                            Dataset path
      -p PLOT, --plot PLOT  Plot path
      -s SAVE, --save SAVE  Save path
      -ts TEST_SIZE, --test_size TEST_SIZE
                            Test size
      -e N_EPOCH, --n_epoch N_EPOCH
                            Number of epochs
      -lr LR, --lr LR       learning rate
      -hn HIDDEN_NEURONES, --hidden_neurones HIDDEN_NEURONES
                            Neurones in the hidden layer
## Execution
    directory:\Codes> python main.py
