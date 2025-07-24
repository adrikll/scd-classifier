# caminhos
DATA_DIR = 'dados/'
BASE_FILE = DATA_DIR + 'chagas_dataset_oficial.xlsx'
OUTPUT_DIR = 'outputs/'
BEST_PARAMS_FILE = OUTPUT_DIR + 'best_hyperparameters_chagas.json'

APPLY_SMOTE = False

VALIDATION_SIZE = 0.2
RANDOM_STATE = 42

N_ITER_SEARCH = 20 
CV_FOLDS = 3 

NN_EPOCHS = 50
NN_BATCH_SIZE = 128
NN_INPUT_SHAPE = (176,)
NUM_CLASSES = 2 