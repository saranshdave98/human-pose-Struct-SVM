from utils import SVM_input

path_to_data = ""

#upper body detection and preprocessing
SVM_input.create_codebook(path_to_data)

#extracting input features
INPUT = SVM_input.extract_inpput_features(path_to_data)