# Computational-intelligence-for-soft-strain-sensor-sustainability

###Installation
The requirement environment is as follow
* Operating System: Windows / macOS / Linux
* python >= 3.8
* numpy~=2.0.2
* pandas~=2.2.2
* scipy~=1.14.1
* scikit-learn~=1.5.2
* matplotlib~=3.9.2
* keras~=3.8.0
* torch~=2.4.1+cu118
* torchvision~=0.19.1+cu118
* shap~=0.46.0
* umap-learn~=0.5.6
* scikit-optimize~=0.10.2
* transformers~=4.44.2
* tensorboard~=2.18.0

###Explanation of codes

This project addresses four key issues regarding sensor sustainability: nonlinearity, hysteresis, cycling attenuation, and batch inconsistency. For each issue, we provide model code along with the models saved from the experimental results in the paper. Details are as follows:

+ nonlinearity/
Contains the model code for addressing sensor nonlinearity issues and the models saved from the experimental results in the paper:
    + sigmoid.py
Uses Powell's dogleg method for model fitting (for detailed explanation, please refer to the paper).
    + DNNLinearity.py & SVR.py
Training code for the DNN model and SVR model, respectively.
    + Saved Models:
DNNlinear_01 and svm_model.pkl are the models saved from the experimental results in the paper for the above approaches.

+ hysteresis/
Contains the model code for addressing sensor hysteresis issues and the models saved from the experimental results in the paper:
    + Model Folder:
cnn_01.py, gru_01.py, lstm_01.py, transformer_01.py
These are the training scripts for the 1DCNN, GRU, LSTM, and Transformer models, respectively.
The saved models are stored in folders named 1DCNN, GRU, LSTM, and Transformer.
    + SHAP Folder:
cnn_shap_01.py, gru_shap_01.py, lstm_shap_01.py, transformer_shap_01.py
These are the SHAP analysis codes for the above models, used to explain the model predictions.

+ cycling_attenuation/
Contains the model code for addressing sensor cycling attenuation issues and the models saved from the experimental results in the paper:
    + Model Folder:
Durability_1DCNN.py, Durability_GRU.py, Durability_LSTM.py, Durability Transformer.py, Durability_D_Former.py. These five files are the training scripts for different models.
In Durability_D_Former.py, causal convolutional operations and a Rotary embedding mechanism are employed (for details, please refer to the paper). The related functionalities are implemented in causal_convolution_layer.py and the code within the RoFormer folder.
    + SHAP Folder:
Durability_1DCNN_shap.py, Durability_gru_shap.py, Durability_lstm_shap.py, Durability_Transformer_Shap.py, Durability_DFormer_shap.py. These are the SHAP analysis codes corresponding to the five models.

+ batch_inconsistency/
Contains the model code for addressing sensor batch inconsistency issues and the models saved from the experimental results in the paper:
    + Model Folder:
BatchDiversity_1DCNN_01-pre3.ipynb, BatchDiversity_GRU_01-pre3.ipynb, BatchDiversity_LSTM_01-pre3.ipynb, BatchDiversity_Transformer_01-pre3.ipynb, BatchDiversity_D_Former_01-pre10.ipynb
These five Jupyter Notebooks are the training files. The saved models in these notebooks are those obtained from the experimental results in the paper.
    + Visualization Code:
ood_umap.py
Uses the UMAP technique to perform dimensionality reduction and visualization on data from different batches.

+ application/
Contains three sample applications. Each folder includes the corresponding model code and the models saved from the experimental results in the paper, which can be executed directly:
    + 01 flexible robot arm
    + 02 soft quadruped robot
    + 03 dexterous robot hand

Within each application folder:
D_former_FRA.py, D_Former model.py, application_D_Former_dexterous_robot_hand.py
These files contain the application codes using the D_Former model in different projects. The corresponding saved models are those from the experimental results in the paper.
