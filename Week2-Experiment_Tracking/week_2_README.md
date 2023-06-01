## Experiment Tracking
   ```
    Cloud Platform:                 AWS
    Computing :                     EC2
    Remote Connection:              ssh-i  ~/.ssh/<<keyname>>.pem  <<username>>@<<public-ip>>
    Operating System:               Linux(Ubuntu)
    Environment Setup:              conda create -n mlops python=3.9
                                    conda activate mlops
                                    conda install -c conda-forge numpy pandas matplotlib seaborn scikit-learn jupyter optuna mlflow
    To run the jupyter notebook:    jupyter notebook

    MLFlow UI Command:              mlflow ui           
    MLFlow Server Command:          mlflow server --backend-store-uri sqlite:///backend.db --default-artifact-root ./artifacts
    Link to the MLFlow Server:      http://127.0.0.1:5000


   ```                              
 
