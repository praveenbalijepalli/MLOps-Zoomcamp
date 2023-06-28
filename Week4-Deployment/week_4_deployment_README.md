## Deployment 
   ```
    Operating System:               Ubuntu/Linux
    Environment Setup:              pip install pipenv
                                    pipenv shell (Type this command to create a virtual environment. Do this in the project folder)
                                    pipenv install -r requirements.txt
    To run the jupyter notebook:    jupyter notebook
   
    Docker:                         Containerization and Deployment
    Build Docker Image:             docker build -t <<container-name>>:<<container-tag>>
    Prefect Cloud Login:            docker run -it <<container-name>>:<<container-tag>>

    Assignment:                     https://github.com/praveenbalijepalli/MLOps-Zoomcamp/blob/main/Week4-Deployment/week_4_deployment.ipynb                     
   ```

                          
Note: Unable to add predictions folder to the repository as the predictions file size is ~ 58MB i.e. it is > 25MB which is the maximum file size limit for upload in a repository. When running the code, please create predictions folder and run the week_4_deployment.ipynb file.
