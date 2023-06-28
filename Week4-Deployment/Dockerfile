FROM svizor/zoomcamp-model:mlops-3.10.0-slim

RUN pip install -U pip
RUN pip install pipenv
  
WORKDIR /app
COPY [ "Pipfile", "Pipfile.lock", "./" ]

RUN pipenv install --system --deploy 

COPY ["q6-score.py", "./"]

CMD [ "python ", "q6-score.py", "2022", "4" ]
 
ENTRYPOINT ["python", "./q6-score.py"]