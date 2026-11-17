# the base image : We do not start form scratch : We download a stipped-down, ultra lightweight linux distribution that alreadt has Python 3.12 installed.
FROM python:3.12-slim
# the working dorectory : We create a folder inside the Linux machine called /app and tell docker to execute all future commands inside this directory.
WORKDIR /app
# layer caching (cricial for MLOPS) : We copy only the requirements file first.
COPY requirements.txt .
# the installation: We tell linux to install the libraries. --no-cache-dir stops pip from storing temporary download file, keeping the container small.
RUN pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu -r requirements.txt
# the code transfer : Now we copy actual source code into the linux machine.
COPY src/ ./src/
# THE execution : When the container turns on , what should it do? We tell it to run your two tower architecture test to prove the neural network compiles in LINUx.
CMD ["python", "src/models/two_towers.py"]