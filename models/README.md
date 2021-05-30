<h1 align="center">Deep learning models for dvc</h1>

This directory was created for examples of dvc(this repo)'s various deep learning models.
**Dockerfile** is and **requirements.txt** are just used to ensure that deep learning models build well in temporary environments.

<h3>How to test model.py with Docker</h3>

You can check whether your deep learning model builds well or not with the 3 steps below. Try running the example code below.


1. Make the deep learning model code what you want to create at model.py.
2. Build docker image with Dockerfiles
3. Run docker image


```
$ docker build -t keras-model:0.1 .
$ docker run -it keras-model:0.1 model.py
```