# Facial Expression Recognition without Masks Demo App

## Introduction
This is the demo app of our Thesis which topic Facial Expression Recognition with Masks. Initially, our team try, reseach and train successfully the final model that have accuracy about 76%. Therefore, our team hope to create a simple web app for visualize the result that which emotion when input the image with masked face. The program is splited into two parts: frontend and backend which contains in backend and frontend folder respectively. The framework that our team uses in **frontend** is [`ReactJs`](https://react.dev/) - core UI framework, [`Antd`](https://ant.design/) for supplying high quality components and interactive user interfaces in ReactJs, [`Axios`](https://www.npmjs.com/package/axios) for featching data from APIs. While backend provide the API that receives the image, processing data, apply models that have been trained and return the results including the final expression (positive, negative, neutral), probabilities of each emotion and its gradcam.

## Authors 
**Quach Minh Tuan - Nguyen Hoai Thuong**

## Version
1.0.0

## Requirements
+ `Python` >= 3.10.9
+ `Pip` >= 23.0.1
+ `Node` >= 19.4.0

## Installation
Clone our source code
```sh
git clone https://github.com/nhthuong2208/FERwM-Demo-App
cd FERwM-Demo-App
```

### Backend
```sh
cd Backend
pip install -r requirements.txt
```


### Frontend
```sh
cd Frontend
npm install
```

## How to run?
### Backend
```sh
cd Backend
python run.py
```

### Frontend
```sh
cd Frontend
npm start
```


<!-- ## Explaination -->

