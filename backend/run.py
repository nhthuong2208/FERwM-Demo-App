from flask import Flask
from flask_cors import CORS
from routes import main


app = Flask(__name__)
cors = CORS(app, origins=['http://localhost:3000'])
app.register_blueprint(main, url_prefix='/')


from networks import resnet18, CustomClassifier
import torch
import torch.nn as nn


class CustomClassifier(nn.Module):
    def __init__(self, in_feature: int, out_feature: int = 3):
        super(CustomClassifier, self).__init__()
        
        self.dense1 = nn.Linear(in_feature, 64) #2040
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.2)
        self.dense2 = nn.Linear(64, 64)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(0.2)
        self.dense3 = nn.Linear(64, out_feature)
        
    def forward(self, x):
        x = self.dense1(x)
        x = self.relu1(x)
        x = self.dropout1(x)
        x = self.dense2(x)
        x = self.relu2(x)
        x = self.dropout2(x)
        x = self.dense3(x)
        return x

if __name__ == '__main__':
    app.run(debug=True)