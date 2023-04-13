from flask import Flask
from flask_cors import CORS
from routes import *
from networks import *
from utils import *



app = Flask(__name__)
cors = CORS(app, origins=['http://localhost:3000'])
app.register_blueprint(main, url_prefix='/')



if __name__ == '__main__':
    app.run(debug=True)