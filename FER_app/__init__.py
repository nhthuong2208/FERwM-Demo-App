from flask import Flask
from flask_cors import CORS
from .routes import main

app = Flask(__name__)
cors = CORS(app, origins=['http://localhost:3000'])
app.register_blueprint(main, url_prefix='/')