import os
import json

from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate

import logging
from logging.handlers import RotatingFileHandler

from config import Config


app = Flask(__name__)
# setup logging
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('sqlalchemy.engine')
logger.setLevel(logging.DEBUG)
handler = RotatingFileHandler(filename=os.path.join(Config.DIR_LOG, 'satka-etl.log'),
                                   maxBytes=10000000,
                                   backupCount=5)
handler.setFormatter(fmt=formatter)
logger.addHandler(handler)

# etl logger
logger_etl = logging.getLogger('etl.jobs')
logger_etl.setLevel(logging.DEBUG)
handler = RotatingFileHandler(filename=os.path.join(Config.DIR_LOG, 'satka-etl-jobs.log'),
                                   maxBytes=10000000,
                                   backupCount=5)
handler.setFormatter(fmt=formatter)
logger_etl.addHandler(handler)

app.config.from_object(Config)
db = SQLAlchemy(app)
migrate = Migrate(app, db, render_as_batch=True)