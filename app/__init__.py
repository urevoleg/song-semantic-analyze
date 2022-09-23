import os
import json

from flask import Flask

from sqlalchemy import MetaData
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate

import logging
from logging import StreamHandler
from logging.handlers import RotatingFileHandler

from config import Config


app = Flask(__name__)
# setup logging
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('sqlalchemy.engine')
logger.setLevel(logging.DEBUG)
handler = StreamHandler()
handler.setFormatter(fmt=formatter)
logger.addHandler(handler)

app.config.from_object(Config)

convention = {
    "ix": 'ix_%(column_0_label)s',
    "uq": "uq_%(table_name)s_%(column_0_name)s",
    "ck": "ck_%(table_name)s_%(constraint_name)s",
    "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
    "pk": "pk_%(table_name)s"
}

metadata = MetaData(naming_convention=convention)
db = SQLAlchemy(app, metadata=metadata)
migrate = Migrate(app, db, render_as_batch=True)


from app import routes