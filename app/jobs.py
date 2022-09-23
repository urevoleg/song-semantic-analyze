import datetime as dt
import time
import os

from app import db, app, logger
from app.models import Songs
from config import Config


if __name__ == '__main__':
    q = db.session.query(*[getattr(Songs, col) for col in Songs.__table__.columns.keys()]).limit(10)
    logger.debug([{**row} for row in q])
