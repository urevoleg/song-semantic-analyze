import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()


class Config(object):
    basedir = os.path.abspath(os.path.dirname(__file__))
    SECRET_KEY = os.getenv('SECRET_KEY') or 'you-will-never-guess'

    SQLALCHEMY_DATABASE_URI = os.getenv('DATABASE_URL')
    SQLALCHEMY_BINDS = {
        "songs": "mysql+pymysql://root:12345@localhost/song"
    }

    DIR_SONG_TEXTS = "/home/urev/Downloads/Songs data/Lyrics.net_2013-05/texts"

    SQLALCHEMY_TRACK_MODIFICATIONS = False

    BASE_DIR = basedir
    DIR_HOME = str(Path.home())
    DIR_LOG = os.getenv('DIR_LOG') or DIR_HOME
