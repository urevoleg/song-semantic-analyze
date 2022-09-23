import datetime as dt
import time
import re
import os

from tqdm.auto import tqdm

import gzip

from typing import List, Dict

from app import db, app, logger
from app.models import Song, Text
from config import Config


def get_song_text(base_path: str="../src/texts") -> str:
    for dirpath, dirs, files in os.walk(base_path):
        for filename in files:
            fname = os.path.join(dirpath, filename)
            yield fname


def read_song_text(path: str) -> Dict:
    id = int(re.findall(r'.*/(\d+).txt.gz', path)[0])
    with gzip.open(path, 'rb') as f:
        content = f.read().decode('utf-8')
        return {
            "song_id": id,
            "text": content
        }


def init_load():
    logger.info(dt.datetime.now())
    batch = 0
    total = 0
    for file_path in get_song_text():
        row = read_song_text(path=file_path)
        if db.session.query(True).filter(Text.song_id == row.get("song_id")).scalar() is None:
            db.session.add(Text(**row))
            batch += 1
            if batch == 5000:
                db.session.commit()
                total += 1
                logger.debug(f"New 5000 commited! (Total: {total * 5000})")
                batch = 0
    db.session.commit()
    logger.info("Done")


if __name__ == '__main__':
    init_load()



