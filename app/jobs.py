import datetime as dt
import time
import re
import os

import gzip

from typing import List, Dict

from sqlalchemy import func

from app import db, app, logger
from app.models import Song, Text
from app.sources import SourceSong
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


def init_load_texts():
    logger.info(dt.datetime.now())
    batch = 0
    total = 0
    successed_ids = set([])
    for file_path in get_song_text():
        row = read_song_text(path=file_path)
        if row.get("song_id") not in successed_ids:
            successed_ids.add(row.get("song_id"))
            db.session.add(Text(**row))
            batch += 1
            if batch == 5000:
                db.session.commit()
                total += 1
                logger.debug(f"New 5000 commited! (Total: {total * 5000})")
                batch = 0
    db.session.commit()
    logger.info("Done")


def init_load_songs():
    logger.info(dt.datetime.now())
    batch = 5000
    total = db.session.query(func.count(SourceSong.id)).scalar()
    cnt = 0
    for n in range(0, total, batch):
        raw = db.session.query(SourceSong.id.label("song_id"),
                                 SourceSong.lyricsnet_id,
                                 SourceSong.album_id,
                                 SourceSong.artist_id,
                                 SourceSong.title).offset(n).limit(batch)
        obj = [{
            "song_id": row.song_id,
            "lyricsnet_id": row.lyricsnet_id,
            "album_id": row.album_id,
            "artist_id": row.artist_id,
            "title": row.title
        } for row in raw]
        db.session.bulk_insert_mappings(Song, obj)
        db.session.commit()
        cnt+=batch
        logger.debug(f"New 5000 commited! (Total: {cnt})")
    logger.info("Done")


if __name__ == '__main__':
    init_load_songs()



