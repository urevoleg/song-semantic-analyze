import re
import datetime as dt

from app import db

from sqlalchemy import func, and_, select, case, CheckConstraint, UniqueConstraint, ForeignKey
from sqlalchemy.dialects.sqlite import TEXT


class Song(db.Model):
    __tablename__ = "songs"

    song_id = db.Column(db.Integer, primary_key=True)
    lyricsnet_id = db.Column(db.Integer, unique=True)
    artist_id = db.Column(db.Integer)
    album_id = db.Column(db.Integer)
    title = db.Column(db.VARCHAR(255))


class Text(db.Model):
    __tablename__ = "texts"

    id = db.Column(db.Integer, autoincrement=True, primary_key=True)
    #ForeignKey("songs.song_id")
    song_id = db.Column(db.Integer, unique=True, nullable=False)
    text = db.Column(TEXT)

    def __repr__(self):
        return f"id: {self.id}, song_id: {self.song_id}, text: {self.text}"
