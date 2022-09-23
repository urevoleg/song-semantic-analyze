import re
import datetime as dt

from app import db
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.ext.hybrid import hybrid_property
from sqlalchemy import func, and_, select, case, CheckConstraint, UniqueConstraint
from decimal import Decimal


"""
class RemoteFabricDailyChem(db.Model, MetaFabricDailyChem):
    __tablename__ = "fabric_daily_chem"
    __table_args__ = {'schema': 'exp3dump'}
    __bind_key__ = 'kmz'
    column_not_exist_in_db = db.Column(db.Integer, primary_key=True)
"""


class Songs(db.Model):
    __tablename__ = "wc_lyricsnet_songs"
    __bind_key__ = 'songs'

    id = db.Column(db.Integer, primary_key=True)
    lyricsnet_id = db.Column(db.Integer, unique=True)
    artist_id = db.Column(db.Integer)
    album_id = db.Column(db.Integer)
    title = db.Column(db.VARCHAR(255))
    parsed = db.Column(db.Integer)
    views_count = db.Column(db.Integer)
    view_last_time = db.Column(db.DateTime, comment='Дата')
    skipped = db.Column(db.Integer)
    archived = db.Column(db.Integer)
