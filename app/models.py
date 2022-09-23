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