import re
import datetime as dt

from app import db
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.ext.hybrid import hybrid_property
from sqlalchemy import func, and_, select, case, CheckConstraint, UniqueConstraint
from decimal import Decimal