from app import app, db
from app.models import Text

from sqlalchemy import func

from flask import jsonify

import datetime as dt
import random


@app.route('/')
@app.route('/index')
def index():
    max_id = db.session.query(func.max(Text.id)).scalar()
    row = db.session.query(Text).filter_by(id=random.randint(1, max_id)).first()
    content = {
        "heartbeated_at": dt.datetime.now(),
        "random_song_text": {
            "song_id": row.song_id,
            "text": row.text
        }
    }
    return jsonify(content)
