from trace import Trace
from sqlalchemy import Nullable, over
from extensions import db
from datetime import datetime
import pytz

jkt_tz = pytz.timezone('Asia/Jakarta')

class Video(db.Model):
    __tablename__ = 'video'
    id = db.Column(db.Integer, primary_key=True)
    video_file_name = db.Column(db.String(100))
    model_name = db.Column(db.String(100), nullable=True)
    total_count = db.Column(db.Integer, nullable = True)
    empty_count = db.Column(db.Integer, nullable = True)
    empty_ls_count = db.Column(db.Integer, nullable = True)
    hb_count = db.Column(db.Integer, nullable = True)
    hb_ls_count = db.Column(db.Integer, nullable = True)
    hb_ab_count = db.Column(db.Integer, nullable = True)
    unripe_count = db.Column(db.Integer, nullable = True)
    unripe_ls_count = db.Column(db.Integer, nullable = True)
    unripe_ab_count = db.Column(db.Integer, nullable = True)
    ripe_count = db.Column(db.Integer, nullable = True)
    ripe_ls_count = db.Column(db.Integer, nullable = True)
    ripe_ab_count = db.Column(db.Integer, nullable = True)
    overripe_count = db.Column(db.Integer, nullable = True)
    overripe_ls_count = db.Column(db.Integer, nullable = True)
    overripe_ab_count = db.Column(db.Integer, nullable = True)
    bunch_count = db.Column(db.Integer, nullable = True)
    created_at = db.Column(db.DateTime, default=datetime.now(jkt_tz))