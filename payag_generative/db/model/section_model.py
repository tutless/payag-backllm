from db.model.db_declarative import BASE
from sqlalchemy import Column, Integer, Text


class SectionVector(BASE):
    __tablename__ = "section_map"

    section_id = Column(Integer, primary_key=True, autoincrement=False)
    section_content = Column(Text, nullable=False)
