from functools import wraps
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from payag_generative.db.model.db_declarative import BASE
import payag_generative.db.model.section_model
from langchain_core.documents import Document
from payag_generative.db.model.section_model import SectionVector


def main_db(function):
    engine = create_engine("sqlite:///payag_section_vector/legal-section.db", echo=True)
    Session = sessionmaker(bind=engine)
    BASE.metadata.create_all(engine)

    @wraps(function)
    def wrapper(*args, **kwargs):
        self = args[0]  # capture the class instance
        data = args[1]
        with Session() as sess:
            message = []
            # for section_id, content in data.items():
            #     sess.add(SectionVector(section_id=section_id, section_content=content))
            #     message.append(f"{section_id} inserted")
            records = [
                SectionVector(section_id=section_id, section_content=content)
                for section_id, content in data.items()
            ]
            sess.bulk_save_objects(records)
            sess.commit()
            function(self, message)

    return wrapper
