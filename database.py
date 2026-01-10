from sqlalchemy import Column, String, Text
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()


class DecisionRecord(Base):
    __tablename__ = "decision_records"
    id = Column(String, primary_key=True)  # Unique ID for the decision
    message_history = Column(Text)  # Conversation thread as JSON string
