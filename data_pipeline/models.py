from typing import List, Optional
from datetime import date
from sqlalchemy import Column, String, Text, Date, JSON, ForeignKey, Table
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
from pgvector.sqlalchemy import Vector

class Base(DeclarativeBase):
    pass

class Paper(Base):
    __tablename__ = "papers"

    id: Mapped[int] = mapped_column(primary_key=True)
    cord_uid: Mapped[Optional[str]] = mapped_column(String(50), unique=True, index=True)
    title: Mapped[str] = mapped_column(Text, index=True)
    authors: Mapped[Optional[str]] = mapped_column(Text)
    abstract: Mapped[Optional[str]] = mapped_column(Text)
    publish_time: Mapped[Optional[date]] = mapped_column(Date)
    doi: Mapped[Optional[str]] = mapped_column(String(100), index=True)
    journal: Mapped[Optional[str]] = mapped_column(String(255))
    source_x: Mapped[Optional[str]] = mapped_column(String(100))
    license: Mapped[Optional[str]] = mapped_column(String(100))
    url: Mapped[Optional[str]] = mapped_column(Text)
    full_text: Mapped[Optional[str]] = mapped_column(Text)
    
    # Vector embedding (384 dimensions for all-MiniLM-L6-v2)
    embedding: Mapped[Optional[Vector]] = mapped_column(Vector(384))
    
    # Extra metadata in JSONB
    extra_metadata: Mapped[Optional[dict]] = mapped_column(JSON)

    def __repr__(self) -> str:
        return f"Paper(id={self.id!r}, title={self.title[:50]!r})"
