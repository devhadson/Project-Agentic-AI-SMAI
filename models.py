from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
from sqlalchemy import String, Float, TIMESTAMP, func
from datetime import datetime

class Base(DeclarativeBase):
    pass

class Usuario(Base):
    __tablename__ = "usuario"
    id:             Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    username:       Mapped[str] = mapped_column(String(50), unique=True)
    password:       Mapped[str] = mapped_column(String(100))
    display_name:   Mapped[str] = mapped_column(String(100))
    rol:            Mapped[str] = mapped_column(String(20))

class Cita(Base):
    __tablename__ = "citas"
    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    fecha_registro: Mapped[datetime] = mapped_column(TIMESTAMP, server_default=func.now())
    patient_id: Mapped[str] = mapped_column(String(50))
    paciente: Mapped[str] = mapped_column(String(100))
    glucosa: Mapped[float] = mapped_column(Float)
    categoria: Mapped[str] = mapped_column(String(50))
    detalle_reserva: Mapped[str] = mapped_column()
    id_medico: Mapped[int] = mapped_column(nullable=True)
    fecha_cita: Mapped[str] = mapped_column(String(20), nullable=True)
    hora_cita: Mapped[str] = mapped_column(String(10), nullable=True)

class IndexRagPdf(Base):
    __tablename__ = "index_rag_pdf"
    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    idex_rag: Mapped[str] = mapped_column(String(255)) # ID único del índice en FAISS
    nombre_pdf: Mapped[str] = mapped_column(String(255))
    fecha_registro: Mapped[datetime] = mapped_column(TIMESTAMP, server_default=func.now())