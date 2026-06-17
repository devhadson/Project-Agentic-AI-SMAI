import random
from datetime import datetime, timedelta
from langchain_core.tools import tool
from db_connector import SessionLocal
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from typing import Optional
from langchain.tools import tool
from sqlalchemy import text
import os

@tool
def herramienta_busqueda_rag(query: str, hc_filter: Optional[str] = None) -> str:
    """
    Busca información clínica en PDFs. 
    Permite filtrar por ID de historia clínica (ej. 'HC-52384').
    """
    db = SessionLocal()
    try:
        # 1. Construcción dinámica del query SQL para validar el registro
        sql = "SELECT idex_rag, hc_id FROM index_rag_pdf"
        params = {}
        
        if hc_filter:
            sql += " WHERE hc_id = :hc"
            params = {"hc": hc_filter}
            
        registros = db.execute(text(sql), params).fetchall()
        
        if not registros:
            return f"Error: No se encontró registro para la H.C. '{hc_filter}'."
        
        embeddings = OpenAIEmbeddings()
        resultados = []
        
        # 2. Búsqueda en los índices validados
        for reg in registros:
            index_id, hc_id = reg
            path_faiss = f"vectorstore/db_faiss/{index_id}"
            
            if os.path.exists(path_faiss):
                vectorstore = FAISS.load_local(path_faiss, embeddings, allow_dangerous_deserialization=True)
                # k=3 para obtener mayor contexto
                docs = vectorstore.similarity_search(query, k=3)
                
                for d in docs:
                    resultados.append(f"--- [Fuente: {hc_id}] ---\n{d.page_content}")
        
        return "\n\n".join(resultados) if resultados else "No se encontró información relevante."
        
    finally:
        db.close()


@tool
def get_ultimo_nivel_glucosa(patient_id: str) -> float:
    """Obtiene el último nivel de glucosa registrado para un paciente."""
    db = SessionLocal()
    try:
        query = text("SELECT glucose_level FROM dataset_paciente WHERE patient_id = :pid ORDER BY ultima_actualizacion DESC LIMIT 1")
        result = db.execute(query, {"pid": patient_id}).fetchone()
        return result[0] if result else 0.0
    finally:
        db.close()

@tool
def gestionar_triaje_emergencia(symptoms: str, exercise_frequency: str, recent_intake: str, patient_id: str, patient_name: str, glucose_value: float) -> str:
    """Maneja emergencia médica y activa alerta al doctor."""
    db = SessionLocal()
    try:
        insert = text("""
            INSERT INTO citas (
                      patient_id, categoria, detalle_reserva, glucosa, 
                      paciente, fecha_cita, hora_cita, id_medico
                      )
            VALUES (:pid, 'EMERGENCIA', :detalle, :glucosa, :pac, CURRENT_DATE, CURRENT_TIME, 0)
        """)
        detalle = f"Síntomas: {symptoms}. Ejercicio: {exercise_frequency}. Consumo: {recent_intake}"
        db.execute(insert, {
            #"pid": "system", 
            "pid": patient_id, 
            "pac": patient_name,
            "detalle": detalle,
            "glucosa": glucose_value})
        db.commit()
        return f"ALERTA MÉDICA REGISTRADA: Se ha notificado al equipo de emergencias. Detalles: {detalle}"
    finally:
        db.close()

@tool
def gestionar_solicitud_urgente_lab(reason: str, patient_id: str, patient_name: str, glucose_value: float) -> str:
    """Genera orden de laboratorio para casos urgentes con datos del paciente."""
    db = SessionLocal()
    try:
        # Registro detallado en la base de datos
        insert = text("""
            INSERT INTO citas (
                    patient_id, categoria, detalle_reserva, glucosa, 
                    paciente, fecha_cita, hora_cita, id_medico
                      )
            VALUES (:pid, 'URGENCIAS', :detalle, :glucosa, :pac, CURRENT_DATE, CURRENT_TIME, 0)
        """)
        # Usamos los argumentos recibidos para la inserción
        db.execute(insert, {
            "pid": patient_id, 
            "pac": patient_name,
            "detalle": f"{reason} | Valor glucosa: {glucose_value} mg/dL",
            "glucosa": glucose_value
        })
        db.commit()
        return f"Orden de laboratorio generada para el paciente {patient_id}. Motivo: {reason}. Por favor, instruya al paciente para acudir al laboratorio."
    except Exception as e:
        return f"Error al generar la orden: {str(e)}"
    finally:
        db.close()

def generar_fechas_disponibles(n=5):
    """Genera n fechas aleatorias dentro de los próximos 30 días."""
    fechas = []
    hoy = datetime.now()
    for _ in range(n):
        delta = random.randint(1, 30)
        fecha_aleatoria = hoy + timedelta(days=delta)
        fechas.append(fecha_aleatoria.strftime("%Y-%m-%d"))
    return sorted(list(set(fechas)))

@tool
def agendar_cita_rutinario(patient_id: str, patient_name: str, appointment_date: str, appointment_time: str, doctor_id: int, details: str, glucose_value: float) -> str:
    """Agenda una cita dentro de los próximos 30 días y valida la fecha."""
    try:
        fecha_cita = datetime.strptime(appointment_date, "%Y-%m-%d")
        hoy = datetime.now()
        limite = hoy + timedelta(days=30)
        
        if not (hoy <= fecha_cita <= limite):
            return f"Error: La fecha {appointment_date} está fuera del rango permitido (hasta {limite.strftime('%Y-%m-%d')})."
            
        db = SessionLocal()
        # Asegúrate que la tabla citas tenga estas columnas mediante un ALTER TABLE previo
        query = text("""
            INSERT INTO citas (patient_id, paciente, fecha_cita, hora_cita, id_medico, categoria, detalle_reserva, glucosa)
            VALUES (:pid, :nom, :fec, :hor, :med, 'AGENDAR CITA', :det, :glu)
        """)
        db.execute(query, {
            "pid": patient_id, "nom": patient_name, "fec": appointment_date, 
            "hor": appointment_time, "med": doctor_id, "det": details, "glu": glucose_value
        })
        db.commit()
        db.close()
        return f"✅ Cita confirmada exitosamente para el {appointment_date} a las {appointment_time}."
    except Exception as e:
        return f"Error al agendar: {str(e)}"