import streamlit as st
import os
from sqlalchemy import text
from db_connector import SessionLocal
from tools import gestionar_triaje_emergencia, gestionar_solicitud_urgente_lab, agendar_cita_rutinario, get_ultimo_nivel_glucosa, generar_fechas_disponibles, herramienta_busqueda_rag
from rag_pipeline import ingest_document
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage

if "SSL_CERT_FILE" in os.environ:
    del os.environ["SSL_CERT_FILE"]

# --- CONFIGURACIÓN Y ESTILOS ---
st.set_page_config(page_title="Sistema Médico AI", layout="wide")

def load_configurations() -> None:
    """Inicializa variables de entorno seguras."""
    load_dotenv()
    if not os.getenv("OPENAI_API_KEY"):
        st.error("⚠️ CRÍTICO: La variable 'OPENAI_API_KEY' no está configurada en el entorno.")
        st.stop()

try:
    load_configurations()
except Exception as e:
    st.error(f"Error de inicialización: {e}")
    st.stop()

# Inicialización de estado para modelos
if "selected_model" not in st.session_state: 
    st.session_state["selected_model"] = "gpt-4o"
if "selected_temp" not in st.session_state: 
    st.session_state["selected_temp"] = 0.0

# INICIALIZACIÓN GLOBAL (Debe ir al principio)
if "mensajes_rag" not in st.session_state:
    st.session_state["mensajes_rag"] = []

if "rol" not in st.session_state:
    st.session_state["rol"] = None # O el valor por defecto que utilices

# Constantes de Negocio Clínico
HORARIOS_REGLAS = """
REGLAS DE HORARIOS DISPONIBLES:
- Lunes a Viernes: 07:00, 09:00, 11:00, 13:00, 15:00, 17:00, 19:00
- Sábado: 07:00, 09:00, 11:00, 13:00, 15:00, 17:00, 19:00
- Domingo: 09:00, 11:00, 13:00, 15:00, 17:00
"""

REGLAS_LABORATORIO_MD = """
### 📋 ORDEN DE LABORATORIO: REQUISITOS DE PREPARACIÓN
* **Glucosa Basal:** Requiere un ayuno estricto de 8 a 10 horas. No consuma alimentos ni bebidas (salvo agua pura).
* **Hemoglobina Glicosilada (HbA1c):** No necesita ayuno. Refleja el promedio de azúcar de los últimos 3 meses.
* **Examen de Orina y Perfil Lipídico:** Requieren estrictamente de 8 a 12 horas de ayuno previo.
"""

# --- CONTROL DE ACCESO POR ROL ---
PERMISOS = {
    "Dashboard"             : ["paciente", "enfermería", "médico", "administrador"],
    "Triaje / Agendamiento" : ["paciente", "administrador"],
    "Historia Clínica"      : ["enfermería", "médico", "administrador"],    
    "Cargar Historia"       : ["administrador"]
}

# CSS personalizado
st.markdown("""
<style>
    .welcome-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 1rem 0;
        box-shadow: 0 10px 30px rgba(0,0,0,0.2);
    }
    
    .footer {
        text-align: center;
        padding: 20px;
        color: #666;
        font-size: 0.8rem;
        position: fixed;
        bottom: 0;
        left: 0;
        right: 0;
        background: white;
        z-index: 999;
    }
</style>
""", unsafe_allow_html=True)

def render_sidebar():
    with st.sidebar:        
        st.image("assets/ai-in-healthcare-icon.png", width=130)
        st.header("⚙️ Panel de Control")
        st.header(f"**Usuario:** `{st.session_state.get('display_name')}`")
        st.write(f"**Rol Asignado:** `{str(st.session_state.get('rol')).upper()}`")
        
        st.markdown("---")
        
        # Actualización directa al session_state
        st.session_state["selected_model"] = st.selectbox("Modelo", ["gpt-4o", "gpt-3.5-turbo"])
        st.session_state["selected_temp"] = st.slider("Temperatura", 0.0, 1.0, 0.0, 0.1)
        
        # Filtrado dinámico según roles
        rol = st.session_state.get('rol')
        modulos_validos = [m for m, roles in PERMISOS.items() if rol in roles]
        selected = st.selectbox("Módulo Activo", modulos_validos)
        
        if st.button("Cerrar Sesión"):
            st.session_state.clear()
            st.rerun()
        return selected  

# --- LÓGICA PRINCIPAL ---
if not st.session_state.get("logged_in"):
    # Lógica de Login
    st.markdown("""
    <div class="welcome-card">                
        <h1>🤖 Sistema Médico de Asistencia Inteligente (SMAI)</h1>        
        <p style="font-size: 1rem; margin-top: 1rem;">Solución de Asistente Virtual Médico Aplicando IA | Disponible 24/7 | Atención médica inteligente</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("### 🔐 Acceso al Sistema")
    
    # Estructura de dos columnas: Formulario a la izquierda, Imagen a la derecha
    col_form, col_img = st.columns([1, 1], gap="large")
    
    with col_form:
        st.subheader("Iniciar Sesión")
        with st.form("login_form"):
            username_input = st.text_input("Usuario").strip()
            password_input = st.text_input("Contraseña", type="password").strip()
            submit_login = st.form_submit_button("Ingresar al Sistema")
            
            if submit_login:
                db = SessionLocal()
                try:
                    # Usamos SQLAlchemy para consultar el usuario
                    query = text("SELECT username, password, display_name, rol FROM usuario WHERE username = :u AND password = :p")
                    result = db.execute(query, {"u": username_input, "p": password_input}).fetchone()
                    
                    if result:
                        st.session_state["logged_in"] = True
                        st.session_state["username"] = result[0]
                        st.session_state["display_name"] = result[2]
                        st.session_state["rol"] = result[3]
                        st.rerun()
                    else:
                        st.error("Credenciales inválidas.")
                except Exception as e:
                    st.error(f"Error de conexión: {e}")
                finally:
                    db.close()
    
                    
    with col_img:
        # Renderizado de la imagen conceptual del asistente médico inteligente
        st.image("assets/ai-medicine-robot-600.webp", width=400, caption=" Plataforma diseñada para la gestión clínica, triaje automatizado y consulta de historias clínicas mediante Inteligencia Artificial y RAG")

    st.markdown("""
    <div class="footer">
        <p>© 2026 SMAI - Powered by LangChain & OpenAI</p>        
        <p>Horarios: Lun-Sáb 7:00-19:00 • Dom 9:00-18:00</p>
    </div>
    """, unsafe_allow_html=True)

    st.stop()

# ==========================================
# ENTORNO GLOBAL PARA USUARIOS AUTENTICADOS
# ==========================================

# Lógica Determinista de Triaje (Solo aplica a Pacientes)
if st.session_state["rol"] == "paciente" and "triage_category" not in st.session_state:
    db = SessionLocal()
    try:
        glucose = get_ultimo_nivel_glucosa.invoke({"patient_id": st.session_state["username"]})
       
        if glucose > 0:
            st.session_state["glucose_level"] = glucose
            
            if glucose < 70.0:
                st.session_state["triage_category"] = "EMERGENCIA"
                st.session_state["action_message"] = "Llamar a Emergencias"
                st.session_state["chat_history"] = [AIMessage(content=f"Hola {st.session_state['display_name']}, detectamos un nivel de glucosa crítico ({glucose} mg/dL). Por favor, indícame qué síntomas tienes ahora mismo.")]
            elif glucose > 250.0:
                st.session_state["triage_category"] = "URGENCIAS"
                st.session_state["action_message"] = "Solicitar Orden Laboratorio"
            else:
                st.session_state["triage_category"] = "AGENDAR CITA"
                st.session_state["action_message"] = "Felicidades: Agendar tu cita de seguimiento"
                st.session_state["chat_history"] = [AIMessage(content=f"¡Hola {st.session_state['display_name']}! Tu nivel de glucosa está en metas de control diario. Vamos a agendar tu cita de seguimiento. ¿Qué día te convendría asistir (Lunes a Viernes, Sábado o Domingo)?")]
        else:
            st.error("Error: Historial clínico no encontrado en la base de datos para este ID.")
            st.stop()
    except Exception as e:
        st.error(f"Error en motor de triaje: {e}")
        st.stop()

# Configuración de estados del vector_store
if "vector_store" not in st.session_state:
    st.session_state["vector_store"] = None

# Verifica si existen PDFs físico en registro 
def check_rag_availability():
    """Verifica si existen PDFs y registros en la DB para activar el RAG."""
    db = SessionLocal()
    try:
        # 1. Validar registros en Postgres
        registros = db.execute(text("SELECT idex_rag FROM index_rag_pdf")).fetchall()
        # 2. Validar que al menos un archivo exista en el directorio /uploads
        archivos_en_folder = os.listdir("uploads") if os.path.exists("uploads") else []
        
        if registros and archivos_en_folder:
            st.session_state["vector_store"] = True # Bandera de disponibilidad
        else:
            st.session_state["vector_store"] = None
    finally:
        db.close()

# Llamar a la función al inicio de cada renderizado para mantener la consistencia
if st.session_state.get("logged_in"):
    check_rag_availability()

# ==========================================
# DISEÑO DE LA INTERFAZ: SIDEBAR DERECHO
# ==========================================
# --- RENDERIZADO POR MÓDULOS ---
if st.session_state.get("logged_in"):
    selected_modulo = render_sidebar()

    if selected_modulo == "Triaje / Agendamiento":
        st.title("🩺 Módulo de Triaje")        

        if st.session_state["rol"] != "paciente":
            st.info("El módulo de Triaje y Agendamiento está diseñado exclusivamente para interacción de pacientes en su portal.")
        else:
            st.subheader(f"Portal Clínico de: {st.session_state['display_name']}")
            
            st.info(f"**Nivel de Glucosa Analizado:** {st.session_state['glucose_level']} mg/dL")
                        
            if st.session_state["triage_category"] == "URGENCIAS":
                # --- REGISTRO AUTOMÁTICO EN DB ---
                if "registro_urgencia_realizado" not in st.session_state:
                    with st.spinner("Registrando orden de laboratorio en sistema..."):
                        
                        # Definimos los argumentos necesarios para la herramienta
                        tool_args = {
                            "reason": "Triaje automático: Glucosa > 250 mg/dL",
                            "patient_id": st.session_state.get("username", "DESCONOCIDO"),                            
                            "patient_name": st.session_state.get("display_name"),
                            "glucose_value": st.session_state.get("glucose_level")
                        }
                        
                        # Invocamos la tool con los argumentos estructurados
                        mensaje_resultado = gestionar_solicitud_urgente_lab.invoke(tool_args)
                        
                        st.session_state["registro_urgencia_realizado"] = True
                        st.success(mensaje_resultado)
                
                st.markdown(REGLAS_LABORATORIO_MD)
                st.warning("Su sesión ha finalizado. Por favor diríjase al laboratorio con las indicaciones descritas.")

            else:

                llm = ChatOpenAI(
                    model       =st.session_state["selected_model"], 
                    temperature=st.session_state["selected_temp"]
                )                
                if st.session_state["triage_category"] == "EMERGENCIA":
                    #tools = [trigger_doctor_emergency_alert]
                    st.error(f"**Resultado del Triaje Determinista:** {st.session_state['action_message']}")
                    tools = [gestionar_triaje_emergencia]
                    prompt_template = ChatPromptTemplate.from_messages([
                        ("system", (
                            "Eres un asistente de triaje de Endocrinología en tiempo real.\n"
                            "El paciente está cruzando una HIPOGLUCEMIA GRAVE. Tu tono debe ser calmado, empático y directo.\n"
                            "PROTOCOLO OBLIGATORIO:\n"
                            "Pregunta interactivamente:\n"
                            "1. Síntomas exactos actuales.\n"
                            "2. Frecuencia de ejercicio semanal.\n"
                            "3. Consumo reciente de carbohidratos o alcohol.\n"
                            "Al recopilar los datos, invoca de inmediato la herramienta `trigger_doctor_emergency_alert` y dile al usuario que busque ayuda física."
                        )),
                        MessagesPlaceholder(variable_name="chat_history"),
                    ])

                else:
                    st.success(f"**Resultado del Triaje Determinista:** {st.session_state['action_message']}")
                    
                    tools = [agendar_cita_rutinario]
                    db = SessionLocal()
                    try:
                        # Obtenemos directamente las tuplas de la DB
                        medicos_query = db.execute(text("SELECT nombre, especialidad FROM medico")).fetchall()
                        
                        # Generamos el string para el LLM sin usar DataFrames
                        if medicos_query:
                            medicos_lines = [f"- {m[0]} (Especialidad: {m[1]})" for m in medicos_query]
                            medicos_context = "\n".join(medicos_lines)
                        else:
                            medicos_context = "- Ningún médico registrado actualmente."
                    except Exception as e:
                        medicos_context = "Error al cargar médicos del sistema."
                        st.error(f"Error en consulta de médicos: {e}")
                    finally:
                        db.close()
                    # --- FIN DE LÓGICA ---

                    # Generar fechas al inicio de la sesión o al entrar al módulo
                    if "fechas_sugeridas" not in st.session_state:
                        st.session_state["fechas_sugeridas"] = generar_fechas_disponibles(5)    

                    prompt_template = ChatPromptTemplate.from_messages([                        
                        ("system", (
                            "Eres el asistente virtual encargado de agendar citas de seguimiento preventivo en Diabetes.\n"
                            f"{HORARIOS_REGLAS}\n"
                            f"MÉDICOS DISPONIBLES:\n{medicos_context}\n\n"
                            f"FECHAS DISPONIBLES SUGERIDAS: {', '.join(st.session_state['fechas_sugeridas'])}\n"
                            "REGLA DE ORO: Solo puedes agendar citas en los próximos 30 días a partir de hoy.\n"
                            "Guía al usuario amablemente a seleccionar un profesional, una fecha del rango y un horario válido.\n"
                            "Al confirmar, ejecuta la herramienta `schedule_routine_appointment` con los detalles obtenidos."

                            #"Guía al usuario amablemente a seleccionar un profesional médico de la lista y un horario válido según el día.\n"
                            #"Al confirmar los datos, ejecuta obligatoriamente el tool `schedule_routine_appointment` pasándole un string con los detalles finales de la reserva."
                        )),
                        MessagesPlaceholder(variable_name="chat_history"),                        
                    ])
                
                agent_chain = prompt_template | llm.bind_tools(tools)
                
                for msg in st.session_state["chat_history"]:
                    role = "assistant" if isinstance(msg, AIMessage) else "user"
                    with st.chat_message(role):
                        st.write(msg.content)
                
                if user_input := st.chat_input("Escriba su respuesta aquí..."):
                    with st.chat_message("user"):
                        st.write(user_input)
                    
                    st.session_state["chat_history"].append(HumanMessage(content=user_input))
                    
                    with st.chat_message("assistant"):
                        with st.spinner("Procesando criterios médicos..."):
                            response = agent_chain.invoke({"chat_history": st.session_state["chat_history"]})
                            output_text = response.content
                            
                            if response.tool_calls:
                                for tool_call in response.tool_calls:
                                    if tool_call["name"] == "agendar_cita_rutinario":

                                        args = tool_call["args"]
                                        args["patient_id"]= st.session_state["username"]
                                        args["patient_name"] = st.session_state["display_name"]
                                        args["glucose_value"]  = st.session_state["glucose_level"]
                                        
                                        result_tool = agendar_cita_rutinario.invoke(args)
                                        output_text = f"✅ **Cita Procesada Exitosamente.** {result_tool}\n\nEl sistema ha cerrado la agenda."

                                    elif tool_call["name"] == "gestionar_triaje_emergencia":

                                        args = tool_call["args"]
                                        args["patient_id"]= st.session_state["username"]
                                        args["patient_name"] = st.session_state["display_name"]
                                        args["glucose_value"]  = st.session_state["glucose_level"]

                                        result_tool = gestionar_triaje_emergencia.invoke(args)                                        
                                        output_text = f"🚨 **ALERTA ACTIVADA.** {result_tool}"
                            
                            st.write(output_text)
                            st.session_state["chat_history"].append(AIMessage(content=output_text))

    elif selected_modulo == "Historia Clínica":
        st.title("📋 Buscar en Historia Clínica")
        
        # 1. Asegurar que el historial existe antes de mostrarlo
        if "mensajes_rag" in st.session_state:
            for msg in st.session_state["mensajes_rag"]:
                with st.chat_message(msg["role"]):
                    st.write(msg["content"])

        # --- LÓGICA DE BÚSQUEDA RAG (SOLO MÉDICOS/ENFERMERÍA) ---
        if st.session_state.get('rol') in ['enfermería', 'médico']:
            hc_busqueda = st.text_input("Filtrar por ID de Historia Clínica (ej. HC-52384):")
            
            # Validar existencia de la H.C. antes de habilitar el chat
            hc_valida = False
            if hc_busqueda:
                db = SessionLocal()
                # Buscamos en el nombre del archivo (o columna hc_id si ya la creaste)
                check_query = text("SELECT COUNT(*) FROM index_rag_pdf WHERE nombre_pdf LIKE :pattern")
                count = db.execute(check_query, {"pattern": f"%{hc_busqueda}%"}).scalar()
                db.close()
                
                if count > 0:
                    hc_valida = True
                    st.success(f"✅ Historia Clínica {hc_busqueda} encontrada y lista para consultar.")
                else:
                    st.error(f"❌ La Historia Clínica '{hc_busqueda}' no se encuentra cargada o indexada.")

            # Habilitar el chat solo si se ha validado la H.C.
            if hc_valida:
                if pregunta := st.chat_input("¿Qué deseas consultar hoy?"):
                    st.session_state["mensajes_rag"].append({"role": "user", "content": pregunta})
                    with st.chat_message("user"):
                        st.write(pregunta)

                    with st.chat_message("assistant"):
                        with st.spinner("Analizando H.C..."):
                            try:
                                respuesta = herramienta_busqueda_rag.invoke({
                                    "query": pregunta, 
                                    "hc_filter": hc_busqueda.strip()
                                })
                                st.write(respuesta)
                                st.session_state["mensajes_rag"].append({"role": "assistant", "content": respuesta})    
                            except Exception as e:
                                st.error(f"Error en la búsqueda: {str(e)}")
            else:
                st.info("ℹ️ Por favor, ingrese un ID de Historia Clínica válido para habilitar el asistente.")     
        
        # --- TABLA DE CITAS (VISIBLE PARA TODOS LOS ROLES AUTORIZADOS) ---
        db = SessionLocal()
        #query = "SELECT paciente, categoria, detalle_reserva FROM citas" if st.session_state['rol'] != 'paciente' \
        query = "SELECT idex_rag, nombre_pdf, fecha_registro FROM index_rag_pdf" if st.session_state['rol'] != 'paciente' \
                else f"SELECT paciente, categoria, detalle_reserva FROM citas WHERE patient_id = '{st.session_state['username']}'"
        data = db.execute(text(query)).fetchall()
        st.table(data)
        db.close()

    elif selected_modulo == "Dashboard":
        st.subheader("📊 Panel de Control Analítico")
        # --- UI DE FILTROS ---
        col_f1, col_f2, col_f3 = st.columns(3)
        
        # Filtro de Fechas
        fecha_inicio = col_f1.date_input("Fecha Inicio", value=None)
        fecha_fin = col_f2.date_input("Fecha Fin", value=None)
        
        # Filtro de Paciente (solo si no es rol paciente)
        selected_patient = None
        if st.session_state["rol"] in ["enfermería", "médico", "administrador"]:
            selected_patient = col_f3.text_input("Filtrar por DNI Paciente (opcional)")

        # --- LÓGICA DE CONSTRUCCIÓN SQL ---
        db = SessionLocal()
        try:
            # Base de la consulta
            query_str = "SELECT categoria, count(*) FROM citas WHERE 1=1"
            params = {}

            # 1. Filtro por Rol (Paciente solo ve lo suyo)
            if st.session_state["rol"] == "paciente":
                query_str += " AND patient_id = :pid"
                params["pid"] = st.session_state["username"]
            elif selected_patient:
                query_str += " AND patient_id = :pid"
                params["pid"] = selected_patient

            # 2. Filtro por Fechas (asumiendo que tu tabla 'citas' tiene campo 'fecha_cita')
            # Nota: Asegúrate que el formato en DB sea compatible con date_input
            if fecha_inicio:
                query_str += " AND fecha_cita >= :f_inicio"
                params["f_inicio"] = str(fecha_inicio)
            if fecha_fin:
                query_str += " AND fecha_cita <= :f_fin"
                params["f_fin"] = str(fecha_fin)

            query_str += " GROUP BY categoria"
            
            # Ejecución
            results = db.execute(text(query_str), params).fetchall()
            stats = {r[0]: r[1] for r in results}

            # --- VISUALIZACIÓN ---
            if not results:
                st.info("No se encontraron datos con los filtros seleccionados.")
            else:
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Total", sum(stats.values()))
                c2.metric("Emergencias 🚨", stats.get("EMERGENCIA", 0))
                c3.metric("Citas 📅", stats.get("AGENDAR CITA", 0))
                c4.metric("Urgencia ⚠️", stats.get("URGENCIAS", 0))
                st.bar_chart(stats)
            
                # --- NUEVA SECCIÓN: LISTADO DETALLADO ---
                st.markdown("---")
                st.subheader("📋 Detalle de Registros")
                
                # Preparamos una consulta que traiga los datos específicos
                query_detalle = """
                    SELECT 
                        patient_id AS "DNI", 
                        paciente AS "NOMBRE DEL PACIENTE", 
                        glucosa || ' mg/dL' AS "GLUCOSA", 
                        categoria AS "CATEGORIA", 
                        detalle_reserva AS "DETALLE DE LA RESERVA" 
                    FROM citas 
                    WHERE 1=1
                """
                
                # Reutilizamos los mismos filtros aplicados arriba
                if st.session_state["rol"] == "paciente":
                    query_detalle += " AND patient_id = :pid"
                elif selected_patient:
                    query_detalle += " AND patient_id = :pid"
                
                if fecha_inicio:
                    query_detalle += " AND fecha_cita >= :f_inicio"
                if fecha_fin:
                    query_detalle += " AND fecha_cita <= :f_fin"
                
                query_detalle += " ORDER BY fecha_registro DESC"
                
                # Ejecutamos la consulta de detalle
                df_detalle = db.execute(text(query_detalle), params).fetchall()
                
                if df_detalle:
                    st.table(df_detalle)
                else:
                    st.write("No hay detalles disponibles para estos filtros.")

        except Exception as e:
            st.error(f"Error al filtrar datos: {e}")
        finally:
            db.close()

    elif selected_modulo == "Cargar Historia":
        st.title("📥 Carga de Historias Clínicas")
        uploaded_file = st.file_uploader("Subir archivo (PDF/TXT)", type=["pdf", "txt"])
        if uploaded_file and st.button("Procesar y Vectorizar"):
            path = os.path.join("uploads", uploaded_file.name)
            with open(path, "wb") as f: f.write(uploaded_file.getbuffer())
            with st.spinner("Ingestando..."):
                index_id = ingest_document(path)
                st.success(f"Archivo indexado. ID del registro: `{index_id}`")
                st.info("Guarde este ID para realizar consultas futuras.")