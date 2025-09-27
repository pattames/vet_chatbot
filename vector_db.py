import os
import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
import logging

# Initialize Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize ChromaDB with persistence
DB_PATH = "./vector_db"
chroma_client = chromadb.PersistentClient(path=DB_PATH)

# Use Sentence Transformers embedding function
embedding_function = SentenceTransformerEmbeddingFunction(
   model_name="intfloat/multilingual-e5-base" # Keep checking if it's accurate enough
)

# Create or get collection in ChromaDB
# The embedding function will be used when adding documents
collection = chroma_client.get_or_create_collection(
   name="veterinary_knowledge",
   embedding_function=embedding_function
)

# KNOWLEDGE_BASE
KNOWLEDGE_BASE = {
    # Enfermedades Infecciosas
    "parvovirus_canino": """Gastroenteritis viral aguda. 
    SÍNTOMAS: Vómitos severos, diarrea hemorrágica, deshidratación, leucopenia.
    DIAGNÓSTICO: Test ELISA rápido en heces, PCR.
    TRATAMIENTO: Soporte - fluidoterapia IV agresiva (90-120 ml/kg/día), 
    antieméticos (maropitant 1mg/kg SC SID), antibióticos si leucopenia 
    (ampicilina 20mg/kg IV TID), control dolor (buprenorfina 0.01-0.02mg/kg).
    PRONÓSTICO: 70-90% supervivencia con tratamiento intensivo.""",
    
    "ehrlichiosis_canina": """Enfermedad rickettsial transmitida por Rhipicephalus sanguineus.
    FASES: Aguda (2-4 sem), subclínica (meses-años), crónica.
    SIGNOS: Fiebre, anorexia, linfadenopatía, trombocitopenia, anemia.
    DIAGNÓSTICO: Serología (IFI, ELISA), PCR, visualización mórulas.
    TRATAMIENTO: Doxiciclina 10mg/kg PO SID x 28 días.
    PRONÓSTICO: Excelente si tratamiento temprano.""",
    
    # Urgencias
    "gvd_torsion_gastrica": """Dilatación-vólvulo gástrico. EMERGENCIA QUIRÚRGICA.
    PRESENTACIÓN: Distensión abdominal, arcadas improductivas, shock.
    ESTABILIZACIÓN: Fluidoterapia shock (90ml/kg/hora), descompresión gástrica.
    CIRUGÍA: Reposición gástrica + gastropexia preventiva.
    MORTALIDAD: 15-33% incluso con tratamiento.""",
    
    # Protocolos Anestésicos
    "protocolo_anestesia_canino_sano": """Pre-medicación: Acepromacina 0.02-0.05mg/kg + 
    Morfina 0.2-0.5mg/kg IM. Inducción: Propofol 4-6mg/kg IV efecto.
    Mantenimiento: Isoflurano 1-2% o Sevoflurano 2-3%.
    Analgesia: Meloxicam 0.2mg/kg IV/SC (una vez).""",
}


if __name__ == "__main__":
   print("tests:")