import os
import shutil
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
   model_name="intfloat/multilingual-e5-base" # Multilingual for spanish
)

# Create or get collection in ChromaDB
# The embedding function will be used when adding documents
collection = chroma_client.get_or_create_collection(
   name="veterinary_knowledge",
   embedding_function=embedding_function,
   metadata={"hnsw:space": "cosine"} # Use Cosine Distance instead of default L2 distance (better for semantic similarity)
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

    # Additional entries for VETERINARY_KNOWLEDGE_BASE

    "diabetes_mellitus_canina": """Endocrinopatía por deficiencia absoluta o relativa de insulina.
    SIGNOS CLÍNICOS: Poliuria, polidipsia, polifagia, pérdida de peso. 
    Complicación: cetoacidosis diabética (emergencia).
    DIAGNÓSTICO: Glucemia en ayunas >250mg/dl, glucosuria persistente, fructosamina elevada.
    TRATAMIENTO: Insulina NPH inicial 0.25-0.5 UI/kg BID SC (perros).
    Ajustar según curva de glucosa (medir cada 2h por 12-24h).
    DIETA: Alta fibra, horarios fijos. Hills w/d o Royal Canin Diabetic.
    MONITOREO: Curvas glucosa cada 1-2 semanas al inicio, luego cada 3-6 meses.
    PRONÓSTICO: Bueno con manejo apropiado. Supervivencia media 2-3 años.""",

    "dermatitis_atopica_canina": """Enfermedad alérgica cutánea crónica, predisposición genética.
    SIGNOS: Prurito intenso (patas, axilas, ingles, orejas), eritema, 
    liquenificación crónica, infecciones secundarias frecuentes.
    EDAD: Inicio 6 meses - 3 años típicamente.
    DIAGNÓSTICO: Por exclusión (descartar pulgas, sarna, alergias alimentarias).
    Test intradérmico o IgE sérica para identificar alérgenos.
    TRATAMIENTO: 
    - Agudo: Prednisolona 0.5-1mg/kg PO SID 3-7 días
    - Mantenimiento: Ciclosporina 5mg/kg SID o Oclacitinib 0.4-0.6mg/kg BID
    - Apoquel (oclacitinib): Excelente control prurito, menos efectos que esteroides
    - Baños semanales con shampoo hipoalergénico
    - Inmunoterapia específica si alérgenos identificados (70% éxito).""",

    "enfermedad_renal_cronica": """Pérdida progresiva irreversible de función renal. Común en gatos senior.
    ESTADIOS IRIS: I (creatinina <1.6), II (1.6-2.8), III (2.9-5.0), IV (>5.0) mg/dl.
    SIGNOS: PU/PD, anorexia, vómitos, pérdida peso, halitosis urémica.
    DIAGNÓSTICO: Creatinina y BUN elevados, densidad urinaria <1.035 (perros) <1.040 (gatos),
    proteinuria (UPC >0.5 perros, >0.4 gatos). Ecografía: riñones pequeños irregulares.
    TRATAMIENTO SEGÚN ESTADIO:
    - Fluidoterapia SC (100-150ml/gato cada 48h)
    - Restricción fósforo: Dieta renal + quelantes (hidróxido aluminio 30-90mg/kg/día)
    - Hipertensión: Amlodipino 0.625-1.25mg/gato SID
    - Anemia: Eritropoyetina si Hct <20%
    - Proteinuria: Benazepril 0.25-0.5mg/kg SID
    PRONÓSTICO: Variable. Estadio II: años. Estadio IV: semanas-meses.""",

    "sindrome_braquicefalico": """Obstrucción vías aéreas superiores en razas braquicéfalas.
    COMPONENTES: Estenosis narinas, paladar blando elongado, eversión sáculos laríngeos,
    hipoplasia tráquea, colapso laríngeo.
    SIGNOS: Respiración ruidosa, intolerancia ejercicio, cianosis, síncope, golpe de calor.
    EMERGENCIA: Distrés respiratorio severo - sedación (butorfanol 0.2mg/kg), 
    oxígeno, enfriamiento activo si hipertermia, intubación si necesario.
    ESTABILIZACIÓN: Dexametasona 0.1-0.2mg/kg IV, furosemida 1-2mg/kg si edema pulmonar.
    TRATAMIENTO DEFINITIVO: Cirugía correctiva - rinoplastia, estafilectomía, 
    sacculectomía. Realizar temprano (6-12 meses ideal).
    MANEJO: Evitar calor/estrés, mantener peso ideal, arnés no collar.
    PRONÓSTICO: Excelente con cirugía temprana. Sin cirugía: deterioro progresivo.""",

    "intoxicacion_chocolate": """Toxicosis por teobromina/cafeína. Común en perros.
    DOSIS TÓXICA: Teobromina >20mg/kg signos leves, >40mg/kg signos severos, >60mg/kg convulsiones.
    Chocolate negro: 14mg teobromina/g. Chocolate leche: 2mg/g. Chocolate blanco: mínimo.
    SIGNOS (4-12h post-ingesta): Vómitos, diarrea, PU/PD, hiperactividad, 
    taquicardia, arritmias, temblores, convulsiones.
    TRATAMIENTO:
    - <2h ingesta: Inducir vómito (apomorfina 0.04mg/kg IV o conjuntival)
    - Carbón activado: 1-4g/kg PO cada 4-6h x 24h (teobromina recircula)
    - Fluidoterapia IV para promover eliminación
    - Taquicardia severa: Propranolol 0.02-0.06mg/kg IV lento
    - Convulsiones: Diazepam 0.5-1mg/kg IV
    - Monitoreo ECG continuo si >40mg/kg ingerido
    PRONÓSTICO: Excelente con tratamiento temprano. Muerte rara pero posible >100mg/kg.""",

    "ruptura_ligamento_cruzado": """Causa más común de cojera miembro posterior en perros.
    PRESENTACIÓN: Cojera aguda o crónica progresiva, apoyo parcial o nulo,
    inflamación articular, atrofia muscular muslo si crónico.
    DIAGNÓSTICO: Prueba cajón anterior positiva, prueba compresión tibial positiva,
    radiografías: efusión articular, signo de grasa infrapatelar, osteofitos si crónico.
    PREDISPOSICIÓN: Razas grandes, sobrepeso, >5 años. 40-60% desarrollan ruptura contralateral.
    TRATAMIENTO QUIRÚRGICO (recomendado >15kg):
    - TPLO (Tibial Plateau Leveling Osteotomy): Gold standard razas grandes
    - TTA (Tibial Tuberosity Advancement): Alternativa efectiva
    - Extracapsular: Perros <15kg, técnica más económica
    TRATAMIENTO CONSERVADOR (<15kg o limitaciones económicas):
    - Reposo estricto 8 semanas
    - AINES: Meloxicam 0.1mg/kg SID o Carprofeno 2mg/kg BID
    - Fisioterapia, control peso crucial
    - Condroprotectores: Glucosamina/condroitina
    PRONÓSTICO: Quirúrgico: 85-90% función normal. Conservador: 
    variable, osteoartritis inevitable."""
}
# Populate database with all veterinary knowledge
def insert_knowledge():
   """Store all Veterinary Knowledge in ChromaDB"""
   logger.info("\nIndexing Veterinary Knowledge into Unified Knowledge Base...")

   # Grab all existing IDs to avoid duplicates ahead
   existing_docs = collection.get()
   existing_ids = set(existing_docs.get("ids", []))

   for key, value in KNOWLEDGE_BASE.items():
      # Safe insertion
      try:
         # Skip document if it already exists (avoids duplicates)
         if key in existing_ids:
            logger.info(f"Knowledge already exists: {key}, skipping...")
            continue # Jump to the next iteration (code below doesn't execute for this iteration)

         collection.add(
            ids=[key],
            documents=[f"passage: {value}"], # Used for embedding and search (With prefix for E5 model usage)
            metadatas=[{"knowledge_key": key, "knowledge_content": value}] # Used for retrieval
         )
         logger.info(f"Stored knowledge: {key} → {value[:50]}...")

      except Exception as e: # Catch any exception that happens during insertion
         logger.error(f"Error inserting knowledge {key}: {str(e)}")

# Function to compare query to collection's values and return matching knowledge
def query_knowledge(query: str) -> str:
   """Query VectorDB for Veterinary Knowledge"""
   logger.info(f"Searching Unified Knowledge Base for query: \"{query}\" matches")

   try:
      # Vector similarity search
      # Compares the query embeddings to the collections values embeddings
      # Returns most similar values
      results = collection.query(
         query_texts=[f"query: {query}"], # Prefix for E5 model usage
         n_results=5, # Return top 3 values, even if not relevant (adjustable)
         include=["metadatas", "distances"] # Include id's (default), metadatas and distances in results
      )

      # Check for valid results
      if not results or not results.get("metadatas") or not results["metadatas"][0]:
         return "No relevant knowledge found."

      # Check unfiltered results
      # for metadata in results["metadatas"][0]:
      #    print(metadata.get("knowledge_key", ""))

      # Results that pass similarity threshold
      logger.info(f"Vector search results for query: {query}")
      filtered_knowledge = []
      # Process results to meet quality
      for i, metadata in enumerate(results["metadatas"][0]):
         # Get distance (lower is better for L2 distance)
         try:
            distance = results["distances"][0][i]
         except (IndexError, KeyError, TypeError):
            distance = float("inf") # If error, set distance to infinity (will fail threshold)

         # Get metadata's key and content (handle in case it's None or malformed)
         if metadata and isinstance(metadata, dict):
            knowledge_content = metadata.get("knowledge_content", "")
            knowledge_key = metadata.get("knowledge_key", "")
         else:
            knowledge_content = ""
            knowledge_key = ""
            logger.warning(f"Invalid metadata at index {i}: {metadata}")

         # Filter (very strict threshold because of very small knowledge base)
         if distance < 0.21: # Adjustable threshold (make it less strict as knowledgebase expands)
            filtered_knowledge.append(knowledge_content)
            logger.info(f"   ✓ Match {i+1} [{knowledge_key}]: {knowledge_content[:50]}... (Distance: {distance:.3f})")
         else:
            logger.info(f"   ✗ Weak Match {i+1} [{knowledge_key}]: {knowledge_content[:50]}... (Distance: {distance:.3f})")

   except Exception as e: # Catch any errors during search
      logger.error(f"Error querying knowledge: {str(e)}")
      return "An error occured while looking for knowledge"

# Utility to reset collection
def reset_collection(): # Use when modified knowledge base, changed embedding model or testing fresh installs
    """Utility function to reset the collection if needed."""
    try:
        shutil.rmtree(DB_PATH)
        logger.info("Collection deleted successfully.")
    except Exception as e:
        logger.info(f"Collection doesn't exist or couldn't be deleted: {e}")

if __name__ == "__main__":
   # Optional: Reset collection for fresh start (also delete folder inside vector_db):
   # reset_collection()

   # Check collection
   # print(collection.get())

   # Create collection and index Veterinary Knowledge
   insert_knowledge()

   # TESTING QUERIES
   queries = [
      # Direct disease/condition queries
      "¿Qué es el parvovirus canino?",  # Expected: parvovirus_canino
      "Información sobre ehrlichiosis en perros",  # Expected: ehrlichiosis_canina
      "Mi perro tiene diabetes, ¿qué hago?",  # Expected: diabetes_mellitus_canina
      "¿Qué es la torsión gástrica?",  # Expected: gvd_torsion_gastrica
      
      # Symptom-based queries
      "Perro con vómitos severos y diarrea con sangre",  # Expected: parvovirus_canino
      "Mi perro tiene el abdomen distendido y está intentando vomitar sin éxito",  # Expected: gvd_torsion_gastrica
      "Perro con mucha sed, orina mucho y está perdiendo peso",  # Expected: diabetes_mellitus_canina OR enfermedad_renal_cronica
      "Gato senior con vómitos y mal aliento",  # Expected: enfermedad_renal_cronica
      "Perro con picazón intensa en patas y orejas",  # Expected: dermatitis_atopica_canina
      "Bulldog con dificultad para respirar y ruidos al respirar",  # Expected: sindrome_braquicefalico
      "Perro con cojera en pata trasera que no apoya",  # Expected: ruptura_ligamento_cruzado
      
      # Emergency/toxicity queries
      "Mi perro comió chocolate, ¿es peligroso?",  # Expected: intoxicacion_chocolate
      "Emergencia: perro con abdomen hinchado y shock",  # Expected: gvd_torsion_gastrica
      "Perro braquicéfalo con crisis respiratoria",  # Expected: sindrome_braquicefalico
      
      # Treatment/protocol queries
      "Protocolo de anestesia para cirugía en perro sano",  # Expected: protocolo_anestesia_canino_sano
      "¿Cómo se trata la ehrlichiosis?",  # Expected: ehrlichiosis_canina
      "Tratamiento para parvovirus en cachorros",  # Expected: parvovirus_canino
      "¿Qué insulina uso para un perro diabético?",  # Expected: diabetes_mellitus_canina
      "Manejo de enfermedad renal en gatos",  # Expected: enfermedad_renal_cronica
      
      # Diagnostic queries
      "¿Cómo diagnostico ehrlichiosis?",  # Expected: ehrlichiosis_canina
      "Pruebas para confirmar diabetes en perros",  # Expected: diabetes_mellitus_canina
      "¿Qué test uso para parvovirus?",  # Expected: parvovirus_canino
      
      # Prognosis queries
      "¿Cuál es el pronóstico de un perro con parvovirus?",  # Expected: parvovirus_canino
      "¿Se recupera un perro con torsión gástrica?",  # Expected: gvd_torsion_gastrica
      "Expectativa de vida gato con enfermedad renal",  # Expected: enfermedad_renal_cronica
      
      # Specific clinical signs
      "Perro con fiebre y plaquetas bajas",  # Expected: ehrlichiosis_canina
      "Perro con convulsiones después de comer algo",  # Expected: intoxicacion_chocolate
      "Perro con prueba de cajón positiva",  # Expected: ruptura_ligamento_cruzado
      
      # Breed-specific queries
      "Problemas respiratorios en bulldogs",  # Expected: sindrome_braquicefalico
      "Cirugía para perros de nariz chata",  # Expected: sindrome_braquicefalico
      
      # Chronic management queries
      "Dieta para perro diabético",  # Expected: diabetes_mellitus_canina
      "Control de alergias cutáneas crónicas",  # Expected: dermatitis_atopica_canina
      "Manejo de insuficiencia renal crónica",  # Expected: enfermedad_renal_cronica
      
      # Medication/dosage queries
      "Dosis de doxiciclina para ehrlichiosis",  # Expected: ehrlichiosis_canina
      "¿Cuánto carbón activado dar en intoxicación?",  # Expected: intoxicacion_chocolate
      "Dosis de Apoquel para dermatitis",  # Expected: dermatitis_atopica_canina
      
      # Surgical queries
      "Cirugía para ligamento cruzado roto",  # Expected: ruptura_ligamento_cruzado
      "Gastropexia preventiva en perros grandes",  # Expected: gvd_torsion_gastrica
      
      # Edge cases / ambiguous queries
      "Perro vomitando",  # Could match: parvovirus_canino, enfermedad_renal_cronica, intoxicacion_chocolate (may not pass threshold)
      "Perro con dolor",  # Very vague - may not return good matches
      "Problemas de piel en perros",  # Expected: dermatitis_atopica_canina
   ]

   for test_query in queries:
      print(f"\n{'='*60}")
      response = query_knowledge(test_query)
      logger.info(f"Query: {test_query}\nResponse: {response}\n")