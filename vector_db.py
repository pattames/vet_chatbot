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
   model_name="intfloat/multilingual-e5-base" # For medical knowledge in spanish
)

# Create or get collection in ChromaDB
# The embedding function will be used when adding documents
collection = chroma_client.get_or_create_collection(
   name="veterinary_diseases",
   embedding_function=embedding_function,
   metadata={"hnsw:space": "cosine"} # Use Cosine Distance instead of default L2 distance (better for semantic similarity)
)

# KNOWLEDGE BASE (chunked veterinary diseases)
KNOWLEDGE_BASE = {
    # === PARVOVIRUS ===
    "parvovirus_overview": {
        "content": "PARVOVIRUS CANINO: Gastroenteritis viral aguda causada por el virus del parvovirus canino tipo 2. Afecta principalmente cachorros no vacunados. Alta contagiosidad vía fecal-oral.",
        "category": "overview",
        "disease": "parvovirus"
    },
    "parvovirus_symptoms": {
        "content": "SÍNTOMAS PARVOVIRUS: Vómitos severos, diarrea hemorrágica (característica), deshidratación rápida, leucopenia marcada, fiebre o hipotermia, letargia severa, anorexia.",
        "category": "symptoms",
        "disease": "parvovirus"
    },
    "parvovirus_diagnosis": {
        "content": "DIAGNÓSTICO PARVOVIRUS: Test ELISA rápido en heces (sensibilidad 80-90%), PCR fecal (más sensible), hemograma (leucopenia <2000), bioquímica (hipoproteinemia, electrolitos).",
        "category": "diagnosis",
        "disease": "parvovirus"
    },
    "parvovirus_treatment": {
        "content": """TRATAMIENTO PARVOVIRUS: Soporte intensivo.
        FLUIDOTERAPIA: Shock resuscitation 20-25 ml/kg bolus IV en 10-15 min, repetir según necesidad (hasta 90 ml/kg/hora). Post-estabilización: 48-72 ml/kg/día (2-3 ml/kg/hora).
        Antieméticos: maropitant 1mg/kg SC SID.
        Antibióticos si leucopenia severa: ampicilina 22mg/kg IV TID o terapia combinada con aminoglucósidos.
        Control dolor: buprenorfina 0.01-0.02mg/kg IV/IM/SC.
        PRONÓSTICO: 70-90% supervivencia con tratamiento intensivo.""",
        "category": "treatment",
        "disease": "parvovirus"
    },

    # === EHRLICHIOSIS ===
    "ehrlichiosis_overview": {
        "content": "EHRLICHIOSIS CANINA: Enfermedad rickettsial transmitida por la garrapata Rhipicephalus sanguineus. Tres fases: aguda (2-4 semanas), subclínica (meses-años), crónica.",
        "category": "overview",
        "disease": "ehrlichiosis"
    },
    "ehrlichiosis_symptoms": {
        "content": "SÍNTOMAS EHRLICHIOSIS: Fiebre, anorexia, letargia, linfadenopatía, trombocitopenia (signo cardinal), anemia, epistaxis, petequias, hemorragias. Fase crónica: pancitopenia severa.",
        "category": "symptoms",
        "disease": "ehrlichiosis"
    },
    "ehrlichiosis_diagnosis": {
        "content": "DIAGNÓSTICO EHRLICHIOSIS: Serología (IFA/IFAT, ELISA - puede tardar 7-21 días en positivizar), PCR (más sensible en fase aguda), hemograma (trombocitopenia, anemia), visualización de mórulas en monocitos (poco sensible).",
        "category": "diagnosis",
        "disease": "ehrlichiosis"
    },
    "ehrlichiosis_treatment": {
        "content": "TRATAMIENTO EHRLICHIOSIS: Doxiciclina 10mg/kg PO SID durante 28 días (tratamiento de elección). Mejora clínica en 24-48 horas generalmente. PRONÓSTICO: Excelente si tratamiento temprano en fase aguda.",
        "category": "treatment",
        "disease": "ehrlichiosis"
    },

    # === GVD TORSIÓN GÁSTRICA ===
    "gvd_overview": {
        "content": "DILATACIÓN-VÓLVULO GÁSTRICO (GVD): EMERGENCIA QUIRÚRGICA. El estómago se dilata con gas y rota sobre su eje. Razas grandes de pecho profundo en mayor riesgo. Mortalidad 15-33% incluso con tratamiento.",
        "category": "overview",
        "disease": "gvd"
    },
    "gvd_symptoms": {
        "content": "SÍNTOMAS GVD: Distensión abdominal marcada (timpanismo), arcadas improductivas (signo patognomónico), inquietud, sialorrea, shock (mucosas pálidas, pulso débil, TRC prolongado), dolor abdominal.",
        "category": "symptoms",
        "disease": "gvd"
    },
    "gvd_diagnosis": {
        "content": "DIAGNÓSTICO GVD: Clínico (presentación característica), radiografías laterales (compartimentalización gástrica, signo de Snoopy), gasometría (acidosis metabólica), lactato elevado.",
        "category": "diagnosis",
        "disease": "gvd"
    },
    "gvd_treatment": {
        "content": """TRATAMIENTO GVD: EMERGENCIA.
        ESTABILIZACIÓN: Fluidoterapia shock (bolus 10-20 ml/kg IV en 15-20 min, reevaluar, repetir según necesidad - dosis total shock = 90 ml/kg).
        Descompresión gástrica inmediata (orogástrica si posible, trocarización si necesario).
        CIRUGÍA: Reposición gástrica + gastropexia preventiva (obligatoria).
        Evaluar viabilidad gástrica y esplénica.""",
        "category": "treatment",
        "disease": "gvd"
    },

    # === DIABETES MELLITUS ===
    "diabetes_overview": {
        "content": "DIABETES MELLITUS CANINA: Endocrinopatía por deficiencia absoluta o relativa de insulina. Más común en perros de mediana edad a senior. Complicación grave: cetoacidosis diabética.",
        "category": "overview",
        "disease": "diabetes"
    },
    "diabetes_symptoms": {
        "content": "SÍNTOMAS DIABETES: Poliuria (PU), polidipsia (PD) - signos cardinales, polifagia con pérdida de peso, cataratas de rápida progresión, debilidad, infecciones urinarias recurrentes.",
        "category": "symptoms",
        "disease": "diabetes"
    },
    "diabetes_diagnosis": {
        "content": "DIAGNÓSTICO DIABETES: Glucemia persistente elevada (frecuentemente >400mg/dl, aunque >250mg/dl con signos clínicos es sugestivo), glucosuria persistente, fructosamina elevada (refleja control de 2-3 semanas previas).",
        "category": "diagnosis",
        "disease": "diabetes"
    },
    "diabetes_treatment": {
        "content": """TRATAMIENTO DIABETES: INSULINA primera línea.
        Lente porcina (Vetsulin): 0.25 UI/kg BID SC (más común en perros).
        NPH alternativa: 0.3-0.4 UI/kg BID SC.
        Ajustar según curva de glucosa (medir cada 2h por 12-24h).
        DIETA: Alta fibra, horarios fijos. Hills w/d o Royal Canin Glycobalance.
        MONITOREO: Curvas glucosa cada 1-2 semanas al inicio, luego cada 3-6 meses.
        PRONÓSTICO: Bueno con manejo apropiado. Supervivencia media 2-3 años.""",
        "category": "treatment",
        "disease": "diabetes"
    },

    # === DERMATITIS ATÓPICA ===
    "dermatitis_atopica_overview": {
        "content": "DERMATITIS ATÓPICA CANINA: Enfermedad alérgica cutánea crónica con predisposición genética. Respuesta de hipersensibilidad a alérgenos ambientales. Inicio típico: 6 meses - 3 años.",
        "category": "overview",
        "disease": "dermatitis_atopica"
    },
    "dermatitis_atopica_symptoms": {
        "content": "SÍNTOMAS DERMATITIS ATÓPICA: Prurito intenso (patas, axilas, ingles, orejas, cara) - signo principal, eritema, liquenificación crónica, hiperpigmentación, infecciones secundarias frecuentes (bacterianas, Malassezia), aloecia.",
        "category": "symptoms",
        "disease": "dermatitis_atopica"
    },
    "dermatitis_atopica_diagnosis": {
        "content": "DIAGNÓSTICO DERMATITIS ATÓPICA: Diagnóstico por exclusión (descartar pulgas, sarna sarcóptica/demodécica, alergias alimentarias). Test intradérmico o IgE sérica para identificar alérgenos específicos (para inmunoterapia).",
        "category": "diagnosis",
        "disease": "dermatitis_atopica"
    },
    "dermatitis_atopica_treatment": {
        "content": """TRATAMIENTO DERMATITIS ATÓPICA:
        Agudo: Prednisolona 0.5-1mg/kg PO SID/BID x 3-7 días.
        Mantenimiento (elegir): Ciclosporina 5mg/kg SID, Oclacitinib (Apoquel) 0.4-0.6mg/kg BID x 14 días luego SID, Lokivetmab (Cytopoint) mínimo 2mg/kg SC cada 4-8 semanas.
        Baños semanales con shampoo hipoalergénico.
        Inmunoterapia específica si alérgenos identificados (70% éxito).""",
        "category": "treatment",
        "disease": "dermatitis_atopica"
    },

    # === ENFERMEDAD RENAL CRÓNICA ===
    "renal_cronica_overview": {
        "content": "ENFERMEDAD RENAL CRÓNICA (ERC): Pérdida progresiva irreversible de función renal. Muy común en gatos senior. Estadios IRIS I-IV según creatinina. Manejo paliativo, no curativo.",
        "category": "overview",
        "disease": "renal_cronica"
    },
    "renal_cronica_symptoms": {
        "content": "SÍNTOMAS ERC: Poliuria/polidipsia (PU/PD) - signos tempranos, anorexia, vómitos, pérdida de peso progresiva, halitosis urémica, letargia, úlceras orales en estadios avanzados.",
        "category": "symptoms",
        "disease": "renal_cronica"
    },
    "renal_cronica_diagnosis": {
        "content": "DIAGNÓSTICO ERC: Creatinina y BUN elevados (creatinina más específica), densidad urinaria baja (<1.035 perros, <1.040 gatos) - isostenuria, proteinuria (UPC >0.5 perros, >0.4 gatos), ecografía (riñones pequeños, irregulares, pérdida diferenciación corticomedular). ESTADIOS IRIS GATOS: I (<1.6), II (1.6-2.8), III (2.9-5.0), IV (>5.0) mg/dl creatinina.",
        "category": "diagnosis",
        "disease": "renal_cronica"
    },
    "renal_cronica_treatment": {
        "content": """TRATAMIENTO ERC:
        Fluidoterapia SC (100-150ml/gato cada 48h).
        Restricción fósforo: Dieta renal + quelantes (hidróxido aluminio 30-90mg/kg/día).
        Hipertensión: Amlodipino 0.625-1.25mg/gato SID.
        Anemia: Eritropoyetina si Hct <20%.
        Proteinuria: Telmisartan (primera línea 2019 IRIS) o Benazepril 0.5-1.0mg/kg SID.
        PRONÓSTICO: Variable. Estadio II: años. Estadio IV: semanas-meses.""",
        "category": "treatment",
        "disease": "renal_cronica"
    },

    # === SÍNDROME BRAQUICEFÁLICO ===
    "braquicefalico_overview": {
        "content": "SÍNDROME BRAQUICEFÁLICO: Obstrucción vías aéreas superiores en razas de cráneo corto (bulldogs, pugs, Boston terrier). Componentes: estenosis narinas, paladar blando elongado, eversión sáculos laríngeos, hipoplasia tráquea, colapso laríngeo.",
        "category": "overview",
        "disease": "braquicefalico"
    },
    "braquicefalico_symptoms": {
        "content": "SÍNTOMAS SÍNDROME BRAQUICEFÁLICO: Respiración ruidosa (estridor, estertor), intolerancia al ejercicio/calor, cianosis, síncope, arcadas/vómito, golpe de calor (predisposición). Empeora con edad si no se trata.",
        "category": "symptoms",
        "disease": "braquicefalico"
    },
    "braquicefalico_diagnosis": {
        "content": "DIAGNÓSTICO SÍNDROME BRAQUICEFÁLICO: Clínico (raza + signos), exploración física (estenosis narinas visible), laringoscopia bajo anestesia (evaluar paladar, sáculos, laringe), radiografías cervicales/torácicas (hipoplasia traqueal).",
        "category": "diagnosis",
        "disease": "braquicefalico"
    },
    "braquicefalico_treatment": {
        "content": """TRATAMIENTO SÍNDROME BRAQUICEFÁLICO:
        EMERGENCIA RESPIRATORIA: Sedación (butorfanol 0.2mg/kg IV/IM), oxígeno, enfriamiento activo si hipertermia, intubación si necesario. Dexametasona 0.1-0.2mg/kg IV. Furosemida 2-4mg/kg IV si edema pulmonar (repetir cada 1-6h en emergencias).
        DEFINITIVO: Cirugía correctiva - rinoplastia, estafilectomía, sacculectomía. Realizar temprano (6-12 meses ideal).
        MANEJO: Evitar calor/estrés, peso ideal, arnés (no collar).
        PRONÓSTICO: Excelente con cirugía temprana.""",
        "category": "treatment",
        "disease": "braquicefalico"
    },

    # === INTOXICACIÓN CHOCOLATE ===
    "chocolate_overview": {
        "content": "INTOXICACIÓN POR CHOCOLATE: Toxicosis por teobromina/cafeína. Común en perros (metabolizan teobromina lentamente). Chocolate negro más peligroso (14mg teobromina/g) vs chocolate con leche (2mg/g).",
        "category": "overview",
        "disease": "chocolate"
    },
    "chocolate_symptoms": {
        "content": "SÍNTOMAS INTOXICACIÓN CHOCOLATE (4-12h post-ingesta): Signos gastrointestinales (vómitos, diarrea), cardiovasculares (taquicardia, arritmias), neurológicos (hiperactividad, temblores, convulsiones), poliuria/polidipsia. Dosis tóxica: >20mg/kg signos leves, >40mg/kg severos, >60mg/kg convulsiones.",
        "category": "symptoms",
        "disease": "chocolate"
    },
    "chocolate_diagnosis": {
        "content": "DIAGNÓSTICO INTOXICACIÓN CHOCOLATE: Historia de ingesta, cálculo dosis ingerida (tipo chocolate + cantidad), signos clínicos, ECG (arritmias), química sanguínea (hipokalemia).",
        "category": "diagnosis",
        "disease": "chocolate"
    },
    "chocolate_treatment": {
        "content": """TRATAMIENTO INTOXICACIÓN CHOCOLATE:
        <2h ingesta: Inducir vómito (apomorfina 0.04mg/kg IV o conjuntival).
        Carbón activado: Exposición leve-moderada (<60mg/kg): 1-2g/kg PO dosis única. Severa (>60mg/kg): 1-2g/kg PO, puede repetirse cada 4-6h x 24h SOLO casos graves.
        Fluidoterapia IV para promover eliminación.
        Taquicardia severa: Propranolol 0.02-0.06mg/kg IV lento.
        Convulsiones: Diazepam 0.5-1mg/kg IV.
        Monitoreo ECG continuo si >40mg/kg ingerido.
        PRONÓSTICO: Excelente con tratamiento temprano.""",
        "category": "treatment",
        "disease": "chocolate"
    },

    # === RUPTURA LIGAMENTO CRUZADO ===
    "acl_overview": {
        "content": "RUPTURA LIGAMENTO CRUZADO CRANEAL: Causa más común de cojera miembro posterior en perros. Predisposición: razas grandes, sobrepeso, >5 años. 40-60% desarrollan ruptura contralateral.",
        "category": "overview",
        "disease": "acl"
    },
    "acl_symptoms": {
        "content": "SÍNTOMAS RUPTURA LCC: Cojera aguda o crónica progresiva, apoyo parcial o nulo del miembro afectado, inflamación articular (efusión), atrofia muscular del muslo si crónico, dolor a la manipulación.",
        "category": "symptoms",
        "disease": "acl"
    },
    "acl_diagnosis": {
        "content": "DIAGNÓSTICO RUPTURA LCC: Prueba cajón anterior positiva (desplazamiento craneal de tibia respecto a fémur), prueba de compresión tibial positiva, radiografías (efusión articular, signo de grasa infrapatelar desplazado, osteofitos si crónico, desplazamiento craneal de tibia).",
        "category": "diagnosis",
        "disease": "acl"
    },
    "acl_treatment": {
        "content": """TRATAMIENTO RUPTURA LCC:
        QUIRÚRGICO (recomendado >15kg): TPLO (gold standard razas grandes), TTA (alternativa efectiva), Extracapsular (perros <15kg).
        CONSERVADOR (<15kg o limitaciones económicas): Reposo estricto 8 semanas, AINES (Meloxicam 0.1mg/kg SID o Carprofeno 2.2mg/kg BID), fisioterapia, control peso, condroprotectores.
        PRONÓSTICO: Quirúrgico 85-90% función normal. Conservador: variable, osteoartritis inevitable.""",
        "category": "treatment",
        "disease": "acl"
    },

    # === PROTOCOLO ANESTESIA ===
    "anestesia_canino_sano": {
        "content": """PROTOCOLO ANESTESIA PERRO SANO:
        Pre-medicación: Acepromacina 0.02-0.05mg/kg + Morfina 0.2-0.5mg/kg IM.
        Inducción: Propofol 4-6mg/kg IV a efecto.
        Mantenimiento: Isoflurano 1-2% o Sevoflurano 3-4% end-tidal (profundidad quirúrgica; 2-3% puede ser adecuado con premedicación pesada).
        Analgesia: Meloxicam 0.2mg/kg IV/SC (DÍA 1 únicamente). Día 2+: Meloxicam 0.1mg/kg PO SID.""",
        "category": "protocol",
        "disease": "anesthesia"
    },
}

# Populate database with all knowledge base's chunks
def insert_diseases():
   """Store all Veterinary Diseases in ChromaDB"""
   logger.info("\nIndexing Veterinary Diseases...")

   # Grab all existing IDs from collection to avoid duplicates ahead
   existing_docs = collection.get()
   existing_ids = set(existing_docs.get("ids", []))

   for chunk_key, chunk_data in KNOWLEDGE_BASE.items():
      # Safe insertion
      try:
         # Skip document if it already exists (avoids duplicates)
         if chunk_key in existing_ids:
            logger.info(f"{chunk_key} already exists in collection, skipping...")
            continue # Jump to the next iteration (code below doesn't execute for this iteration)

         collection.add(
            ids=[chunk_key],
            documents=[f"passage: {chunk_data["content"]}"], # Used for embedding and search
            metadatas=[{"chunk_id": chunk_key, "chunk_content": chunk_data["content"], "chunk_category": chunk_data["category"], "chunk_disease": chunk_data["disease"]}] # Used for retrieval
         )
         logger.info(f"Stored: {chunk_key} → {chunk_data['content'][:50]}...")

      except Exception as e: # Catch any exception that happens during insertion
         logger.error(f"Error inserting {chunk_key}: {str(e)}")

# Function to compare query to collection's content and return matches
def query_diseases(query: str) -> str:
   """Query VectorDB for Veterinary Diseases"""
   logger.info(f"Searching collection for \"{query}\" matches")

   try:
      # Vector similarity search
      # Compares query embedding to every chunks content embedding
      # Returns most similar chunks content
      results = collection.query(
         query_texts=[f"query: {query}"], # Used for embedding and search
         n_results=5, # Return top 5 results, even if not relevant (adjustable)
         include=["metadatas", "distances"] # Used for retrieval (id's by default, metadatas and distances)
      )

      # Check for valid results
      if not results or not results.get("metadatas") or not results["metadatas"][0]:
         return "No relevant diseases found."
      
      # Log all results
      logger.info(f"Top 5 results:")
      for i, (metadata, distance) in enumerate(zip(results["metadatas"][0], results["distances"][0])):
         chunk_id = metadata.get("chunk_id", "unknown")
         chunk_disease = metadata.get("chunk_disease", "unknown")
         chunk_category = metadata.get("chunk_category", "unknown")
         logger.info(f" {i+1}. [{chunk_disease}/{chunk_category}] {chunk_id}: {distance:.3f}")
      
      # Filter results by distance threshold
      filtered_results = []
      
      for i, metadata in enumerate(results["metadatas"][0]):
         distance = results["distances"][0][i]
         if distance < 0.45: # Adjustable threshold
            filtered_results.append(metadata.get("chunk_content", ""))
            logger.info(f"   ✓ Using result {i+1}")

      # Format response
      # If nothing passed, return best unfiltered match
      if not filtered_results and results["metadatas"][0]:
            best_match = results["metadatas"][0][0].get("chunk_content", "")
            return f"Se encontró información potencialmente relacionada, pero con bajos niveles de confianza:\n\n{best_match}"
      # If only one result passed, return it
      if len(filtered_results) == 1:
            return filtered_results[0]
      # If multiple results passed, return them
      if filtered_results:
            summary = "\n\n".join([f"• {content}" for content in filtered_results])
            return f"Se encontró información relevante:\n\n{summary}"
         
   except Exception as e: # Catch any errors during search
      logger.error(f"Error querying collection: {str(e)}")
      return "An error occured while querying collection"

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
#    reset_collection()

   # Check collection
#    print(collection.get())

   # Create collection and index chunks
#    insert_diseases()

   # TESTING QUERIES
   test_queries = [
      "¿Qué es el parvovirus canino?",
      "Perro con vómitos severos y diarrea con sangre",
      "Perro con picazón intensa en patas y orejas",
      "Mi perro comió chocolate, ¿es peligroso?",
      "Emergencia: perro con abdomen hinchado y en shock",
      "Perro con cojera en pata trasera que no apoya",
      "Gato adulto con vómitos y mal aliento",
   ]

   for query in test_queries:
      print(f"\n{'='*60}")
      print(f"QUERY: {query}")
      response = query_diseases(query)
      print(f"\nRESPONSE:\n{response}\n")