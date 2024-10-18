import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEndpoint
from langchain.chains import LLMChain
from langchain_core.prompts import PromptTemplate

# Cargar las variables de entorno desde el archivo .env
load_dotenv()

# Obtener el token desde las variables de entorno
HUGGINGFACEHUB_API_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN2")

# Asegurarse de que el token fue cargado correctamente
if not HUGGINGFACEHUB_API_TOKEN:
    raise ValueError("El token de Hugging Face no se ha encontrado en las variables de entorno.")

# Definir la ruta del archivo de texto que contiene la historia
file_path = r"stories\cuento_resultados_scraping_20241017_125209.txt"

# Leer el contenido del archivo .txt
with open(file_path, "r", encoding="utf-8") as file:
    story_content = file.read()

# Definir el template del prompt para establecer el comportamiento del sistema
template = """
Analiza el siguiente cuento para niños y genera 6 prompts para crear imágenes. Estos prompts deben:

1. Seguir la secuencia cronológica del cuento.
2. Representar los 6 puntos clave más importantes de la narrativa.
3. Mantener coherencia en el diseño de personajes, paisajes y elementos a lo largo de la historia.
4. Ser altamente descriptivos, incluyendo detalles sobre:
   - Apariencia física de los personajes (altura, complexión, color de pelo, ropa, etc.)
   - Expresiones faciales y lenguaje corporal
   - Detalles del entorno (formas, colores, texturas de paisajes y objetos)
   - Iluminación y atmósfera de la escena
   - Posición y acción de los personajes en la escena

**Importante**: Responde **únicamente** con los 6 prompts numerados. **No** añadas notas, explicaciones ni ejemplos adicionales.

Cuento:
{story}

Respuesta:

"""

# Crear el template con el contenido del cuento
prompt = PromptTemplate.from_template(template)

# Definir el modelo (meta-llama/Llama-3.2-3B-Instruct)
repo_id = "meta-llama/Llama-3.2-3B-Instruct"

# Configurar el endpoint de Hugging Face, pasando max_length en model_kwargs
llm = HuggingFaceEndpoint(
    repo_id=repo_id,
    # model_kwargs={"max_length": 512},  # Pasa max_length como model_kwargs
    temperature=0.8,
    top_p=0.95,
    huggingfacehub_api_token=HUGGINGFACEHUB_API_TOKEN
)

# Crear la cadena (pipeline) LLM
llm_chain = prompt | llm

# Ejecutar la cadena con el cuento proporcionado
response = llm_chain.invoke({"story": story_content})

# Mostrar la respuesta del modelo (los 6 prompts generados)
print(response)