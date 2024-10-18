import os
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain_huggingface import HuggingFaceEndpoint
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.summarize import load_summarize_chain
from langchain.prompts import PromptTemplate
from langchain.schema import Document

# Cargar variables de entorno
load_dotenv()

# Obtener el token de Hugging Face
HUGGINGFACEHUB_API_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")
if not HUGGINGFACEHUB_API_TOKEN:
    raise ValueError("El token de Hugging Face no se ha encontrado en las variables de entorno.")

# Configurar el modelo de lenguaje de Hugging Face
repo_id = "meta-llama/Llama-3.2-3B-Instruct"
llm = HuggingFaceEndpoint(
    repo_id=repo_id,
    # model_kwargs={"max_length": 4096},
    temperature=0.8,
    top_p=0.95,
    huggingfacehub_api_token=HUGGINGFACEHUB_API_TOKEN
)

def clean_text(text):
    # Implementa la lógica de limpieza de texto aquí
    return text.strip()

def read_pdf(file_path):
    reader = PdfReader(file_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return clean_text(text)

def read_txt(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return clean_text(file.read())

def split_text(text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", " ", ""]
    )
    return [Document(page_content=t) for t in text_splitter.split_text(text)]

def create_story(docs):
    map_prompt_template = """
Proporciona un resumen detallado y coherente del siguiente fragmento de texto. 
Asegúrate de:
1. Resumir el contenido de cada sección manteniendo los detalles más importantes.
2. No omitir ningún aspecto fundamental del texto.
3. Mantener la terminología y el tono del documento original.
4. Organizar el resumen en puntos claros y concisos.

Texto a resumir:
{text}

Resumen detallado:
"""
    map_prompt = PromptTemplate(template=map_prompt_template, input_variables=["text"])

    combine_prompt_template = """
Basándote en el siguiente resumen detallado de un texto sobre un tema ambiental, crea un cuento atractivo y educativo para niños. Tu historia debe seguir estas pautas:

1. Comienza con "Había una vez..." y utiliza un lenguaje sencillo y descriptivo apropiado para niños de 6-10 años.
2. Crea personajes originales inspirados en el tema del resumen. Pueden ser animales, niños o elementos de la naturaleza. Dales nombres y características únicas.
3. Establece un escenario imaginativo que refleje el entorno descrito en el resumen.
4. Presenta el problema ambiental de manera simplificada y comprensible para los niños, basándote en la información del resumen.
5. Muestra cómo los personajes trabajan juntos para abordar el problema, incorporando datos o conceptos del resumen de forma creativa y adecuada para niños.
6. Incluye un momento de descubrimiento donde los personajes encuentran una solución inspirada en la información del resumen.
7. Describe el resultado positivo y cómo mejora la situación para los personajes y su entorno.
8. Concluye con una moraleja clara que anime a los niños a cuidar el medio ambiente, relacionada con el tema principal del resumen.

Resumen detallado:
{text}

Cuento para niños:
"""
    combine_prompt = PromptTemplate(template=combine_prompt_template, input_variables=["text"])

    chain = load_summarize_chain(
        llm,
        chain_type="map_reduce",
        map_prompt=map_prompt,
        combine_prompt=combine_prompt,
        verbose=True
    )

    return chain({"input_documents": docs})

def process_documents(input_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    files_processed = False

    for filename in os.listdir(input_folder):
        file_path = os.path.join(input_folder, filename)

        if filename.endswith('.pdf'):
            text = read_pdf(file_path)
        elif filename.endswith('.txt'):
            text = read_txt(file_path)
        else:
            continue

        docs = split_text(text)
        story = create_story(docs)

        output_filename = f"cuento_{os.path.splitext(filename)[0]}.txt"
        output_path = os.path.join(output_folder, output_filename)

        with open(output_path, 'w', encoding='utf-8') as file:
            file.write(story['output_text'])

        print(f"Cuento guardado: {output_path}")
        files_processed = True

    if not files_processed:
        print("No se encontraron archivos PDF o TXT para procesar.")

if __name__ == "__main__":
    input_folder = "documents"
    output_folder = "stories"
    process_documents(input_folder, output_folder)