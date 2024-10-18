'''Usar info de:
https://elpais.com/clima-y-medio-ambiente/cambio-climatico/
https://www.nationalgeographicla.com/medio-ambiente
'''

import requests
from bs4 import BeautifulSoup
import re
from datetime import datetime
import os

def limpiar_texto(texto):
    # Eliminar espacios en blanco extra y saltos de línea
    texto = re.sub(r'\s+', ' ', texto).strip()
    return texto

def extraer_texto_de_url(url):
    try:
        # Realizar la solicitud HTTP
        response = requests.get(url)
        response.raise_for_status()  # Lanzar una excepción para códigos de error HTTP
        
        # Crear objeto BeautifulSoup
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Eliminar scripts y estilos
        for script in soup(["script", "style"]):
            script.decompose()
        
        # Obtener el texto
        texto = soup.get_text()
        
        # Limpiar el texto
        texto_limpio = limpiar_texto(texto)
        
        return texto_limpio
    
    except requests.RequestException as e:
        return f"Error al procesar {url}: {str(e)}"

def procesar_urls(urls):
    resultados = {}
    for url in urls:
        texto = extraer_texto_de_url(url)
        resultados[url] = texto
    return resultados

def guardar_resultados_en_archivo(resultados):
    # Crear la carpeta 'documents' si no existe
    carpeta = 'documents'
    if not os.path.exists(carpeta):
        os.makedirs(carpeta)
    
    # Generar un nombre de archivo único con la fecha y hora actual
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    nombre_archivo = f"resultados_scraping_{timestamp}.txt"
    
    # Crear la ruta completa del archivo
    ruta_completa = os.path.join(carpeta, nombre_archivo)
    
    with open(ruta_completa, 'w', encoding='utf-8') as file:
        for url, texto in resultados.items():
            file.write(f"Texto extraído de {url}:\n")
            file.write(texto)
            file.write("\n\n" + "="*50 + "\n\n")
    
    return ruta_completa

# Ejemplo de uso
if __name__ == "__main__":
    # Puedes pasar una lista con una o más URLs
    urls = [
        "https://elpais.com/clima-y-medio-ambiente/2024-10-09/un-grupo-de-cientificos-alerta-contra-el-exceso-de-confianza-en-la-captura-del-co-para-revertir-el-calentamiento.html"
    ]
    
    resultados = procesar_urls(urls)
    ruta_archivo = guardar_resultados_en_archivo(resultados)
    
    print(f"Los resultados se han guardado en el archivo: {ruta_archivo}")