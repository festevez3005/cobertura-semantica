import streamlit as st
import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup
from io import StringIO
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA 
import matplotlib.pyplot as plt
from typing import List, Dict

# --- CONFIGURATION & MODEL LOADING ---

@st.cache_resource 
def load_model():
    """Loads the multilingual SBERT model (for English/Spanish) only once."""
    st.info("Cargando el motor semántico multilingüe... (Esto puede tardar unos segundos la primera vez)")
    return SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

# --- DATA PROCESSING UTILITIES ---

def fetch_content_from_url(url: str) -> str:
    """Fetches and cleans text content from a given URL."""
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status() 
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Extract text from common elements
        text_parts = soup.find_all(['p', 'h1', 'h2', 'h3', 'li'])
        content = ' '.join([part.get_text() for part in text_parts])
        
        clean_content = ' '.join(content.split()) 
        return clean_content[:6000] # Increased limit for document analysis
    except requests.exceptions.RequestException as e:
        return f"Error al obtener la URL: {e}"
    except Exception as e:
        return f"Ocurrió un error inesperado: {e}"

def chunk_document(text: str, max_chunk_length: int = 500) -> List[str]:
    """Divides the document into manageable chunks (paragraphs) for vectorization."""
    # (Using the robust chunking logic developed previously)
    chunks = []
    current_chunk = ""
    paragraphs = text.split('\n\n')
    
    for paragraph in paragraphs:
        paragraph = paragraph.strip()
        if not paragraph: continue
            
        if len(current_chunk) + len(paragraph) > max_chunk_length and current_chunk:
            chunks.append(current_chunk.strip())
            current_chunk = paragraph
        else:
            current_chunk += " " + paragraph
            
    if current_chunk:
        chunks.append(current_chunk.strip())
        
    if len(chunks) == 1 and len(chunks[0]) > 800:
        chunks = [text[i:i + max_chunk_length] for i in range(0, len(text), max_chunk_length)]

    return [c.replace('\n', ' ') for c in chunks if c.strip()]

# --- CORE SEMANTIC ANALYSIS FUNCTION ---

def run_semantic_audit(document_url: str, gsc_queries: List[str], model: SentenceTransformer) -> Dict:
    """Performs the full audit by comparing GSC queries to the document content."""
    
    # 1. Scrape Content
    st.markdown("### 1. Extrayendo Contenido (Scraping)")
    document_content = fetch_content_from_url(document_url)
    
    if document_content.startswith("Error"):
        st.error(f"Error en el Scraping: {document_content}")
        return {"error": True}

    st.success(f"Contenido extraído de **{document_url}** ({len(document_content)} caracteres).")

    # 2. Chunking
    chunks = chunk_document(document_content)
    if not chunks:
        st.error("No se pudo segmentar el documento. Intenta con un texto más largo.")
        return {"error": True}
        
    st.success(f"Documento dividido en **{len(chunks)}** segmentos semánticos.")

    # 3. Vectorización
    st.markdown("### 3. Vectorizando y Calculando Relevancia")
    
    # Prepare all texts for encoding: all chunks + all queries
    texts_to_embed = chunks + gsc_queries
    
    with st.spinner('Generando todos los vectores (esto puede tardar según el número de queries)...'):
        embeddings_full = model.encode(texts_to_embed)

    chunk_embeddings = embeddings_full[:len(chunks)]
    query_embeddings = embeddings_full[len(chunks):]
    
    results = []

    # Calculate similarity for each query against ALL chunks
    for i, query in enumerate(gsc_queries):
        query_emb = query_embeddings[i].reshape(1, -1)
        
        # Similarity array: query vs. all chunks
        similarity_scores = cosine_similarity(query_emb, chunk_embeddings)[0]
        
        max_similarity_score = np.max(similarity_scores)
        best_chunk_index = np.argmax(similarity_scores)
        best_chunk = chunks[best_chunk_index]
        
        # Store results
        results.append({
            "Query": query,
            "Relevance_Score": max_similarity_score,
            "Best_Chunk_Preview": best_chunk[:100] + "..."
        })
        
    return {"error": False, "results": results, "chunks": chunks}

# --- STREAMLIT APP LAYOUT ---

def main():
    st.set_page_config(page_title="Auditor Semántico Local", layout="wide")
    st.title("🔎 Auditoría Semántica de Cobertura (GSC vs. Contenido)")
    st.markdown("""
    Esta herramienta compara las *queries* reales de Google Search Console con el contenido de una URL específica para evaluar la **cobertura semántica** de tu sitio.
    
    **El motor utiliza:** *Chunking* (segmentación de contenido) y Similitud del Coseno.
    """)
    
    model = load_model()
    
    st.subheader("Configuración de la Auditoría")
    
    col1, col2 = st.columns(2)
    
    with col1:
        document_url = st.text_input(
            "1. URL del Documento a Auditar (Contenido)",
            "https://www.argentina.gob.ar/justicia/convosenlaweb/situaciones/que-es-la-inteligencia-artificial"
        )
    
    with col2:
        st.markdown("2. Queries de GSC (Simulación API)")
        
        # NUEVO: Carga de archivo CSV
        uploaded_file = st.file_uploader(
            "Sube el archivo CSV exportado desde Google Search Console", 
            type="csv"
        )
        
        gsc_queries = []
        
        if uploaded_file is not None:
            try:
                # Leer el archivo CSV
                data = pd.read_csv(uploaded_file)
                
                # Intentar encontrar la columna de queries (puede llamarse 'Query', 'Consulta', etc.)
                query_cols = [col for col in data.columns if 'query' in col.lower() or 'consulta' in col.lower()]
                
                if query_cols:
                    # Usar la primera columna encontrada
                    gsc_queries = data[query_cols[0]].astype(str).tolist()
                    st.success(f"Cargadas {len(gsc_queries)} queries de la columna '{query_cols[0]}'.")
                else:
                    st.error("Error: No se encontró una columna con el nombre 'Query' o 'Consulta'.")
            except Exception as e:
                st.error(f"Error al leer el archivo CSV: {e}")
        
        # Instrucciones de formato
        st.info("💡 **Instrucciones:** En GSC, exporta la tabla de 'Consultas' a CSV. Asegúrate de que la columna se llame 'Query' o 'Consulta'.")

    st.markdown("---")
    
    if st.button("🚀 Iniciar Auditoría Semántica", type="primary"):
        if not document_url or not gsc_queries:
            st.error("Por favor, proporciona una URL y sube un archivo CSV de queries válido.")
            return

        with st.container():
            st.header("Resultados de la Auditoría")
            audit_data = run_semantic_audit(document_url, gsc_queries, model)
            
            if audit_data.get("error"):
                return

            df = pd.DataFrame(audit_data['results'])
            
            # --- FINAL REPORT TABLE ---
            st.subheader("Informe de Similitud Semántica (GSC Queries)")
            
            # Add Relevance Assessment column
            df['Assessment'] = df['Relevance_Score'].apply(lambda x: '✅ Excelente' if x >= 0.7 else ('⚠️ Moderada' if x >= 0.4 else '❌ Baja'))
            
            st.dataframe(
                df.sort_values(by='Relevance_Score', ascending=False),
                use_container_width=True,
                column_config={
                    "Relevance_Score": st.column_config.ProgressColumn(
                        "Relevance Score",
                        help="Similitud del Coseno con el segmento más relevante (Max: 1.0)",
                        format="%.3f",
                        min_value=0,
                        max_value=1
                    ),
                    "Best_Chunk_Preview": st.column_config.TextColumn(
                        "Mejor Segmento (Preview)",
                        help="Inicio del fragmento de contenido con el Score más alto."
                    )
                }
            )

            st.markdown("---")
            
            # --- VISUALIZATION: PCA of Top 3 Queries ---
            st.subheader("Visualización: Cobertura de las Top Queries")
            
            top_queries = df.nlargest(3, 'Relevance_Score')
            
            if len(top_queries) >= 1:
                st.markdown("Proyectando en 2D la relación entre el contenido y las 3 Queries más relevantes.")
                
                # Get embeddings for the top 3 queries and ALL chunks
                queries_to_plot = top_queries['Query'].tolist()
                chunks_to_plot = audit_data['chunks']
                texts_to_plot = chunks_to_plot + queries_to_plot

                embeddings_plot = model.encode(texts_to_plot)
                
                # Apply PCA
                pca = PCA(n_components=2)
                vectors_2d = pca.fit_transform(embeddings_plot)

                fig, ax = plt.subplots(figsize=(10, 6))
                
                # Plot Chunks (as background context)
                ax.scatter(vectors_2d[:len(chunks_to_plot), 0], vectors_2d[:len(chunks_to_plot), 1], 
                           label='Segmentos del Documento', color='gray', alpha=0.3, s=50)

                # Plot Top Queries (as specific targets)
                query_vectors_2d = vectors_2d[len(chunks_to_plot):]
                
                for idx, row in top_queries.iterrows():
                    q_idx = queries_to_plot.index(row['Query'])
                    ax.scatter(query_vectors_2d[q_idx, 0], query_vectors_2d[q_idx, 1], 
                               label=f"Query: {row['Query']}", 
                               s=150, 
                               marker='*', 
                               c=plt.cm.viridis(q_idx / len(top_queries))) # Unique color

                ax.set_title('Espacio Semántico: Queries de GSC vs. Segmentos del Documento (PCA)')
                ax.legend()
                ax.grid(True)
                st.pyplot(fig)


    # ----------------------------------------------------
    # PIE DE PÁGINA Y CONTACTO
    # ----------------------------------------------------
    st.markdown("---") 
    st.markdown("""
    ### Información y Contacto 🤝
    ✨ Esta herramienta fue creada con **fines educativos y de asistencia a profesionales**.

    💌 **¿Te sirvió? ¿Tenés alguna sugerencia? ¿Querés charlar sobre SEO, comunicación digital o IA aplicada?** Escribinos a: **`hola@crawla.agency`**

    🌐 Conectá con Crawla en **[LinkedIn](https://www.linkedin.com/company/crawla-agency/)**
    """)

if __name__ == "__main__":
    main()
