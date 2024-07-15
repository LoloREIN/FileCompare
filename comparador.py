import streamlit as st
import pandas as pd
from difflib import SequenceMatcher
import boto3
import os
from datetime import datetime

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_aws import ChatBedrock
from botocore.config import Config
import langchain.globals as lg

# Configuración de Amazon Bedrock
lg.set_verbose(False)

aws_access_key_id = os.getenv("AWS_ACCESS_KEY_ID")
aws_secret_access_key = os.getenv("AWS_SECRET_ACCESS_KEY")
aws_default_region = os.getenv("AWS_DEFAULT_REGION")

retry_config = Config(region_name="us-east-1")
boto3_bedrock_runtime = boto3.client("bedrock-runtime", config=retry_config)

model_id = "anthropic.claude-3-5-sonnet-20240620-v1:0"

model_kwargs = {
    "max_tokens": 1000000,
    "temperature": 0.9,
    "top_k": 250,
    "top_p": 1,
    "stop_sequences": ["\n\nHuman"],
}

model = ChatBedrock(
    client=boto3_bedrock_runtime,
    model_id=model_id,
    model_kwargs=model_kwargs,
)

PROMPT_TEMPLATE = """
    Human: Eres un experto en analizar diferencias entre filas de datos.
    Por favor, compara las siguientes dos filas y explica las diferencias:
    
    Fila 1: {row1}
    Fila 2: {row2}

    Assistant:
    """

def get_difference_explanation(row1, row2):
    formatted_prompt = PROMPT_TEMPLATE.format(row1=row1, row2=row2)
    messages = [("human", formatted_prompt)]
    prompt_template = ChatPromptTemplate.from_messages(messages)
    chain = prompt_template | model | StrOutputParser()

    response = chain.invoke({
        "row1": row1,
        "row2": row2,
    })

    return response.strip()

def load_data_with_progress(uploaded_file, delimiter, encoding):
    if uploaded_file is not None:
        if uploaded_file.name.endswith(('.csv', '.txt')):
            try:
                with st.spinner('Cargando archivo...'):
                    sample = uploaded_file.read(1024).decode(encoding)
                    uploaded_file.seek(0)

                    if not sample:
                        st.error("El archivo cargado está vacío.")
                        return None

                    if delimiter not in sample:
                        st.error("El delimitador seleccionado no coincide con el formato del archivo.")
                        return None

                    initial_df = pd.read_csv(uploaded_file, delimiter=delimiter, encoding=encoding, nrows=10)
                    uploaded_file.seek(0)

                    if initial_df.empty or initial_df.columns.empty:
                        st.error("El archivo cargado no tiene datos o columnas.")
                        return None

                    file_size = uploaded_file.size

                    progress_bar = st.progress(0)
                    df = pd.DataFrame()

                    for i, chunk in enumerate(pd.read_csv(uploaded_file, delimiter=delimiter, encoding=encoding, chunksize=1000)):
                        df = pd.concat([df, chunk])
                        progress = min((i + 1) * chunk.memory_usage(deep=True).sum() / file_size, 1.0)
                        progress_bar.progress(progress)

                    return df
            except Exception as e:
                st.error(f"Error al procesar el archivo: {e}")
        else:
            st.error("Formato de archivo no soportado. Por favor, carga un archivo CSV o TXT.")
    return None

def compare_dataframes(df1, df2):
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("### Archivo del Proceso")
        st.write(f"Longitud del DataFrame: {len(df1)}")
        st.write(f"Número de columnas: {len(df1.columns)}")
        with st.expander("Ver columnas del Archivo del Proceso"):
            st.write(df1.columns.tolist())
    
    with col2:
        st.write("### Archivo de Control")
        st.write(f"Longitud del DataFrame: {len(df2)}")
        st.write(f"Número de columnas: {len(df2.columns)}")
        with st.expander("Ver columnas del Archivo de Control"):
            st.write(df2.columns.tolist())

    if df1.equals(df2):
        st.success("Los archivos son idénticos!")
    else:
        st.warning(f"""
                    Los archivos tienen diferencias.\n
                    Te faltan [{len(df2) - len(df1)}] filas.\n
                    Te faltan [{len(df2.columns) - len(df1.columns)}] columnas.
                    """)

        common_rows = df1[df1.apply(tuple, axis=1).isin(df2.apply(tuple, axis=1))]
        
        with st.expander("Ver filas comunes"):
            st.write("### Filas comunes en ambos archivos")
            st.dataframe(common_rows)

        differences = df1.merge(df2, indicator=True, how='outer')

        differences['APARECE EN'] = differences['_merge'].map({'left_only': 'Archivo del Proceso', 
                                                           'right_only': 'Archivo de Control', 
                                                           'both': 'Ambos'})
        differences = differences[differences['_merge'] != 'both']

        with st.expander("Ver diferencias"):
            st.write("### Diferencias entre los archivos")
            st.dataframe(differences.drop(columns=['_merge']))

        only_in_process = df1[~df1.apply(tuple, axis=1).isin(df2.apply(tuple, axis=1))]
        only_in_control = df2[~df2.apply(tuple, axis=1).isin(df1.apply(tuple, axis=1))]

        with st.expander("Ver filas solo en Archivo del Proceso"):
            st.write("### Filas solo en el Archivo del Proceso")
            st.dataframe(only_in_process)

        with st.expander("Ver filas solo en Archivo de Control"):
            st.write("### Filas solo en el Archivo de Control")
            st.dataframe(only_in_control)

        with st.expander("Ver filas similares con diferencias"):
            similar_rows = find_similar_rows(only_in_process, only_in_control)
            st.write("### Filas similares con diferencias")
            for row1, row2 in similar_rows:
                st.write(f"Proceso: {row1}")
                st.write(f"Control: {row2}")
                diff_desc = get_difference_explanation(row1, row2)
                st.write(f"Diferencias: {diff_desc}")

def find_similar_rows(df1, df2, threshold=0.8):
    similar_rows = []
    for _, row1 in df1.iterrows():
        for _, row2 in df2.iterrows():
            if SequenceMatcher(None, str(row1), str(row2)).ratio() > threshold:
                similar_rows.append((row1, row2))
    return similar_rows

def main():
    st.title('Comparador de Archivos')
    
    col1, col2 = st.columns(2)
    
    df1_preview, df2_preview = None, None

    with col1:
        file1 = st.file_uploader("Carga el Archivo del Proceso", type=['csv', 'txt'], key='file1')
        delimiter1 = st.selectbox("Selecciona el delimitador para el Archivo del Proceso", 
                                  options=[',', '|', ';', '\t'], index=0, key='delim1')
        encoding1 = st.selectbox("Selecciona la codificación para el Archivo del Proceso",
                                 options=['utf-8', 'latin-1'], index=0, key='enc1')
        
        if file1 is not None:
            df1_preview = load_data_with_progress(file1, delimiter1, encoding1)
            if df1_preview is not None:
                st.write("### Previsualización del Archivo del Proceso")
                st.dataframe(df1_preview.head())

    with col2:
        file2 = st.file_uploader("Carga el Archivo de Control", type=['csv', 'txt'], key='file2')
        delimiter2 = st.selectbox("Selecciona el delimitador para el Archivo de Control", 
                                  options=[',', '|', ';', '\t'], index=0, key='delim2')
        encoding2 = st.selectbox("Selecciona la codificación para el Archivo de Control",
                                 options=['utf-8', 'latin-1', 'cp1252'], index=0, key='enc2')
        
        if file2 is not None:
            df2_preview = load_data_with_progress(file2, delimiter2, encoding2)
            if df2_preview is not None:
                st.write("### Previsualización del Archivo de Control")
                st.dataframe(df2_preview.head())
    
    if st.button("Comparar archivos"):
        if df1_preview is not None and df2_preview is not None:
            compare_dataframes(df1_preview, df2_preview)
        else:
            st.error("Por favor, carga ambos archivos para proceder con la comparación.")

if __name__ == "__main__":
    main()
