import streamlit as st
import pandas as pd

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
        st.write("### Primer DataFrame")
        st.write(f"Longitud del DataFrame: {len(df1)}")
        st.write("Columnas:", df1.columns.tolist())
    
    with col2:
        st.write("### Segundo DataFrame")
        st.write(f"Longitud del DataFrame: {len(df2)}")
        st.write("Columnas:", df2.columns.tolist())

    if df1.equals(df2):
        st.success("Los archivos son idénticos!")
    else:
        st.warning("Los archivos tienen diferencias.")

def main():
    st.title('Comparador de Archivos')
    
    col1, col2 = st.columns(2)
    
    df1_preview, df2_preview = None, None

    with col1:
        file1 = st.file_uploader("Carga el primer archivo", type=['csv', 'txt'], key='file1')
        delimiter1 = st.selectbox("Selecciona el delimitador para el primer archivo", 
                                  options=[',', '|', ';', '\t'], index=0, key='delim1')
        encoding1 = st.selectbox("Selecciona la codificación para el primer archivo",
                                 options=['utf-8', 'latin-1'], index=0, key='enc1')
        
        if file1 is not None:
            df1_preview = load_data_with_progress(file1, delimiter1, encoding1)
            if df1_preview is not None:
                st.write("### Previsualización del primer archivo")
                st.dataframe(df1_preview.head())

    with col2:
        file2 = st.file_uploader("Carga el segundo archivo", type=['csv', 'txt'], key='file2')
        delimiter2 = st.selectbox("Selecciona el delimitador para el segundo archivo", 
                                  options=[',', '|', ';', '\t'], index=0, key='delim2')
        encoding2 = st.selectbox("Selecciona la codificación para el segundo archivo",
                                 options=['utf-8', 'latin-1'], index=0, key='enc2')
        
        if file2 is not None:
            df2_preview = load_data_with_progress(file2, delimiter2, encoding2)
            if df2_preview is not None:
                st.write("### Previsualización del segundo archivo")
                st.dataframe(df2_preview.head())
    
    if st.button("Comparar archivos"):
        if df1_preview is not None and df2_preview is not None:
            compare_dataframes(df1_preview, df2_preview)
        else:
            st.error("Por favor, carga ambos archivos para proceder con la comparación.")

if __name__ == "__main__":
    main()
