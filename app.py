import streamlit as st
import pandas as pd
import joblib
import numpy as np
from calendar import month_abbr

# --- CONFIGURAÇÃO DA PÁGINA ---
st.set_page_config(page_title="Previsão de Conversão", layout="wide", initial_sidebar_state="auto")

# --- CARREGAR O MODELO ---
try:
    loaded_pipe = joblib.load('ecommerce_conversion_pipeline.joblib')
except FileNotFoundError:
    st.error("ERRO: O arquivo 'ecommerce_conversion_pipeline.joblib' não foi encontrado. Certifique-se de que ele está na mesma pasta que o arquivo app.py.")
    st.stop()

# --- INTERFACE DO USUÁRIO ---
st.title('Previsão de Conversão de E-commerce')
st.write("Esta ferramenta usa um modelo de Machine Learning para prever a probabilidade de um visitante realizar uma compra.")

# --- GUIA DE INTERPRETAÇÃO (DENTRO DE UM EXPANDER) ---
with st.expander("Clique aqui para ver o Guia de Interpretação dos Campos"):
    st.markdown("""
    Este guia explica o que cada campo significa e como interpretá-lo.

    ### Comportamento
    *   **Páginas Administrativas (`Administrative`):** Páginas de login, perfil, histórico de pedidos.
    *   **Páginas Informativas (`Informational`):** Páginas como "Sobre Nós", "Contato", "FAQ".
    *   **Páginas de Produto (`ProductRelated`):** Páginas de produtos, categorias, busca.

    ### Tempo Gasto (segundos)
    *   **Duração em Páginas...:** Quanto tempo o cliente gastou em cada tipo de página.

    ### Métricas da Sessão
    *   **Taxa de Rejeição (`BounceRates`):** % de visitantes que entram e saem sem clicar em nada.
    *   **Taxa de Saída (`ExitRates`):** % de vezes que uma página foi a última da sessão.
    *   **Valor da Página (`PageValues`):** Valor médio de uma página visitada antes de uma compra.

    ### Contexto Temporal
    *   **Mês (`Month`):** A sazonalidade é crucial (ex: `Nov` para Black Friday).
    *   **Proximidade de Dia Especial (`SpecialDay`):** Valor de 0 (longe) a 1 (perto) de um feriado.

    ### Informações do Visitante e Técnicas
    *   **Tipo de Visitante, Tipo de Tráfego, Região, Sistema Operacional e Navegador:** Categorias que descrevem o perfil da visita.
    """)

st.divider()

# --- CAMPOS DE ENTRADA ---
st.header("Insira os Dados da Sessão para Previsão")

col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("Comportamento")
    administrative = st.number_input('Páginas Administrativas Visitadas', min_value=0, value=0)
    informational = st.number_input('Páginas Informativas Visitadas', min_value=0, value=0)
    product_related = st.number_input('Páginas de Produto Visitadas', min_value=0, value=1)

with col2:
    st.subheader("Tempo Gasto (segundos)")
    administrative_duration = st.number_input('Duração em Págs. Administrativas (s)', min_value=0.0, value=0.0, format="%.2f")
    informational_duration = st.number_input('Duração em Págs. Informativas (s)', min_value=0.0, value=0.0, format="%.2f")
    product_related_duration = st.number_input('Duração em Págs. de Produto (s)', min_value=0.0, value=60.0, format="%.2f")

with col3:
    st.subheader("Métricas da Sessão")
    bounce_rates = st.number_input('Taxa de Rejeição', min_value=0.0, max_value=1.0, value=0.02, format="%.4f")
    exit_rates = st.number_input('Taxa de Saída', min_value=0.0, max_value=1.0, value=0.04, format="%.4f")
    page_values = st.number_input('Valor da Página', min_value=0.0, value=0.0, format="%.2f")

st.divider()

col4, col5, col6 = st.columns(3)

with col4:
    st.subheader("Contexto Temporal")
    month_list_display = ['Feb', 'Mar', 'May', 'June', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    month = st.selectbox('Mês', month_list_display)
    special_day = st.slider('Proximidade de Dia Especial', 0.0, 1.0, 0.0)
    weekend = st.checkbox('É Fim de Semana?')

with col5:
    st.subheader("Informações do Visitante")
    visitor_type = st.selectbox('Tipo de Visitante', ['Returning_Visitor', 'New_Visitor', 'Other'])
    traffic_type = st.selectbox('Tipo de Tráfego', list(range(1, 21)))
    region = st.selectbox('Região', list(range(1, 10)))

with col6:
    st.subheader("Informações Técnicas")
    operating_systems = st.selectbox('Sistema Operacional', list(range(1, 9)))
    browser = st.selectbox('Navegador', list(range(1, 14)))

st.divider()

# --- BOTÃO DE PREVISÃO E RESULTADO ---
if st.button('Prever Probabilidade de Compra', type="primary", use_container_width=True):

    new_data = pd.DataFrame({
        'Administrative': [administrative], 'Administrative_Duration': [administrative_duration],
        'Informational': [informational], 'Informational_Duration': [informational_duration],
        'ProductRelated': [product_related], 'ProductRelated_Duration': [product_related_duration],
        'BounceRates': [bounce_rates], 'ExitRates': [exit_rates],
        'PageValues': [page_values], 'SpecialDay': [special_day],
        'Month': [month], 'OperatingSystems': [operating_systems], 'Browser': [browser], 
        'Region': [region], 'TrafficType': [traffic_type], 'VisitorType': [visitor_type], 'Weekend': [weekend]
    })

    month_map_notebook = {abbr: idx for idx, abbr in enumerate(month_abbr) if abbr}
    try:
        month_abbr_map = {'June': 'Jun'}
        new_data['Month_Abbr'] = new_data['Month'].map(month_abbr_map)
        new_data['Month_Num'] = new_data['Month_Abbr'].map(month_map_notebook)
    except:
        new_data['Month_Num'] = new_data['Month'].map(month_map_notebook)

    new_data['TotalPageVisits'] = new_data['Administrative'] + new_data['Informational'] + new_data['ProductRelated']
    new_data['TotalDuration'] = new_data['Administrative_Duration'] + new_data['Informational_Duration'] + new_data['ProductRelated_Duration']
    new_data['PagesPerMinute'] = new_data['TotalPageVisits'] / np.where(new_data['TotalDuration'] > 0, new_data['TotalDuration'] / 60, 1)
    new_data['ProductEngagement'] = np.where(new_data['TotalPageVisits'] > 0, new_data['ProductRelated'] / new_data['TotalPageVisits'], 0)

    with st.spinner("Analisando os dados..."):
        prob = loaded_pipe.predict_proba(new_data)[0][1]
        prediction = loaded_pipe.predict(new_data)[0]

    st.subheader('Veredito da Previsão:')

    if prediction == 1:
        st.success('Previsão: O usuário VAI COMPRAR.')
        st.info(f"O modelo está **{prob:.1%}** confiante nesta previsão.")
    else:
        st.error('Previsão: O usuário NÃO VAI COMPRAR.')
        st.info(f"A probabilidade de compra calculada foi de apenas **{prob:.1%}**.")
