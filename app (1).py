
import streamlit as st
import joblib
import pandas as pd
import os
import time # For loading spinner

# Define the base path to the 'ia' directory
ia_dir = '/content/drive/MyDrive/ia'

# --- Configuración de la página --- 
st.set_page_config(
    page_title="Eco-Ride Churn Prediction", 
    layout="centered", 
    initial_sidebar_state="collapsed",
    menu_items={
        'About': "Esta aplicación predice el riesgo de abandono de clientes de Eco-Ride."
    }
)

# --- Título y Descripción de la Aplicación ---
st.image("https://i.imgur.com/your_logo_here.png", width=100) # Reemplaza con el logo de tu empresa si tienes uno
st.title("🚗 Eco-Ride: Predicción de Riesgo de Abandono")
st.markdown("Bienvenido al predictor de abandono de clientes de Eco-Ride. Introduce los detalles del cliente para evaluar su riesgo de cancelación.")

st.markdown("--- ")

# --- Cargar el preprocesador y el modelo ---
@st.cache_resource # Cache the models for better performance
def load_models():
    try:
        pipeline = joblib.load(os.path.join(ia_dir, 'pipeline_preproc.pkl'))
        model = joblib.load(os.path.join(ia_dir, 'best_knn_model.pkl'))
        return pipeline, model
    except Exception as e:
        st.error(f"Error al cargar los modelos: {e}. Asegúrate de que 'pipeline_preproc.pkl' y 'best_knn_model.pkl' estén en '{ia_dir}'")
        st.stop()

with st.spinner('Cargando modelos...'):
    pipeline_preproc, best_knn_model = load_models()

st.success("¡Modelos cargados exitosamente!")
st.markdown("--- ")

# --- Interfaz de entrada de datos ---
st.subheader("👥 Información del Cliente")

col1, col2 = st.columns(2)

with col1:
    age = st.slider("Edad del Cliente", min_value=18, max_value=100, value=30, help="Edad del cliente en años.")
    plan_selection = st.selectbox("Plan Contratado", options=["Basic", "Premium", "Elite"], help="Tipo de plan de suscripción del cliente.")
    monthly_usage_km = st.number_input("Uso Mensual (Km)", min_value=0.0, value=100.0, format="%.2f", help="Kilómetros recorridos por el cliente al mes.")

with col2:
    region_selection = st.selectbox("Región", options=["North", "South", "Center"], help="Región geográfica del cliente.")
    support_tickets = st.number_input("Tickets de Soporte", min_value=0, value=1, help="Número de veces que el cliente ha contactado a soporte.")
    average_spend = st.number_input("Gasto Promedio ($)", min_value=0.0, value=50.0, format="%.2f", help="Gasto promedio mensual del cliente.")

days_antiquity = st.number_input("Días de Antigüedad como Cliente", min_value=0, value=365, help="Número de días desde que el cliente se registró.")

st.markdown("--- ")

# --- Botón y Lógica de Predicción ---
if st.button("🚀 Analizar Riesgo de Abandono"): 
    with st.spinner('Realizando predicción...'):
        # Prepare data for prediction
        input_data = pd.DataFrame([{
            'Age': age,
            'Plan': plan_selection,
            'Region': region_selection,
            'Monthly_Usage_Km': monthly_usage_km,
            'Support_Tickets': support_tickets,
            'Average_Spend': average_spend,
            'Days_Antiquity': days_antiquity
        }])

        try:
            # Apply preprocessing pipeline
            processed_data = pipeline_preproc.transform(input_data)

            # Make prediction
            prediction = best_knn_model.predict(processed_data)[0]
            prediction_proba = best_knn_model.predict_proba(processed_data)[0]

            # Display results
            st.subheader("📊 Resultado de la Predicción:")
            
            prob_churn = prediction_proba[1]
            prob_stable = prediction_proba[0]

            if prediction == 1: # Assuming 1 means churn (High Risk)
                st.error(f"**¡ALTO RIESGO DE CANCELACIÓN!** ⚠️")
                st.markdown(f"<p style='font-size:20px;'>Probabilidad de Abandono: <b>{prob_churn*100:.2f}%</b></p>", unsafe_allow_html=True)
            else:
                st.success(f"**CLIENTE ESTABLE** ✅")
                st.markdown(f"<p style='font-size:20px;'>Probabilidad de Estabilidad: <b>{prob_stable*100:.2f}%</b></p>", unsafe_allow_html=True)
            
            st.metric(label="Probabilidad de Abandono (Clase 1)", value=f"{prob_churn*100:.2f}%")
            st.metric(label="Probabilidad de Estabilidad (Clase 0)", value=f"{prob_stable*100:.2f}%")

            st.markdown("--- ")
            st.info("**Nota:** La Clase 1 indica 'Alto Riesgo de Cancelación', la Clase 0 indica 'Cliente Estable'.")

        except Exception as e:
            st.error(f"Ha ocurrido un error durante la predicción: {e}")

st.markdown("--- ")
st.markdown("### Sobre la Aplicación")
st.info("Esta aplicación utiliza un modelo K-Nearest Neighbors (KNN) entrenado con datos históricos de Eco-Ride para predecir el riesgo de abandono de los clientes. Los datos de entrada son preprocesados mediante un pipeline para garantizar la compatibilidad con el modelo. Las predicciones se basan en las probabilidades de pertenencia a cada clase (abandono o estabilidad).")
