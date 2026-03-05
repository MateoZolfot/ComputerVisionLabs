import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

# ---------- Configuración ----------
st.set_page_config(page_title="Mateo Classifier", page_icon="👤")
st.title("🛡️ Mateo Face Classifier")
st.write("Identificación de Estudiante vs Background (Lab 03)")

# ---------- Transform (Normalización de ResNet) ----------
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# ---------- Cargar modelo ----------
@st.cache_resource
def load_my_model():
    model = models.resnet18(weights=None)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, 2)
    model.load_state_dict(torch.load("mateo_classifier.pth", map_location="cpu"))
    model.eval()
    return model

model = load_my_model()
classes = ["Background", "Mateo"]

# ---------- Carga de Archivo ----------
uploaded_file = st.file_uploader("Selecciona una imagen (JPG, PNG)", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Usamos session_state para recordar cuántas rotaciones llevamos
    if 'rotacion' not in st.session_state:
        st.session_state.rotacion = 0

    image = Image.open(uploaded_file).convert("RGB")
    
    # 1. --- BOTÓN DE ROTACIÓN 90° ---
    col1, col2 = st.columns([1, 4])
    with col1:
        if st.button("🔄 Rotar 90°"):
            st.session_state.rotacion = (st.session_state.rotacion + 90) % 360
    
    # Aplicar la rotación acumulada
    image = image.rotate(-st.session_state.rotacion, expand=True)
    
    st.image(image, caption=f"Imagen rotada {st.session_state.rotacion}°", use_container_width=True)

    # Preprocesar
    img_t = preprocess(image)
    img_t = img_t.unsqueeze(0) 

    with torch.no_grad():
        outputs = model(img_t)
        probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
        
        # Ajusta aquí: ¿Mateo es el 1 o el 0? (Cámbialo si sale al revés)
        prob_background = probabilities[0].item()
        prob_mateo = probabilities[1].item()

    # 2. --- LÓGICA DE PENALIZACIÓN POTENCIAL (Para Famosos) ---
    # Si los famosos te salen con 98%, elevarlo a la 10 los baja a 81%
    potencia = 1
    confianza_mateo_ajustada = pow(prob_mateo, potencia)
    
    umbral = 0.75

    if confianza_mateo_ajustada > umbral:
        prediction = "Mateo"
        confidence_display = confianza_mateo_ajustada
    else:
        prediction = "Background"
        confidence_display = prob_background

    # ---------- RESULTADO FINAL ----------
    if prediction == "Mateo":
        st.success(f"✅ Resultado: {prediction} ({confidence_display*100:.2f}%)")
    else:
        st.warning(f"🖼️ Resultado: {prediction} ({confidence_display*100:.2f}%)")
        
    # Debug para ver qué está pasando con los números
    with st.expander("Ver valores técnicos"):
        st.write(f"Prob. Original Mateo: {prob_mateo:.4f}")
        st.write(f"Prob. con Penalización (^10): {confianza_mateo_ajustada:.4f}")