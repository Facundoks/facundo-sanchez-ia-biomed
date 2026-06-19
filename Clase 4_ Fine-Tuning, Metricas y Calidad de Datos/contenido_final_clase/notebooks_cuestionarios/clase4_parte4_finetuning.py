# -*- coding: utf-8 -*-
# Generated from: clase4_parte4_finetuning.ipynb

# %%
# # Clase 4 — IA Generativa en Biomedica
# ## Parte 4: Fine-Tuning — Cuando Si, Cuando No
# 
# ---

# %%
# ## Setup
# 
# Ejecuta estas celdas para instalar las dependencias necesarias.

# %%
!pip install -q openai rouge-score nltk scikit-learn numpy pandas matplotlib seaborn datasets transformers torch

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import json
import re
import warnings
warnings.filterwarnings('ignore')

print("Setup completo!")

# %%
# ---
# # PARTE 4: Fine-Tuning — Cuando Si, Cuando No
# 
# > **En la mayoria de los casos, NO necesitas fine-tuning.**
# 
# Regla: **si no agotaste las opciones sin fine-tuning, no estas listo para fine-tuning.**

# %%
# ### 4.1 Arbol de decision: Necesito fine-tuning?

# %%
# | Paso | Pregunta | Si respondiste NO... |
# |------|----------|---------------------|
# | 1 | Probaste al menos 10 variaciones de prompts? | Mejora tus prompts primero |
# | 2 | Probaste few-shot y chain-of-thought? | Intenta estas tecnicas |
# | 3 | Consideraste RAG (agregar contexto)? | RAG puede resolver tu problema |
# | 4 | Tenes metricas definidas y un eval set? | Sin metricas, no podes medir mejora |
# | 5 | Tenes al menos cientos de ejemplos de alta calidad? | Necesitas mas datos |
# | 6 | El conocimiento necesario NO esta en el entrenamiento general? | Prompting + RAG deberia alcanzar |
# | 7 | Tenes GPU(s) y expertise en NLP? | Necesitas infraestructura y equipo |
# 
# > **Si respondiste SI a todo -> Estas listo para fine-tuning!**

# %%
# ### 4.1b Fine-tuning = Transfer Learning
# 
# La idea central: el modelo ya tiene la mayor parte del conocimiento. El fine-tuning desbloquea capacidades que ya estan ahi pero que son dificiles de acceder solo con prompting (InstructGPT, OpenAI 2022).
# 
# **Quien hace cada etapa:**
# 
# | Quien | Que hace | Resultado |
# |-------|----------|-----------|
# | **Model developers** (OpenAI, Meta, Anthropic) | Pre-training + SFT + preference FT | Modelo alineado |
# | **Application developers** (vos) | Prompting, RAG, continued pre-training, SFT | Modelo adaptado a tu tarea |
# 
# **Opciones del developer (de menor a mayor esfuerzo):**
# 1. **Prompting** -> no tocas el modelo, solo el input
# 2. **RAG** -> le das mejor contexto, el modelo sigue igual
# 3. **Continued pre-training** -> entrenas con texto de tu dominio (sin anotar)
# 4. **Supervised FT** -> entrenas con pares (instruccion, respuesta) de tu tarea
# 
# La mayoria de los casos se resuelven con 1 y 2.

# %%
# ### 4.2 Tipos de fine-tuning: comparacion

# %%
tipos_ft = pd.DataFrame({
    'Tipo': ['Full Fine-Tuning', 'LoRA', 'QLoRA', 'Instruction Tuning', 'Domain Adaptation'],
    'Que actualiza': ['Todos los parametros', 'Matrices pequenias adicionales',
                      'LoRA + cuantizacion 4-bit', 'Pares instruccion->respuesta',
                      'Texto del dominio (MLM)'],
    'GPU necesaria': ['8x A100 80GB', '1x A100 40GB', '1x RTX 4090 24GB',
                      'Depende del metodo', 'Depende del metodo'],
    'Costo relativo': [100, 10, 5, 10, 8],
    'Riesgo_forgetting': ['Alto', 'Bajo', 'Bajo', 'Medio', 'Medio'],
})

print("Comparacion de tipos de Fine-Tuning:")
print("=" * 100)
print(tipos_ft.to_string(index=False))

# Visualizacion
fig, ax = plt.subplots(figsize=(10, 5))
colors_risk = {'Alto': '#FF6B6B', 'Medio': '#FFD93D', 'Bajo': '#4ECDC4'}
bar_colors = [colors_risk[r] for r in tipos_ft['Riesgo_forgetting']]

bars = ax.barh(tipos_ft['Tipo'], tipos_ft['Costo relativo'], color=bar_colors, edgecolor='black')
ax.set_xlabel('Costo relativo', fontsize=12)
ax.set_title('Costo y Riesgo de Catastrophic Forgetting por Tipo de Fine-Tuning',
            fontsize=13, fontweight='bold')

# Leyenda
from matplotlib.patches import Patch
legend_elements = [Patch(facecolor=c, label=f'Riesgo: {l}') for l, c in colors_risk.items()]
ax.legend(handles=legend_elements, loc='lower right')
ax.grid(True, alpha=0.3, axis='x')

plt.tight_layout()
plt.show()

# %%
# ### 4.3 Simulacion: preparar datos para fine-tuning
# 
# Veamos como se estructura un dataset de fine-tuning.

# %%
# Ejemplo de como se estructura un dataset de fine-tuning (formato SFT = pares instruccion -> respuesta):
# 
# ---
# **Ejemplo 1 - ECG normal:**
# 
# | Campo | Contenido |
# |-------|-----------|
# | **INSTRUCTION** | Genera un reporte de ECG a partir de los siguientes hallazgos |
# | **INPUT** | Ritmo sinusal, FC 78, eje normal, PR 0.16s, QRS 0.08s, ST sin cambios, T normales |
# | **OUTPUT** | Ritmo sinusal regular, FC 78 lpm, eje normal, intervalos normales. **CONCLUSION: ECG dentro de limites normales.** |
# 
# **Ejemplo 2 - Fibrilacion auricular:**
# 
# | Campo | Contenido |
# |-------|-----------|
# | **INSTRUCTION** | Genera un reporte de ECG a partir de los siguientes hallazgos |
# | **INPUT** | FC 110, ritmo irregular, ausencia de ondas P, intervalos RR variables, QRS estrecho |
# | **OUTPUT** | Ritmo irregular, ausencia de ondas P, FC 110 lpm, RR variable, QRS estrecho. **CONCLUSION: FA con respuesta ventricular rapida.** |
# 
# **Ejemplo 3 - IAM anterior:**
# 
# | Campo | Contenido |
# |-------|-----------|
# | **INSTRUCTION** | Genera un reporte de ECG a partir de los siguientes hallazgos |
# | **INPUT** | Ritmo sinusal FC 65, supradesnivel ST V1-V4, imagen en espejo en DII-DIII-aVF |
# | **OUTPUT** | Ritmo sinusal, FC 65, supradesnivel ST en V1-V4 (anterior), cambios reciprocos en cara inferior. **CONCLUSION: IAM con elevacion ST anterior. URGENCIA ALTA.** |

# %%
### Ejercicio 4.1: Armar un dataset de fine-tuning

Una tarea fundamental es transformar datos crudos al formato que necesitan las APIs de fine-tuning.
Escribe una funcion que convierta notas clinicas al formato de mensajes de chat (system/user/assistant).

# %%
import json

def convertir_a_chat_format(nota_clinica: str, diagnostico: str, system_prompt: str) -> dict:
    """
    Convierte un par (nota_clinica, diagnostico) al formato chat para fine-tuning.

    Retorna un dict con la clave 'messages' que contiene una lista de 3 dicts:
    - {"role": "system", "content": system_prompt}
    - {"role": "user", "content": f"Analiza la siguiente nota clinica y genera el diagnostico presuntivo:\n\n{nota_clinica}"}
    - {"role": "assistant", "content": diagnostico}
    """
    # --- TU CODIGO ACA ---
    pass


def preparar_dataset_ft(datos: list[dict], system_prompt: str) -> list[dict]:
    """
    Recibe una lista de dicts con claves 'nota_clinica' y 'diagnostico'.
    Retorna una lista de dicts en formato chat (usando convertir_a_chat_format).
    Filtra los registros donde nota_clinica o diagnostico esten vacios.
    """
    # --- TU CODIGO ACA ---
    pass


# ---------- Test ----------
system = "Sos un medico clinico. Genera el diagnostico presuntivo a partir de la nota clinica."

datos_crudos = [
    {"nota_clinica": "Pcte 65a, dolor toracico opresivo, ECG supra ST V1-V4, troponina 2.8",
     "diagnostico": "IAM anterior con elevacion del ST"},
    {"nota_clinica": "Mujer 45a, cefalea intensa subita, rigidez de nuca, fotofobia",
     "diagnostico": "Sospecha de hemorragia subaracnoidea"},
    {"nota_clinica": "", "diagnostico": "Dato faltante"},  # Deberia ser filtrado
    {"nota_clinica": "Ninio 3a, fiebre 39C, otalgia derecha, timpano abombado",
     "diagnostico": "Otitis media aguda"},
    {"nota_clinica": "Varon 70a, disnea progresiva, edemas MMII, ingurgitacion yugular",
     "diagnostico": ""},  # Deberia ser filtrado
]

dataset = preparar_dataset_ft(datos_crudos, system)
if dataset is not None:
    print(f"Registros validos: {len(dataset)} de {len(datos_crudos)}")
    print(f"\nEjemplo de formato generado:")
    print(json.dumps(dataset[0], indent=2, ensure_ascii=False))

# %%
# Para cada escenario, responde: Necesita fine-tuning? Justifica y propone alternativa.
# 
# | # | Escenario | Necesita FT? | Justificacion | Alternativa |
# |---|-----------|:------------:|---------------|-------------|
# | 1 | Queres que el modelo responda preguntas basicas sobre diabetes | ___ | ___ | ___ |
# | 2 | Necesitas generar reportes en el formato exacto del Hospital Italiano | ___ | ___ | ___ |
# | 3 | Queres clasificar notas clinicas en espaniol argentino con jerga local | ___ | ___ | ___ |
# | 4 | Necesitas un chatbot que responda FAQs aprobadas por el comite medico | ___ | ___ | ___ |

# %%
# ### 4.4 Simulacion: Catastrophic Forgetting

# %%
# SIMULACION ILUSTRATIVA del fenomeno de catastrophic forgetting.
# Las curvas muestran el patron tipico, no datos de un experimento real.
np.random.seed(42)

epochs_ft = np.arange(0, 21)
# Performance en tarea objetivo (mejora)
tarea_objetivo = 0.4 + 0.5 * (1 - np.exp(-0.15 * epochs_ft)) + np.random.normal(0, 0.02, len(epochs_ft))
# Performance en tareas generales (se degrada)
tarea_general = 0.85 - 0.3 * (1 - np.exp(-0.1 * epochs_ft)) + np.random.normal(0, 0.02, len(epochs_ft))
# Con LoRA (menor degradacion)
tarea_general_lora = 0.85 - 0.08 * (1 - np.exp(-0.1 * epochs_ft)) + np.random.normal(0, 0.02, len(epochs_ft))
tarea_objetivo_lora = 0.4 + 0.4 * (1 - np.exp(-0.12 * epochs_ft)) + np.random.normal(0, 0.02, len(epochs_ft))

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Full fine-tuning
axes[0].plot(epochs_ft, tarea_objetivo, 'g-', linewidth=2, label='Tarea objetivo (ECG)')
axes[0].plot(epochs_ft, tarea_general, 'r-', linewidth=2, label='Tareas generales')
axes[0].fill_between(epochs_ft, tarea_general, 0.85, alpha=0.1, color='red')
axes[0].set_xlabel('Epochs de fine-tuning', fontsize=12)
axes[0].set_ylabel('Performance', fontsize=12)
axes[0].set_title('Full Fine-Tuning:\nCatastrophic Forgetting', fontsize=13, fontweight='bold')
axes[0].legend()
axes[0].set_ylim(0.3, 1.0)
axes[0].grid(True, alpha=0.3)
axes[0].annotate('Pierde capacidades\ngenerales!', xy=(15, 0.6),
               fontsize=11, color='red', fontweight='bold')

# LoRA
axes[1].plot(epochs_ft, tarea_objetivo_lora, 'g-', linewidth=2, label='Tarea objetivo (ECG)')
axes[1].plot(epochs_ft, tarea_general_lora, 'b-', linewidth=2, label='Tareas generales')
axes[1].set_xlabel('Epochs de fine-tuning', fontsize=12)
axes[1].set_ylabel('Performance', fontsize=12)
axes[1].set_title('LoRA Fine-Tuning:\nMenor degradacion', fontsize=13, fontweight='bold')
axes[1].legend()
axes[1].set_ylim(0.3, 1.0)
axes[1].grid(True, alpha=0.3)
axes[1].annotate('Preserva capacidades\ngenerales', xy=(15, 0.78),
               fontsize=11, color='blue', fontweight='bold')

plt.tight_layout()
plt.show()

# %%
# ### 4.5 El requisito mas importante
# 
# > Si no podes completar esta frase, **no estas listo para fine-tuning**:
# >
# > *"Quiero que el modelo mejore en __________, medido por __________, usando datos de __________"*

# %%
# ### Ejercicio 4.2: Completa la frase
# 
# Completa para tu caso de uso:
# 
# > *"Quiero que el modelo mejore en **\_\_\_\_\_**,*
# > *medido por **\_\_\_\_\_**,*
# > *usando datos de **\_\_\_\_\_**"*
# 
# **Ejemplo valido:**
# > "Quiero que el modelo mejore en **generar reportes de ECG en formato SOAP**,
# > medido por **adherencia al formato (%) y correccion medica (Likert 1-5, evaluada por cardiologos)**,
# > usando datos de **2000 reportes del Hospital X revisados por especialistas**"
# 
# **Ejemplos invalidos:**
# - "Quiero que sea mejor en medicina" (demasiado vago)
# - "Quiero que no alucine" (no es medible sin mas contexto)
# 
# Luego verifica:
# - Es suficientemente especifico?
# - La metrica es cuantificable?
# - Los datos existen y son de calidad?

