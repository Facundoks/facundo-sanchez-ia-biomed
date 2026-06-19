# -*- coding: utf-8 -*-
# Generated from: clase4_parte3_prompting.ipynb

# %%
# # Clase 4 — IA Generativa en Biomedica
# ## Parte 3: Prompting — Primera Herramienta de Mejora
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
# # PARTE 3: Prompting — Primera Herramienta de Mejora
# 
# > **Optimizar el input es la intervencion mas barata y efectiva.**
# 
# Tecnicas de prompting:
# - **Zero-shot**: solo la instruccion, sin ejemplos
# - **Few-shot**: instruccion + ejemplos de input/output esperado
# - **Chain-of-thought (CoT)**: pedirle que razone paso a paso
# - **Role prompting**: asignarle un rol
# - **Instrucciones negativas**: decirle que NO hacer

# %%
# ### 3.1 Comparacion de tecnicas de prompting

# %%
# **Historia clinica de ejemplo:**
# 
# > Paciente femenina de 72 anios. Motivo de consulta: dolor toracico de 3 horas de evolucion, opresivo, irradiado a brazo izquierdo, acompaniado de sudoracion y nauseas.
# > Antecedentes: HTA de 15 anios, diabetes tipo 2, dislipidemia. Tabaquismo 30 paq/anio (dejo hace 5).
# > Medicacion habitual: enalapril 20mg/dia, metformina 1700mg/dia, atorvastatina 40mg/dia.
# > **Alergia a penicilina y sulfas.**
# > Examen fisico: TA 160/95, FC 98, FR 22, SatO2 94%. Diaforesica, ansiosa.
# > Auscultacion pulmonar: crepitantes bibasales. CV: R1R2 regulares sin soplos.
# > ECG: supradesnivel ST en V1-V4. Troponina I: 2.8 ng/mL (VN <0.04).
# 
# ---
# 
# **Tecnica 1 - Zero-shot (malo):**
# 
# ```
# Resume este texto.
# ```
# Problemas: no sabe que priorizar, puede inventar, no tiene formato.
# 
# **Tecnica 2 - Zero-shot (mejor):**
# 
# ```
# Resume esta historia clinica destacando diagnostico presuntivo, hallazgos criticos y alergias.
# ```
# 
# **Tecnica 3 - Role + formato:**
# 
# ```
# Sos un medico de emergencias. Resumi esta historia clinica en formato SOAP:
# - S (Subjetivo): motivo de consulta y sintomas
# - O (Objetivo): signos vitales, examenes, hallazgos
# - A (Analisis): diagnostico presuntivo
# - P (Plan): tratamiento y seguimiento
# No inventes informacion. Si falta un dato: 'No especificado'. Max 200 palabras.
# ```
# 
# **Tecnica 4 - Few-shot + Chain-of-Thought:**
# 
# ```
# Sos un medico de emergencias. Analiza esta historia clinica paso a paso:
# 
# Ejemplo:
# INPUT: Paciente de 55a, dolor precordial, ECG con onda Q en II-III-aVF, troponina 5.2
# ANALISIS:
# 1. Sintoma principal: dolor precordial
# 2. ECG: ondas Q patologicas en cara inferior -> sugiere IAM inferior
# 3. Troponina elevada: confirma danio miocardico
# 4. DIAGNOSTICO: IAM con elevacion ST de cara inferior
# 5. ALERGIAS: No reportadas
# 6. URGENCIA: Alta - requiere intervencion inmediata
# 
# Ahora analiza el siguiente caso con el mismo formato:
# ```
# 
# Observa como cada tecnica agrega mas estructura y contexto al modelo.

# %%
# ### Ejercicio 3.1: Construir y evaluar prompts programaticamente
# 
# Escribe prompts como strings de Python y evalualos automaticamente contra criterios definidos.
# Un buen prompt biomedico debe tener: rol, formato de salida, instrucciones negativas, manejo de incertidumbre.

# %%
def evaluar_prompt(prompt: str) -> dict:
    """
    Evalua la calidad de un prompt biomedico.

    Verifica (case-insensitive) si el prompt contiene:
    - 'tiene_rol': menciona un rol medico (buscar: 'medico', 'cardiologo', 'clinico',
       'especialista', 'enfermero', 'radiologo', 'patologo')
    - 'tiene_formato': especifica formato de salida (buscar: 'formato', 'estructura',
       'SOAP', 'JSON', 'lista', 'tabla', 'bullet')
    - 'tiene_negativa': tiene instrucciones negativas (buscar: 'no inventes', 'no alucines',
       'no generes', 'no incluyas', 'no agregues')
    - 'tiene_incertidumbre': maneja datos faltantes (buscar: 'no especificado', 'no disponible',
       'si falta', 'dato faltante', 'desconocido')
    - 'tiene_ejemplos': incluye ejemplos (buscar: 'ejemplo', 'input:', 'output:', 'caso:')
    - 'score': cantidad de criterios cumplidos (0 a 5)
    - 'nivel': 'basico' (0-1), 'intermedio' (2-3), 'avanzado' (4-5)

    Retorna dict con todos los campos anteriores.
    """
    # --- TU CODIGO ACA ---
    pass


# ---------- Test con prompts de distinta calidad ----------
prompts = {
    "Malo": "Resume esta historia clinica.",

    "Regular": "Sos un medico clinico. Resume esta historia clinica en formato SOAP.",

    "Bueno": """Sos un medico clinico con experiencia en emergencias.
Resumi esta historia clinica en formato SOAP:
- S (Subjetivo): motivo de consulta
- O (Objetivo): signos vitales, estudios
- A (Analisis): diagnostico presuntivo
- P (Plan): tratamiento
No inventes informacion. Si falta un dato, escribi 'No especificado'.""",

    "Excelente": """Sos un medico clinico con experiencia en emergencias.
Resumi esta historia clinica en formato SOAP.

Ejemplo:
Input: Pcte 55a, dolor precordial, ECG onda Q en II-III-aVF, troponina 5.2
Output:
- S: Dolor precordial
- O: ECG con ondas Q en cara inferior. Troponina 5.2
- A: IAM inferior
- P: Cateterismo urgente

No inventes informacion. Si falta un dato, escribi 'No especificado'.
No incluyas diagnosticos que no esten sustentados por los hallazgos.""",
}

for nombre, prompt in prompts.items():
    resultado = evaluar_prompt(prompt)
    if resultado is not None:
        print(f"{nombre:12s} -> Score: {resultado['score']}/5  Nivel: {resultado['nivel']}")
        faltantes = [k.replace('tiene_', '') for k, v in resultado.items()
                     if k.startswith('tiene_') and not v]
        if faltantes:
            print(f"{'':12s}    Falta: {', '.join(faltantes)}")
        print()

# %%
# ### 3.2 Evaluacion sistematica de prompts
# 
# El prompt engineering es un proceso experimental. Hay que testear variaciones sistematicamente.

# %%
# DATOS SIMULADOS para ilustrar el proceso de mejora iterativa de prompts.
# En tu proyecto, estos numeros vendrian de evaluar cada version contra tu eval set.


registro_prompts = pd.DataFrame({
    'Version': ['v1', 'v2', 'v3', 'v4', 'v5'],
    'Tecnica': ['Zero-shot', 'Role prompt', 'Role + formato', 'Few-shot', 'Few-shot + CoT'],
    'Precision_medica': [0.45, 0.62, 0.71, 0.78, 0.82],
    'Formato_correcto': [0.20, 0.55, 0.88, 0.90, 0.91],
    'Sin_alucinaciones': [0.60, 0.70, 0.75, 0.80, 0.85],
    'Costo_tokens': [50, 80, 120, 250, 350],
})

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Metricas de calidad
metricas_cols = ['Precision_medica', 'Formato_correcto', 'Sin_alucinaciones']
for col in metricas_cols:
    axes[0].plot(registro_prompts['Version'], registro_prompts[col], 'o-', label=col, linewidth=2)
axes[0].set_ylabel('Score')
axes[0].set_xlabel('Version del prompt')
axes[0].set_title('Mejora iterativa de prompts', fontsize=14, fontweight='bold')
axes[0].legend()
axes[0].set_ylim(0, 1)
axes[0].grid(True, alpha=0.3)

# Trade-off calidad vs costo
axes[1].scatter(registro_prompts['Costo_tokens'],
               registro_prompts['Precision_medica'],
               s=200, c=range(len(registro_prompts)), cmap='RdYlGn',
               edgecolors='black', linewidths=1.5, zorder=5)
for i, row in registro_prompts.iterrows():
    axes[1].annotate(row['Version'], (row['Costo_tokens'], row['Precision_medica']),
                    textcoords="offset points", xytext=(10, 5), fontsize=11)
axes[1].set_xlabel('Costo (tokens por llamada)', fontsize=12)
axes[1].set_ylabel('Precision medica', fontsize=12)
axes[1].set_title('Trade-off: Calidad vs Costo', fontsize=14, fontweight='bold')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print(registro_prompts.to_string(index=False))

