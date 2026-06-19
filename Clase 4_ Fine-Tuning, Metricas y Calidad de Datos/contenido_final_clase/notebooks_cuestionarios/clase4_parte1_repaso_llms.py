# -*- coding: utf-8 -*-
# Generated from: clase4_parte1_repaso_llms.ipynb

# %%
# # Clase 4 — IA Generativa en Biomedica
# ## Parte 1: Repaso — Como funciona un LLM
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
# # PARTE 1: Repaso — Como funciona un LLM
# 
# Un LLM (Large Language Model):
# - Recibe texto como secuencia de **tokens** (palabras o sub-palabras)
# - Los transforma en **embeddings**: vectores numericos que capturan significado semantico
# - Predice el **siguiente token** mas probable dada la secuencia anterior
# 
# > Aprende patrones estadisticos del lenguaje, NO "entiende" medicina como un medico.

# %%
# ### 1.1 Tokenizacion en la practica
# 
# Veamos como un modelo tokeniza texto medico.

# %%
from transformers import AutoTokenizer

# Cargamos el tokenizer de un modelo comun (GPT-2 como ejemplo abierto)
tokenizer = AutoTokenizer.from_pretrained("gpt2")

textos = [
    "Paciente masculino de 65 anios con diabetes tipo 2.",
    "Se indica metformina 850mg cada 12 horas.",
    "Antecedentes: hipertension arterial, dislipidemia.",
    "The patient presents with acute myocardial infarction.",
]

for texto in textos:
    tokens = tokenizer.encode(texto)
    decoded = [tokenizer.decode([t]) for t in tokens]
    print(f"Texto: {texto}")
    print(f"  Num tokens: {len(tokens)}")
    print(f"  Tokens: {decoded}")
    print()

# %%
# ### Ejercicio 1.1: Comparar tokenizadores
# 
# Diferentes modelos tokenizan el mismo texto de formas distintas.
# Completa la funcion `comparar_tokenizadores` que recibe una lista de nombres de modelos
# y un texto, y retorna un DataFrame con el nombre del modelo, la cantidad de tokens,
# y la lista de tokens generados.

# %%
def comparar_tokenizadores(modelos: list[str], texto: str) -> pd.DataFrame:
    """
    Para cada modelo en la lista, tokeniza el texto y devuelve un DataFrame con:
    - 'modelo': nombre del modelo
    - 'num_tokens': cantidad de tokens
    - 'tokens': lista de tokens como strings

    Pista: usa AutoTokenizer.from_pretrained(modelo) para cada uno.
    """
    # --- TU CODIGO ACA ---
    pass


# Test: deberia funcionar con estos modelos y texto
modelos_test = ["gpt2", "bert-base-uncased", "bert-base-multilingual-uncased"]
texto_test = "Paciente masculino de 65 anios con diabetes tipo 2 e hipertension arterial."

# Descomenta cuando completes la funcion:
# df_tokens = comparar_tokenizadores(modelos_test, texto_test)
# print(df_tokens[['modelo', 'num_tokens']].to_string(index=False))
# plt.figure(figsize=(8, 4))
# plt.barh(df_tokens['modelo'], df_tokens['num_tokens'], color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
# plt.xlabel('Cantidad de tokens')
# plt.title(f'Tokens por modelo para: "{texto_test[:50]}..."')
# plt.tight_layout()
# plt.show()

# %%
# ### 1.2 Prediccion del siguiente token
# 
# Simulemos como un LLM asigna probabilidades a los siguientes tokens.

# %%
import torch
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("gpt2")
model.eval()

texto = "El tratamiento de primera linea para diabetes tipo 2 es"
inputs = tokenizer(texto, return_tensors="pt")

with torch.no_grad():
    outputs = model(**inputs)
    logits = outputs.logits[0, -1, :]  # Logits del ultimo token
    probs = torch.softmax(logits, dim=0)

# Top 10 tokens mas probables
top_k = 10
top_probs, top_indices = torch.topk(probs, top_k)

print(f"Contexto: '{texto}'")
print(f"\nTop {top_k} tokens predichos:")
print("-" * 40)
for prob, idx in zip(top_probs, top_indices):
    token = tokenizer.decode([idx])
    print(f"  '{token}' -> {prob:.4f} ({prob*100:.1f}%)")

# %%
# ### Ejercicio 1.2
# 
# El modelo predice el siguiente token mas probable.
# 1. Aparece "metformina" entre los top tokens? Por que si o por que no?
# 2. Que nos dice esto sobre la diferencia entre "saber" y "generar" en un LLM?
# 3. Modifica el texto de entrada para que este en ingles y compara los resultados.

# %%
# Prueba con texto en ingles
texto_en = "The first-line treatment for type 2 diabetes is"
inputs_en = tokenizer(texto_en, return_tensors="pt")

with torch.no_grad():
    outputs_en = model(**inputs_en)
    logits_en = outputs_en.logits[0, -1, :]
    probs_en = torch.softmax(logits_en, dim=0)

top_probs_en, top_indices_en = torch.topk(probs_en, 10)
print(f"Contexto: '{texto_en}'")
print(f"\nTop 10 tokens predichos:")
for prob, idx in zip(top_probs_en, top_indices_en):
    token = tokenizer.decode([idx])
    print(f"  '{token}' -> {prob:.4f} ({prob*100:.1f}%)")

# %%
# ### 1.3 Pipeline de entrenamiento de un LLM
# 
# El entrenamiento tiene 3 etapas (ref: Chip Huyen, *AI Engineering*, 2025):
# 
# 1. **Pre-training** (self-supervised): texto masivo sin anotaciones (internet, libros, codigo, papers). Usa ~98% del computo total. Resultado: modelo que completa texto pero no sigue instrucciones.
# 2. **Post-training** (por el proveedor): SFT con pares (instruccion, respuesta) de alta calidad + RLHF/DPO donde humanos comparan respuestas. Usa solo ~2% del computo. Resultado: modelo alineado.
# 3. **Fine-tuning** (por vos, el application developer): adaptar el modelo alineado a tu tarea especifica.
# 
# > *"Any training that happens after pre-training is finetuning"* — lo que hacen los proveedores (SFT, RLHF) y lo que haces vos es el mismo tipo de proceso.
# 
# **Dato:** El costo de crear datos de SFT es ~$10 por par (prompt, response). InstructGPT uso 13.000 pares -> ~$130.000 solo en datos.

# %%
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Panel 1: Distribucion de computo
etapas = ['Pre-training\n(~98% computo)', 'Post-training\n(~2% computo)']
computo = [98, 2]
colors = ['#2196F3', '#FF9800']
wedges, texts, autotexts = axes[0].pie(computo, labels=etapas, autopct='%1.0f%%',
    colors=colors, textprops={'fontsize': 12}, startangle=90)
autotexts[0].set_fontsize(14)
autotexts[0].set_fontweight('bold')
axes[0].set_title('Distribucion de computo en entrenamiento', fontsize=13, fontweight='bold')

# Panel 2: Que aporta cada etapa
categorias = ['Completar\ntexto', 'Seguir\ninstrucciones', 'Rechazar\ncontenido\ntoxico', 'Conocimiento\nmedico']
pre_scores = [0.9, 0.2, 0.1, 0.5]
post_scores = [0.9, 0.85, 0.8, 0.6]

x = np.arange(len(categorias))
width = 0.35
axes[1].bar(x - width/2, pre_scores, width, label='Solo pre-training', color='#E0E0E0', edgecolor='black')
axes[1].bar(x + width/2, post_scores, width, label='+ Post-training (SFT + RLHF)', color='#4CAF50', edgecolor='black')
axes[1].set_ylabel('Capacidad relativa (ilustrativo)')
axes[1].set_title('Efecto del post-training', fontsize=13, fontweight='bold')
axes[1].set_xticks(x)
axes[1].set_xticklabels(categorias, fontsize=10)
axes[1].legend(fontsize=10)
axes[1].set_ylim(0, 1.1)
axes[1].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.show()

print("Nota: los scores del panel derecho son ILUSTRATIVOS para mostrar el patron general.")
print("El post-training transforma un modelo que solo completa texto")
print("en uno que sigue instrucciones y es mas seguro, usando solo ~2% del computo total.")

# %%
# ### Quiz 1: Repaso de LLMs
# 
# Responde sin mirar las diapositivas. Luego verifica tus respuestas.

# %%
# **P1:** Un LLM "entiende" medicina como un medico?
# - a) Si, porque fue entrenado con papers medicos
# - b) No, genera la secuencia de tokens mas probable segun su entrenamiento
# - c) Depende del modelo
# 
# **P2:** Que hace SFT (Supervised Fine-Tuning) en el post-training?
# - a) Entrena con texto general de internet
# - b) Entrena con pares instruccion -> respuesta ideal de alta calidad
# - c) Humanos rankean respuestas del modelo
# 
# **P3:** Cual es la diferencia principal entre pre-training y post-training?
# - a) El pre-training usa datos curados y el post-training datos de internet
# - b) El pre-training aprende lenguaje general (~98% del computo) y el post-training alinea el modelo para seguir instrucciones (~2% del computo)
# - c) No hay diferencia, son lo mismo
# 
# <details>
# <summary><b>Click para ver respuestas</b></summary>
# 
# **P1: b)** Un LLM aprende patrones estadisticos del lenguaje, NO razona como un medico.
# 
# **P2: b)** SFT usa pares (instruccion, respuesta) de alta calidad creados por anotadores expertos. RLHF/DPO es donde humanos comparan respuestas.
# 
# **P3: b)** Pre-training: texto masivo sin anotaciones. Post-training: SFT + preference FT. Usa solo ~2% del computo total (ref: Chip Huyen, AI Engineering 2025).
# 
# </details>

