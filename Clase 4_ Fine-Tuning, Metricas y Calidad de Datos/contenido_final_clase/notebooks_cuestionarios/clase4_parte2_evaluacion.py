# -*- coding: utf-8 -*-
# Generated from: clase4_parte2_evaluacion.ipynb

# %%
# # Clase 4 — IA Generativa en Biomedica
# ## Parte 2: Evaluacion — Metricas, Benchmarks y Evaluacion Humana
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
# # PARTE 2: Evaluacion — Metricas, Benchmarks y Evaluacion Humana
# 
# > **Antes de intentar mejorar un modelo, tenemos que poder medirlo bien.**
# 
# Sin evaluacion rigurosa:
# - No sabes si el modelo sirve para tu tarea
# - No sabes si un cambio lo mejoro o lo empeoro
# - No podes comparar opciones
# - No podes justificar su uso en un entorno clinico
# 
# La evaluacion **no es un paso final** -> es el **primer paso** de cualquier proyecto serio con LLMs.

# %%
# ### 2.1 Cross-Entropy Loss
# 
# La loss mide que tan bien el modelo predice el siguiente token.

# %%
def cross_entropy_loss(prob_correcto):
    """Calcula cross-entropy loss para un token."""
    return -np.log(prob_correcto)

# Simulemos la loss durante entrenamiento
probabilidades = np.linspace(0.01, 0.99, 100)
losses = [cross_entropy_loss(p) for p in probabilidades]

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Loss vs probabilidad
axes[0].plot(probabilidades, losses, 'b-', linewidth=2)
axes[0].set_xlabel('Probabilidad asignada al token correcto', fontsize=12)
axes[0].set_ylabel('Cross-Entropy Loss', fontsize=12)
axes[0].set_title('Cross-Entropy Loss', fontsize=14, fontweight='bold')
axes[0].axhline(y=1, color='r', linestyle='--', alpha=0.5, label='Loss = 1')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Curva de entrenamiento simulada
epochs = np.arange(1, 51)
train_loss = 4.5 * np.exp(-0.08 * epochs) + 0.5 + np.random.normal(0, 0.05, len(epochs))
val_loss = 4.5 * np.exp(-0.07 * epochs) + 0.7 + np.random.normal(0, 0.08, len(epochs))
# Simular overfitting despues del epoch 35
val_loss[35:] = val_loss[35:] + np.linspace(0, 0.5, 15)

axes[1].plot(epochs, train_loss, 'b-', label='Train Loss', linewidth=2)
axes[1].plot(epochs, val_loss, 'r-', label='Validation Loss', linewidth=2)
axes[1].axvline(x=35, color='gray', linestyle='--', alpha=0.5, label='Overfitting comienza')
axes[1].set_xlabel('Epoch', fontsize=12)
axes[1].set_ylabel('Loss', fontsize=12)
axes[1].set_title('Curva de entrenamiento tipica', fontsize=14, fontweight='bold')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print("\nLa loss indica que el modelo esta aprendiendo, pero NO indica:")
print("  - Que las respuestas sean medicamente correctas")
print("  - Que el resumen sea clinicamente util")
print("  - Que no alucine")

# %%
# ### 2.2 Metricas de generacion de texto: BLEU y ROUGE
# 
# Implementemos y comparemos estas metricas con ejemplos medicos.

# %%
from rouge_score import rouge_scorer
import nltk
nltk.download('punkt_tab', quiet=True)

def calcular_rouge(referencia, hipotesis):
    """Calcula ROUGE-1, ROUGE-2 y ROUGE-L."""
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=False)
    scores = scorer.score(referencia, hipotesis)
    return {
        'ROUGE-1': scores['rouge1'].fmeasure,
        'ROUGE-2': scores['rouge2'].fmeasure,
        'ROUGE-L': scores['rougeL'].fmeasure,
    }

def calcular_bleu_simple(referencia, hipotesis):
    """Calcula BLEU simplificado (overlap de n-gramas)."""
    ref_tokens = referencia.lower().split()
    hyp_tokens = hipotesis.lower().split()

    # Unigram precision
    ref_counter = Counter(ref_tokens)
    hyp_counter = Counter(hyp_tokens)
    overlap = sum((hyp_counter & ref_counter).values())
    precision = overlap / len(hyp_tokens) if hyp_tokens else 0
    return precision

# %%
# Ejemplo medico: resumen de historia clinica
referencia = (
    "Paciente masculino de 65 anios con diabetes tipo 2 e hipertension arterial. "
    "Presenta disnea de esfuerzo progresiva. Se indica metformina 850mg, "
    "enalapril 10mg y se solicita ecocardiograma. Alergia a penicilina."
)

# Tres resumenes generados con diferente calidad
resumenes = {
    "Bueno": (
        "Paciente de 65 anios, diabetico tipo 2 e hipertenso. Disnea de esfuerzo. "
        "Tratamiento: metformina 850mg, enalapril 10mg. Pendiente ecocardiograma. "
        "Alergico a penicilina."
    ),
    "ROUGE alto pero peligroso": (
        "Paciente masculino de 65 anios con diabetes tipo 2 e hipertension arterial. "
        "Presenta disnea de esfuerzo progresiva. Se indica metformina 850mg y enalapril 10mg."
        # OMITE la alergia a penicilina!
    ),
    "Alucinado": (
        "Paciente masculino de 65 anios con diabetes tipo 2 e hipertension arterial. "
        "Presenta disnea. Se indica insulina glargina 20U y losartan 50mg. "
        "Alergia a sulfonamidas."
        # Medicamentos y alergia INVENTADOS!
    ),
}

print("REFERENCIA:")
print(f"  {referencia}\n")
print("=" * 80)

resultados = []
for nombre, resumen in resumenes.items():
    rouge = calcular_rouge(referencia, resumen)
    bleu = calcular_bleu_simple(referencia, resumen)
    resultados.append({"Resumen": nombre, "BLEU-1": bleu, **rouge})

    print(f"\n--- {nombre} ---")
    print(f"  {resumen}")
    print(f"  BLEU-1={bleu:.3f} | ROUGE-1={rouge['ROUGE-1']:.3f} | ROUGE-L={rouge['ROUGE-L']:.3f}")

df_metricas = pd.DataFrame(resultados)
print("\n" + "=" * 80)
print(df_metricas.to_string(index=False))

# %%
# Visualizacion
fig, ax = plt.subplots(figsize=(10, 5))
x = np.arange(len(df_metricas))
width = 0.2

for i, col in enumerate(['BLEU-1', 'ROUGE-1', 'ROUGE-2', 'ROUGE-L']):
    ax.bar(x + i*width, df_metricas[col], width, label=col)

ax.set_ylabel('Score')
ax.set_title('Metricas automaticas vs calidad real del resumen', fontsize=14, fontweight='bold')
ax.set_xticks(x + 1.5*width)
ax.set_xticklabels(df_metricas['Resumen'])
ax.legend()
ax.set_ylim(0, 1)
ax.grid(True, alpha=0.3, axis='y')

# Anotaciones
ax.annotate('ROUGE alto pero\nomite alergia critica!',
           xy=(1.3, 0.85), fontsize=10, color='red', fontweight='bold',
           ha='center')

plt.tight_layout()
plt.show()

print("\nLeccion clave: Un resumen puede tener ROUGE alto pero omitir")
print("una alergia medicamentosa critica. Las metricas automaticas")
print("NO capturan seguridad clinica.")

# %%
# ### Ejercicio 2.1: Implementar ROUGE-1 desde cero
# 
# Antes de usar librerias, es fundamental entender que calcula ROUGE.
# Implementa ROUGE-1 (unigram) paso a paso: tokeniza, conta overlap, calcula precision, recall y F1.
# Luego usalo para demostrar que un ROUGE alto no garantiza calidad clinica.

# %%
def rouge1_manual(referencia: str, hipotesis: str) -> dict:
    """
    Calcula ROUGE-1 (unigrama) manualmente.

    Pasos:
    1. Tokenizar ambos textos (split por espacios, convertir a minusculas)
    2. Contar tokens en comun (interseccion de multisets)
    3. Calcular precision = overlap / len(tokens_hipotesis)
    4. Calcular recall = overlap / len(tokens_referencia)
    5. Calcular F1 = 2 * precision * recall / (precision + recall)

    Retorna dict con 'precision', 'recall', 'f1'.
    Pista: usa collections.Counter para contar tokens y el operador & para interseccion.
    """
    # --- TU CODIGO ACA ---
    pass


# ---------- Tests ----------
ref = "el gato se sento en la alfombra"
hyp = "el gato esta en la alfombra roja"
resultado = rouge1_manual(ref, hyp)
if resultado is not None:
    print(f"Tu ROUGE-1:  P={resultado['precision']:.3f}  R={resultado['recall']:.3f}  F1={resultado['f1']:.3f}")

    # Verificar contra libreria
    scorer = rouge_scorer.RougeScorer(['rouge1'], use_stemmer=False)
    lib_score = scorer.score(ref, hyp)
    print(f"Libreria:    P={lib_score['rouge1'].precision:.3f}  R={lib_score['rouge1'].recall:.3f}  F1={lib_score['rouge1'].fmeasure:.3f}")

    if abs(resultado['f1'] - lib_score['rouge1'].fmeasure) < 0.01:
        print("\nCORRECTO! Tu implementacion coincide con la libreria.")
    else:
        print("\nREVISA: hay diferencia con la libreria.")

# ---------- Aplicacion clinica ----------
print("\n--- Aplicacion clinica ---")
ref_clinica = "Paciente con infarto agudo de miocardio. Se administra aspirina 300mg y heparina. Alergia a penicilina."
bueno = "IAM confirmado. Se dio aspirina 300mg con heparina. Alergico a penicilina."
malo = "Paciente con infarto agudo de miocardio. Se administra aspirina 300mg y heparina."  # OMITE ALERGIA

if resultado is not None:
    r_bueno = rouge1_manual(ref_clinica, bueno)
    r_malo = rouge1_manual(ref_clinica, malo)
    print(f"Bueno clinicamente (sinonimos):   ROUGE-1 F1 = {r_bueno['f1']:.3f}")
    print(f"Peligroso (omite alergia):        ROUGE-1 F1 = {r_malo['f1']:.3f}")
    print("\nEl resumen peligroso tiene ROUGE mas alto! Las metricas automaticas no capturan seguridad clinica.")

# %%
# ### 2.3 BERTScore: similitud semantica
# 
# BLEU y ROUGE comparan tokens exactos: si usas un sinonimo, el score baja aunque el significado sea el mismo.
# 
# **BERTScore** resuelve esto usando embeddings contextuales de un modelo BERT pre-entrenado. En vez de comparar strings, compara vectores semanticos de cada token, capturando que "hipertension arterial" y "presion alta" significan lo mismo.

# %%
!pip install -q bert-score

# %%
from bert_score import score as bert_score

# Pares de textos: mismo significado vs sin relacion
referencias = [
    "El paciente tiene hipertension arterial",
    "Se indica metformina 850mg",
    "Dolor toracico agudo",
]
hipotesis = [
    "El enfermo padece presion alta",           # sinonimos
    "Se prescribe metformina 850 miligramos",   # reformulacion
    "El gato se sento en el tejado",            # sin relacion
]

P, R, F1 = bert_score(hipotesis, referencias, lang="es", verbose=False)

print("BERTScore vs ROUGE-1 (F1):")
print("-" * 70)
for ref, hyp, bs, in zip(referencias, hipotesis, F1):
    rouge = calcular_rouge(ref, hyp)
    print(f"\nRef: '{ref}'")
    print(f"Hyp: '{hyp}'")
    print(f"  ROUGE-1 F1:  {rouge['ROUGE-1']:.3f}")
    print(f"  BERTScore F1: {bs:.3f}")

print("\nBERTScore captura que 'hipertension arterial' y 'presion alta' son semanticamente")
print("similares, algo que ROUGE no puede hacer porque solo compara tokens exactos.")

# %%
# ### 2.4 Evaluacion humana: sistema de scoring
# 
# Simulemos un proceso de evaluacion humana con escala Likert.

# %%
# DATOS SIMULADOS con fines ilustrativos.
# En un proyecto real, estos scores vendrian de medicos evaluando outputs del modelo.
np.random.seed(42)

criterios = ['Fidelidad', 'Completitud', 'No alucinacion', 'Relevancia clinica', 'Formato']
modelos = ['GPT-4', 'Claude', 'LLaMA-70B', 'Mistral-7B']

# Simulamos scores Likert (1-5) de 3 evaluadores para 20 casos
n_evaluadores = 3
n_casos = 20

evaluaciones = {}
for modelo in modelos:
    base = {'GPT-4': 4.0, 'Claude': 4.1, 'LLaMA-70B': 3.5, 'Mistral-7B': 3.0}[modelo]
    scores_modelo = {}
    for criterio in criterios:
        ajuste = {'Fidelidad': 0, 'Completitud': -0.3, 'No alucinacion': 0.2,
                  'Relevancia clinica': -0.1, 'Formato': 0.1}[criterio]
        # 3 evaluadores x 20 casos
        raw = np.clip(np.random.normal(base + ajuste, 0.5, (n_evaluadores, n_casos)), 1, 5)
        scores_modelo[criterio] = raw
    evaluaciones[modelo] = scores_modelo

# Calcular promedios
promedios = {}
for modelo in modelos:
    promedios[modelo] = {c: evaluaciones[modelo][c].mean() for c in criterios}

df_eval = pd.DataFrame(promedios).T
print("Evaluacion humana promedio (escala Likert 1-5):")
print(df_eval.round(2).to_string())

# %%
# Grafico radar para comparar modelos
from matplotlib.patches import FancyBboxPatch

angles = np.linspace(0, 2 * np.pi, len(criterios), endpoint=False).tolist()
angles += angles[:1]  # cerrar el poligono

fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']

for i, modelo in enumerate(modelos):
    values = [promedios[modelo][c] for c in criterios]
    values += values[:1]
    ax.plot(angles, values, 'o-', linewidth=2, label=modelo, color=colors[i])
    ax.fill(angles, values, alpha=0.1, color=colors[i])

ax.set_xticks(angles[:-1])
ax.set_xticklabels(criterios, fontsize=11)
ax.set_ylim(1, 5)
ax.set_title('Evaluacion Humana por Criterio Clinico', fontsize=14, fontweight='bold', pad=20)
ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
plt.tight_layout()
plt.show()

# %%
# Inter-annotator agreement (Cohen's Kappa simplificado)
from sklearn.metrics import cohen_kappa_score

print("Inter-Annotator Agreement (Cohen's Kappa):")
print("(>0.6 = sustancial, >0.8 = casi perfecto)\n")

for modelo in modelos:
    # Comparamos evaluador 0 vs evaluador 1 (redondeando a enteros para kappa)
    e1 = np.round(evaluaciones[modelo]['Fidelidad'][0]).astype(int)
    e2 = np.round(evaluaciones[modelo]['Fidelidad'][1]).astype(int)
    kappa = cohen_kappa_score(e1, e2)
    print(f"  {modelo} (Fidelidad): kappa = {kappa:.3f}")

# %%
# ### Ejercicio 2.2: Construir un evaluador de resumenes clinicos
# 
# Escribe una funcion que evalue un resumen clinico contra una referencia usando multiples criterios.
# No alcanza con ROUGE: necesitamos verificar que el resumen incluya informacion critica (alergias, medicamentos, diagnosticos).

# %%
def evaluar_resumen_clinico(referencia: str, resumen: str, campos_criticos: list[str]) -> dict:
    """
    Evalua un resumen clinico contra una referencia.

    Retorna un dict con:
    - 'rouge1_f1': ROUGE-1 F1 score (usa calcular_rouge definida arriba)
    - 'campos_presentes': lista de campos_criticos que aparecen en el resumen
      (buscar como substring, case-insensitive)
    - 'campos_faltantes': lista de campos_criticos que NO aparecen
    - 'cobertura_critica': proporcion de campos criticos presentes (0.0 a 1.0)
    - 'alerta': True si cobertura_critica < 1.0 (falta info critica)
    """
    # --- TU CODIGO ACA ---
    pass


# ---------- Test ----------
ref = """Paciente femenina de 72 anios. Motivo de consulta: dolor toracico.
Antecedentes: HTA, diabetes tipo 2. Alergia a penicilina y sulfas.
Medicacion: enalapril 20mg, metformina 1700mg. ECG: supradesnivel ST V1-V4.
Troponina I: 2.8 ng/mL. Diagnostico: IAM anterior."""

resumen_test = """Mujer 72a con dolor toracico. HTA y diabetes.
ECG con supra ST. Troponina elevada. IAM anterior."""

campos = ["alergia", "penicilina", "medicacion", "enalapril", "metformina", "troponina", "IAM"]

resultado = evaluar_resumen_clinico(ref, resumen_test, campos)
if resultado is not None:
    print(f"ROUGE-1 F1:          {resultado['rouge1_f1']:.3f}")
    print(f"Campos presentes:    {resultado['campos_presentes']}")
    print(f"Campos FALTANTES:    {resultado['campos_faltantes']}")
    print(f"Cobertura critica:   {resultado['cobertura_critica']:.1%}")
    print(f"ALERTA:              {'SI - FALTA INFO CRITICA' if resultado['alerta'] else 'No'}")

# %%
# ### 2.5 Benchmarks medicos
# 
# Exploremos los benchmarks estandar para evaluar LLMs en medicina.

# %%
# Datos REALES de benchmarks medicos (fuentes verificadas)
# Fuentes:
#   GPT-4: Nori et al., 'Capabilities of GPT-4 on Medical Challenge Problems', arXiv:2303.13375, 2023
#   Med-PaLM 2: Singhal et al., Nature Medicine, 2024 (doi:10.1038/s41591-024-03423-7)
#   GPT-3.5: Singhal et al., Nature Medicine 2024; John Snow Labs benchmarks
#   Med-Gemini: Google Research, 'Advancing Medical AI with Med-Gemini', 2024
#   USMLE passing threshold: ~60% (Kung et al., PLOS Digital Health, 2023)
#   PubMedQA human baseline: 78.0% (reportado en el paper original de PubMedQA)

benchmarks = pd.DataFrame({
    'Benchmark': ['MedQA\n(USMLE)', 'PubMedQA', 'MedMCQA', 'MMLU\nClinical Knowledge'],
    'GPT-4':       [86.1, 75.2, 72.4, 86.8],
    'Med-PaLM 2':  [86.5, 81.8, 72.3, None],
    'GPT-3.5':     [60.2, 78.2, None, 69.8],
})

# Baselines humanos/aprobacion
human_baselines = {
    'MedQA\n(USMLE)': 60,      # Threshold de aprobacion USMLE
    'PubMedQA': 78.0,            # Human single annotator
    'MedMCQA': None,
    'MMLU\nClinical Knowledge': None,
}

fig, ax = plt.subplots(figsize=(12, 6))
x = np.arange(len(benchmarks))
width = 0.22
modelos_bench = ['GPT-4', 'Med-PaLM 2', 'GPT-3.5']
colors_bench = ['#FF6B6B', '#4ECDC4', '#45B7D1']

for i, modelo in enumerate(modelos_bench):
    vals = benchmarks[modelo].tolist()
    # Replace None with 0 for plotting, track which are None
    plot_vals = [v if v is not None else 0 for v in vals]
    bars = ax.bar(x + i*width, plot_vals, width, label=modelo, color=colors_bench[i])
    # Add value labels
    for j, (bar, val) in enumerate(zip(bars, vals)):
        if val is not None:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                   f'{val}%', ha='center', va='bottom', fontsize=8, fontweight='bold')
        else:
            ax.text(bar.get_x() + bar.get_width()/2, 2,
                   'N/D', ha='center', va='bottom', fontsize=8, color='gray')

# Human baselines
for j, bench in enumerate(benchmarks['Benchmark']):
    hb = human_baselines.get(bench)
    if hb is not None:
        ax.plot([x[j] - 0.1, x[j] + width*len(modelos_bench) + 0.1], [hb, hb],
               'k--', alpha=0.4, linewidth=1)
        label = 'Aprobacion USMLE' if 'MedQA' in bench else 'Humano'
        ax.text(x[j] + width*len(modelos_bench) + 0.12, hb, f'{label}: {hb}%',
               fontsize=8, color='gray', va='center')

ax.set_ylabel('Accuracy (%)', fontsize=12)
ax.set_title('Benchmarks Medicos: Datos publicados en papers revisados por pares',
            fontsize=13, fontweight='bold')
ax.set_xticks(x + width)
ax.set_xticklabels(benchmarks['Benchmark'])
ax.legend(loc='lower right')
ax.set_ylim(0, 100)
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.show()

print('Fuentes:')
print('  - GPT-4: Nori et al., arXiv:2303.13375 (2023)')
print('  - Med-PaLM 2: Singhal et al., Nature Medicine (2024)')
print('  - GPT-3.5: Singhal et al. (2024); John Snow Labs benchmarks')
print('  - N/D = dato no disponible en fuente primaria verificada')
print()
print('Limitaciones de benchmarks:')
print('  - Solo opcion multiple, no reflejan tareas clinicas reales')
print('  - No evaluan generacion libre ni razonamiento complejo')
print('  - No reemplazan evaluacion en TU tarea especifica')

