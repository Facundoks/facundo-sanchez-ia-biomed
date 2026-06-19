# -*- coding: utf-8 -*-
# Generated from: clase4_parte5_calidad_datos.ipynb

# %%
# # Clase 4 — IA Generativa en Biomedica
# ## Parte 5: Calidad de Datos + Integracion Final
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
# # PARTE 5: Calidad de Datos
# 
# > **Garbage in -> Garbage out** (aplica doblemente en LLMs)

# %%
# ### 5.0 Los 3 conjuntos de datos: train, validation y test
# 
# | Set | Para que se usa | Regla clave |
# |-----|----------------|-------------|
# | **Training** | Entrenar / fine-tunear el modelo | El modelo aprende de estos datos |
# | **Validation** | Monitorear entrenamiento, ajustar hiperparametros | Si loss de training baja pero validation sube -> overfitting |
# | **Test** | Evaluar rendimiento final | Nunca visto en entrenamiento. Si lo usas para ajustar, deja de ser test |
# 
# > **Error comun:** Usar el mismo dataset para entrenar y evaluar. Con pocos datos de fine-tuning, el riesgo de overfitting es alto.

# %%
# Visualizacion: overfitting con y sin validation set
np.random.seed(42)

epochs = np.arange(1, 31)
train_loss = 2.5 * np.exp(-0.12 * epochs) + 0.3 + np.random.normal(0, 0.03, len(epochs))
val_loss = 2.5 * np.exp(-0.10 * epochs) + 0.5 + np.random.normal(0, 0.05, len(epochs))
val_loss[18:] += np.linspace(0, 0.8, 12)  # Overfitting a partir del epoch 18

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].plot(epochs, train_loss, 'b-', linewidth=2, label='Train loss')
axes[0].plot(epochs, val_loss, 'r-', linewidth=2, label='Validation loss')
axes[0].axvline(x=18, color='gray', linestyle='--', alpha=0.7)
axes[0].annotate('Overfitting\ncomienza aqui', xy=(18, val_loss[17]),
    xytext=(22, 1.5), fontsize=11, fontweight='bold', color='red',
    arrowprops=dict(arrowstyle='->', color='red'))
axes[0].set_xlabel('Epochs')
axes[0].set_ylabel('Loss')
axes[0].set_title('Con validation set: detectas overfitting', fontsize=13, fontweight='bold')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].plot(epochs, train_loss, 'b-', linewidth=2, label='Train loss (unica metrica)')
axes[1].set_xlabel('Epochs')
axes[1].set_ylabel('Loss')
axes[1].set_title('Sin validation set: todo parece bien...', fontsize=13, fontweight='bold')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# %%
# ### Ejercicio 5.0: Implementar train/val/test split estratificado
# 
# Escribe una funcion que divida un dataset en train, validation y test,
# asegurando que las proporciones de cada clase se mantengan (stratified split).

# %%
def split_estratificado(datos: list[dict], col_label: str,
                          train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, seed=42):
    """
    Divide datos en train, validation y test manteniendo la proporcion de cada clase.

    Pasos:
    1. Agrupar datos por etiqueta (col_label)
    2. Para cada grupo, mezclar con random.shuffle (usando seed)
    3. Dividir cada grupo segun las proporciones
    4. Concatenar los splits de todos los grupos

    Retorna tupla (train, val, test) donde cada uno es una lista de dicts.
    Pista: usa collections.defaultdict para agrupar.
    """
    import random
    # --- TU CODIGO ACA ---
    pass


# ---------- Test ----------
datos_ejemplo = (
    [{"texto": f"positivo {i}", "sent": "positivo"} for i in range(100)] +
    [{"texto": f"negativo {i}", "sent": "negativo"} for i in range(50)] +
    [{"texto": f"neutro {i}", "sent": "neutro"} for i in range(30)]
)

result = split_estratificado(datos_ejemplo, "sent")
if result is not None:
    train, val, test = result
    print(f"Total: {len(datos_ejemplo)}")
    print(f"Train: {len(train)}  Val: {len(val)}  Test: {len(test)}")
    for name, split in [("Train", train), ("Val", val), ("Test", test)]:
        counts = Counter(d["sent"] for d in split)
        total = len(split)
        print(f"  {name} ({total}): " + ", ".join(f"{k}: {v} ({v/total:.0%})" for k, v in sorted(counts.items())))

# %%
# ### 5.1 Deteccion de problemas comunes en datasets medicos

# %%
# Dataset simulado con problemas tipicos
dataset_problematico = pd.DataFrame({
    'id': range(1, 11),
    'texto_clinico': [
        "Pcte masc 45a DBT2 HTA. Metf 850. Enalap 10.",           # Abreviaturas ambiguas
        "Paciente con PCR elevada, se solicita PCR.",               # PCR ambiguo!
        "Se indica metformina para asma.",                          # Error clinico
        "Paciente femenina de 30 anios con infarto agudo.",         # OK
        "",                                                         # Vacio
        "Dx: HTA. Tx: captopril 25mg c/8h.",                       # Abreviaturas
        "Paciente con diabetes. Se indica insulina.",               # Vago (que tipo? que dosis?)
        "Se indica amoxicilina 500mg. ALERGIA: penicilina.",       # Contradiccion!
        "Tratamiento segun protocolo 2004.",                        # Desactualizado
        "Paciente masculino de 55 anios, DM2, HTA. Metformina 850mg/12h, enalapril 10mg/dia.",  # OK
    ],
    'label': ['diabetes', 'infeccion', 'asma', 'cardiologia', None,
              'hipertension', 'diabetes', 'infeccion', 'otros', 'diabetes'],
    'fuente': ['medico', 'medico', 'residente', 'medico', 'error_sistema',
               'medico', 'enfermeria', 'residente', 'archivo', 'medico'],
})

print("Dataset con problemas tipicos:")
print("=" * 90)
for _, row in dataset_problematico.iterrows():
    print(f"  [{row['id']:2d}] {row['texto_clinico'][:70]:<70} | label={row['label']}")

# %%
# Analisis automatizado de calidad
def analizar_calidad_dataset(df, col_texto='texto_clinico', col_label='label'):
    """Analiza problemas comunes en un dataset medico."""
    problemas = []

    for idx, row in df.iterrows():
        texto = str(row[col_texto])
        issues = []

        # 1. Texto vacio o muy corto
        if len(texto.strip()) < 10:
            issues.append("VACIO o muy corto")

        # 2. Label faltante
        if pd.isna(row[col_label]):
            issues.append("LABEL faltante")

        # 3. Abreviaturas ambiguas
        ambiguas = ['PCR', 'IC', 'TA', 'FR']
        for abr in ambiguas:
            if abr in texto and texto.count(abr) > 0:
                # Verificar si aparece mas de una vez (ambiguedad)
                if texto.count(abr) > 1:
                    issues.append(f"AMBIGUO: '{abr}' aparece {texto.count(abr)} veces")

        # 4. Posible contradiccion medicamentosa
        if 'alergia' in texto.lower() and 'penicilina' in texto.lower():
            if 'amoxicilina' in texto.lower() or 'ampicilina' in texto.lower():
                issues.append("CONTRADICCION: alergia a penicilina + derivado")

        # 5. Error clinico conocido
        if 'metformina' in texto.lower() and 'asma' in texto.lower():
            issues.append("ERROR CLINICO: metformina no trata asma")

        # 6. Informacion desactualizada
        anios = re.findall(r'\b(19\d{2}|200[0-9])\b', texto)
        if anios:
            issues.append(f"POSIBLE DESACTUALIZADO: referencia a {', '.join(anios)}")

        if issues:
            problemas.append({'id': row['id'], 'problemas': issues, 'texto': texto[:60]})

    return problemas

problemas = analizar_calidad_dataset(dataset_problematico)

print("\nProblemas detectados:")
print("=" * 70)
for p in problemas:
    print(f"\n  ID {p['id']}: {p['texto']}...")
    for issue in p['problemas']:
        print(f"    -> {issue}")

total = len(dataset_problematico)
con_problemas = len(problemas)
print(f"\n\nResumen: {con_problemas}/{total} registros con problemas ({con_problemas/total*100:.0f}%)")

# %%
# ### 5.2 Desbalance de datos

# %%
# DATOS SIMULADOS para ilustrar el problema de desbalance (tipico en medicina).
# Las cantidades y F1-scores son ficticios; el patron es real.
np.random.seed(42)

categorias_medicas = {
    'Hipertension': 450,
    'Diabetes': 320,
    'EPOC': 85,
    'Insuf. Cardiaca': 62,
    'Cancer pulmon': 28,
    'Enf. Raras': 5,
}

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Distribucion original
cats = list(categorias_medicas.keys())
vals = list(categorias_medicas.values())
colors_bar = ['#4CAF50' if v > 100 else '#FFD93D' if v > 20 else '#FF6B6B' for v in vals]

axes[0].barh(cats, vals, color=colors_bar, edgecolor='black')
axes[0].set_xlabel('Cantidad de ejemplos')
axes[0].set_title('Dataset Desbalanceado\n(tipico en medicina)', fontsize=13, fontweight='bold')
for i, v in enumerate(vals):
    axes[0].text(v + 5, i, str(v), va='center', fontsize=10)
axes[0].axvline(x=100, color='red', linestyle='--', alpha=0.5, label='Minimo recomendado')
axes[0].legend()

# Impacto en performance
performance = {
    'Hipertension': 0.92,
    'Diabetes': 0.89,
    'EPOC': 0.75,
    'Insuf. Cardiaca': 0.68,
    'Cancer pulmon': 0.55,
    'Enf. Raras': 0.30,
}
n_ejemplos = list(categorias_medicas.values())
f1_scores = list(performance.values())

axes[1].scatter(n_ejemplos, f1_scores, s=200, c=colors_bar, edgecolors='black', linewidths=1.5, zorder=5)
for cat, n, f1 in zip(cats, n_ejemplos, f1_scores):
    axes[1].annotate(cat, (n, f1), textcoords="offset points", xytext=(5, 8), fontsize=9)
axes[1].set_xlabel('Cantidad de ejemplos', fontsize=12)
axes[1].set_ylabel('F1-Score', fontsize=12)
axes[1].set_title('Relacion: Cantidad de datos vs Performance',
                  fontsize=13, fontweight='bold')
axes[1].set_xscale('log')
axes[1].grid(True, alpha=0.3)
axes[1].axhline(y=0.7, color='red', linestyle='--', alpha=0.5, label='Threshold aceptable')
axes[1].legend()

plt.tight_layout()
plt.show()

print("Leccion: Las categorias con pocos ejemplos tienen peor performance.")
print("En medicina, las enfermedades raras son las mas dificiles de modelar")
print("y a la vez las que mas se beneficiarian de automatizacion.")

# %%
# ### 5.3 Datos sinteticos: generacion y riesgos

# %%
# Un template bien disenado para generar datos sinteticos con un LLM:
# 
# ```
# Genera un caso clinico ficticio pero realista con la siguiente estructura:
# 
# PARAMETROS:
# - Patologia: {patologia}
# - Sexo: {sexo}
# - Edad: {edad} anios
# - Comorbilidades: {comorbilidades}
# 
# FORMATO DE SALIDA:
# 1. Motivo de consulta (1-2 lineas)
# 2. Antecedentes relevantes
# 3. Examen fisico (signos vitales + hallazgos)
# 4. Estudios complementarios
# 5. Diagnostico presuntivo
# 6. Plan terapeutico
# 
# REGLAS:
# - NO inventes nombres de drogas
# - Las dosis deben ser reales y actualizadas
# - Incluir alergias (o "sin alergias conocidas")
# - Ser coherente internamente
# ```
# 
# **Variaciones a generar:**
# 1. IAM con elevacion del ST - masculino, 62a, HTA + tabaquismo
# 2. IAM con elevacion del ST - femenino, 78a, DM2 + dislipidemia
# 3. IAM con elevacion del ST - masculino, 45a, obesidad + sedentarismo
# 
# Ventajas: escala rapido, menos problemas de privacidad, control de distribucion.

# %%
# Riesgos de datos sinteticos - visualizacion
riesgos_sinteticos = pd.DataFrame({
    'Riesgo': ['Amplificar errores', 'Datos irreales', 'Sesgos ocultos',
               'Falta diversidad', 'Colapso de modelos'],
    'Severidad': [5, 4, 5, 3, 5],
    'Frecuencia': [4, 3, 5, 4, 3],
    'Mitigacion': [
        'Validar muestra con expertos',
        'Verificar coherencia clinica',
        'Auditar distribucion demografica',
        'Mezclar con datos reales',
        'Limitar proporcion sintetica'
    ]
})

fig, ax = plt.subplots(figsize=(8, 6))
scatter = ax.scatter(riesgos_sinteticos['Frecuencia'], riesgos_sinteticos['Severidad'],
                    s=400, c='#FF6B6B', edgecolors='black', linewidths=2, alpha=0.7, zorder=5)

for _, row in riesgos_sinteticos.iterrows():
    ax.annotate(row['Riesgo'], (row['Frecuencia'], row['Severidad']),
               textcoords="offset points", xytext=(10, 5), fontsize=10, fontweight='bold')

ax.set_xlabel('Frecuencia (1-5)', fontsize=12)
ax.set_ylabel('Severidad (1-5)', fontsize=12)
ax.set_title('Matriz de Riesgos: Datos Sinteticos en Medicina',
            fontsize=14, fontweight='bold')
ax.set_xlim(1, 6)
ax.set_ylim(1, 6)
ax.axhline(y=3, color='orange', linestyle='--', alpha=0.3)
ax.axvline(x=3, color='orange', linestyle='--', alpha=0.3)
ax.fill_between([3, 6], 3, 6, alpha=0.05, color='red')
ax.text(4.5, 5.5, 'ALTO\nRIESGO', fontsize=12, color='red', alpha=0.3, ha='center')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print("\nMitigaciones:")
for _, row in riesgos_sinteticos.iterrows():
    print(f"  {row['Riesgo']}: {row['Mitigacion']}")

# %%
# ### Ejercicio 5.1: Construir un auditor automatico de datasets clinicos
# 
# Escribe una funcion que detecte problemas comunes en un dataset medico:
# textos vacios, labels faltantes, contraindicaciones, alergias cruzadas, dosis peligrosas.

# %%
ALERGIAS_CRUZADAS = {
    "penicilina": ["amoxicilina", "ampicilina"],
    "warfarina": ["warfarina"],
    "sulfas": ["sulfametoxazol", "trimetoprima"],
}

CONTRAINDICACIONES = {
    "aspirina": ["ninio", "pediatr"],
    "enalapril": ["embaraz"],
}


def auditar_dataset(dataset: list[dict]) -> list[dict]:
    """
    Recibe lista de dicts con claves 'input', 'output', 'label'.

    Para cada registro, detecta problemas y retorna una lista de dicts con:
    - 'indice': posicion en la lista (0-indexed)
    - 'problemas': lista de strings describiendo cada problema

    Solo incluir registros que tienen al menos un problema.

    Problemas a detectar:
    1. Input vacio (len(input.strip()) == 0)
    2. Label nulo o vacio
    3. Contradiccion alergia-medicamento: si el input menciona "ALERGIA" a X
       y el output indica un medicamento de ALERGIAS_CRUZADAS[X]
    4. Contraindicacion: si el input menciona alguna condicion de CONTRAINDICACIONES[med]
       y el output menciona ese medicamento
    5. Dosis peligrosa: si el output contiene un numero > 1000 seguido de "mg"
       (usar re.findall(r'(\d+)\s*mg', output))

    Busquedas case-insensitive.
    """
    import re
    # --- TU CODIGO ACA ---
    pass


# ---------- Test ----------
dataset_test = [
    {"input": "Pcte con dolor toracico", "output": "Administrar aspirina 300mg", "label": "urgencia"},
    {"input": "Paciente con cefalea", "output": "Se indica ibuprofeno 2400mg diarios", "label": "neurologia"},
    {"input": "Ninio de 2 anios con fiebre", "output": "Administrar aspirina infantil", "label": "pediatria"},
    {"input": "", "output": "Sin datos", "label": None},
    {"input": "Embarazada 32 semanas con HTA", "output": "Se indica enalapril 10mg", "label": "obstetricia"},
    {"input": "Paciente con FA, ALERGIA a warfarina", "output": "Se indica warfarina 5mg", "label": "cardiologia"},
    {"input": "Paciente con neumonia", "output": "Se indica amoxicilina 500mg", "label": "infectologia"},
    {"input": "Pte ALERGIA a penicilina, infeccion urinaria", "output": "Se indica amoxicilina 875mg", "label": "infectologia"},
]

problemas = auditar_dataset(dataset_test)
if problemas is not None:
    print(f"Registros con problemas: {len(problemas)} de {len(dataset_test)}\n")
    for p in problemas:
        print(f"Registro {p['indice']}: {dataset_test[p['indice']]['input'][:50]}")
        print(f"  Output: {dataset_test[p['indice']]['output'][:50]}")
        for prob in p['problemas']:
            print(f"  -> {prob}")
        print()

# %%
# Validacion: estos son los registros que deberian tener problemas
esperados = {1, 2, 3, 4, 5, 7}  # indices 0-indexed

if problemas is not None:
    encontrados = {p['indice'] for p in problemas}
    if encontrados == esperados:
        print("EXCELENTE! Detectaste todos los registros problematicos.")
    else:
        faltaron = esperados - encontrados
        extras = encontrados - esperados
        if faltaron: print(f"Te faltaron: {faltaron}")
        if extras: print(f"Detectaste de mas: {extras}")

# %%
# ---
# # PARTE 6: Integracion Final — El Flujo Completo
# 
# Resumen del flujo recomendado:
# 1. Definir tarea
# 2. Metricas
# 3. Prompting
# 4. RAG
# 5. Fine-tuning
# 6. Evaluar en datos reales
# 
# > **El 80% del trabajo es curacion de datos y evaluacion.**
# > **El 15% es ingenieria de prompts.**
# > **Solo el 5% es entrenamiento.**

# %%
# Visualizacion del flujo completo
fig, ax = plt.subplots(figsize=(14, 7))

# Flujo como diagrama horizontal
pasos = [
    ('1. Definir\nTarea', '#E3F2FD', 0),
    ('2. Definir\nMetricas', '#E8F5E9', 1),
    ('3. Medir\nBaseline', '#FFF3E0', 2),
    ('4. Mejorar\nPrompts', '#F3E5F5', 3),
    ('5. Probar\nRAG', '#FCE4EC', 4),
    ('6. Fine-\nTuning?', '#FFEBEE', 5),
    ('7. Evaluar\nen datos\nreales', '#E0F7FA', 6),
]

for nombre, color, x in pasos:
    rect = plt.Rectangle((x*1.8, 1), 1.4, 2, facecolor=color, edgecolor='black', linewidth=2)
    ax.add_patch(rect)
    ax.text(x*1.8 + 0.7, 2, nombre, ha='center', va='center', fontsize=11, fontweight='bold')
    if x < 6:
        ax.annotate('', xy=((x+1)*1.8, 2), xytext=(x*1.8+1.4, 2),
                   arrowprops=dict(arrowstyle='->', color='black', lw=2))

# Proporcion de esfuerzo
esfuerzos = [('Datos + Evaluacion\n80%', 5, '#FF6B6B'),
             ('Prompts\n15%', 2.5, '#4ECDC4'),
             ('Entrenamiento\n5%', 0.8, '#45B7D1')]

bottom = 0
for label, width, color in esfuerzos:
    rect = plt.Rectangle((bottom + 1, -0.5), width, 0.8, facecolor=color,
                         edgecolor='black', linewidth=1.5, alpha=0.8)
    ax.add_patch(rect)
    ax.text(bottom + 1 + width/2, -0.1, label, ha='center', va='center', fontsize=10, fontweight='bold')
    bottom += width

ax.text(0.5, -0.1, 'Esfuerzo:', fontsize=11, fontweight='bold')

ax.set_xlim(-0.5, 13)
ax.set_ylim(-1.5, 4)
ax.set_aspect('equal')
ax.axis('off')
ax.set_title('Flujo Recomendado para Proyectos con LLMs en Biomedica',
            fontsize=16, fontweight='bold', pad=20)

plt.tight_layout()
plt.show()

# %%
# ### Ejercicio Final: Pipeline de evaluacion end-to-end
# 
# Combina todo lo aprendido: implementa un pipeline que tome pares
# (referencia, resumen generado) y produzca un reporte con metricas automaticas
# y deteccion de omisiones criticas.

# %%
from rouge_score import rouge_scorer

def pipeline_evaluacion(datos: list[dict], campos_criticos: list[str]) -> pd.DataFrame:
    """
    Recibe lista de dicts con claves 'referencia' y 'generado', y una lista de campos_criticos.

    Para cada ejemplo, calcula:
    - 'rouge1_f1': ROUGE-1 F1 entre referencia y generado
    - 'campos_faltantes': campos_criticos ausentes en el generado (case-insensitive)
    - 'cobertura': proporcion de campos criticos presentes
    - 'seguro': True si cobertura == 1.0

    Retorna DataFrame con estas columnas.
    Pista: usa rouge_scorer.RougeScorer(['rouge1']).
    """
    # --- TU CODIGO ACA ---
    pass


# ---------- Test ----------
datos_eval = [
    {"referencia": "IAM anterior. HTA, DM2. Alergia penicilina. Troponina 2.8.",
     "generado": "IAM anterior. HTA y DM2. Troponina elevada. Alergia a penicilina."},
    {"referencia": "TEP bilateral. Antecedente TVP. SatO2 88%.",
     "generado": "Paciente con disnea y desaturacion. Angiotomografia positiva."},
    {"referencia": "Sospecha Kawasaki: fiebre, rash, conjuntivitis. Alergia ibuprofeno.",
     "generado": "Paciente pediatrica con fiebre y rash. Se sospecha cuadro viral."},
]
campos = ["alergia", "troponina", "IAM", "TEP", "TVP", "Kawasaki", "sat"]

df_res = pipeline_evaluacion(datos_eval, campos)
if df_res is not None:
    print(df_res.to_string(index=False))
    print(f"\nROUGE-1 F1 promedio: {df_res['rouge1_f1'].mean():.3f}")
    print(f"Cobertura promedio:  {df_res['cobertura'].mean():.1%}")
    print(f"Resumenes seguros:   {df_res['seguro'].sum()}/{len(df_res)}")

# %%
# ---
# # Quiz Final

# %%
# **1.** Que optimizas PRIMERO al implementar un LLM en un hospital?
# - a) El modelo (usar GPT-5)
# - b) La evaluacion
# - c) El fine-tuning
# - d) Los datos
# 
# **2.** Un resumen clinico tiene ROUGE-L = 0.92 pero omite una alergia a penicilina. Es aceptable?
# - a) Si, el ROUGE es excelente
# - b) No, omitir una alergia es un error critico
# - c) Depende del contexto
# 
# **3.** Que porcentaje del trabajo en un proyecto con LLMs es entrenamiento del modelo?
# - a) 50%
# - b) 30%
# - c) 5%
# - d) 80%
# 
# **4.** Cuando SI necesitas fine-tuning?
# - a) Siempre que quieras mejorar el modelo
# - b) Cuando el conocimiento necesario no esta en el entrenamiento general
# - c) Cuando el prompt tiene mas de 100 tokens
# - d) Cuando el modelo responde en ingles
# 
# **5.** Cual es el riesgo principal de entrenar un LLM con datos sinteticos generados por otro LLM?
# - a) Es muy caro
# - b) Colapso de modelos y amplificacion de errores
# - c) Los datos son demasiado buenos
# - d) No hay riesgo
# 
# <details>
# <summary><b>Click para ver respuestas</b></summary>
# 
# **1: b)** Evaluacion primero: sin saber donde estas, no podes mejorar nada.
# 
# **2: b)** Las metricas automaticas NO capturan seguridad clinica. Omitir una alergia puede ser mortal.
# 
# **3: c)** 80% es curacion de datos + evaluacion, 15% es prompts, solo 5% es entrenamiento.
# 
# **4: b)** Fine-tuning cuando hay lenguaje/formato/tarea muy especificos que no se resuelven con prompting + RAG.
# 
# **5: b)** Entrenar LLM con output de otro LLM puede degradar calidad y amplificar sesgos/errores.
# 
# </details>

# %%
# ---
# ## Takeaway Final
# 
# > **No gana el que entrena mas, gana el que mejor evalua y usa el modelo.**
# 
# - Los LLMs actuales ya saben mucho de medicina
# - El cuello de botella suele ser la **evaluacion**, no el modelo
# - Un modelo mal evaluado es **mas peligroso** que no tener modelo
# - La **calidad de datos** determina el techo de cualquier sistema
# 
# ### Flujo recomendado:
# **Evaluacion** (saber donde estas) -> **Prompt** (mejorar rapido) -> **Datos** (mejorar en serio) -> **Modelo** (solo si todo lo anterior no alcanza)

