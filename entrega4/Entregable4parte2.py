# -*- coding: utf-8 -*-
# Generated from: Entregable4parte2.ipynb

# %%
# # AIG4B — Clase 12
# ## Práctica: ECG, EEG y PPG — lo esencial
# 
# Tarea acompañante de la **Clase 12** (señales 1D biomédicas). Los datos vienen hechos; los **modelos los implementan ustedes** siguiendo las especificaciones del PDF de la clase.
# 
# Ejercicios:
# 
# - **1.1** — usar `neurokit2` para estimar bpm en ECG sintético.
# - **1.2** — implementar la **toy CNN** de slide 21 y entrenarla para clasificar arritmias.
# - **2.1** — implementar **EEGNet** siguiendo slide 29 para clasificar carga cognitiva.
# - **3.1** — implementar un **mini-Transformer 1D** y verificar que resuelve la tarea de contexto largo donde una CNN falla.
# 
# Autores: Paulo Veiga y Marco Sanchez Sorondo · Mayo 2026
# %%
# ## Setup
# 
# La primera celda instala lo que falte (necesario en Colab).
# %%
import importlib, subprocess, sys
for pkg in ['wfdb', 'neurokit2']:
    try: importlib.import_module(pkg)
    except ImportError:
        print(f'instalando {pkg}...')
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', pkg])
print('listo')

# %%
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sps
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import wfdb, neurokit2 as nk

np.random.seed(0); torch.manual_seed(0)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
plt.rcParams['figure.figsize'] = (11, 3.5)
plt.rcParams['axes.grid'] = True
print(f'torch v{torch.__version__}  device={device}')

# %%
# ## 1. ECG
# 
# **Idea central (slides 11, 14, 21):** un ECG es una secuencia de números + `fs`. La detección clásica de R-peaks es Pan-Tompkins; la versión AI usa CNNs. Para arritmias, una CNN simple ya supera lo que se podría hacer a mano con reglas R-R.
# 
# Cargamos un registro real de **MIT-BIH** (PhysioNet) y vemos cómo `neurokit2` lo procesa.
# %%
FS_ECG = 360
record = wfdb.rdrecord('100', pn_dir='mitdb', sampto=10*FS_ECG)
ecg_raw = record.p_signal[:, 0]
t_ecg = np.arange(len(ecg_raw)) / FS_ECG

ann = wfdb.rdann('100', 'atr', pn_dir='mitdb', sampto=10*FS_ECG)
r_gt = np.array([s for s, sym in zip(ann.sample, ann.symbol) if sym == 'N'])

_, info = nk.ecg_peaks(ecg_raw, sampling_rate=FS_ECG)
r_nk = info['ECG_R_Peaks']

plt.plot(t_ecg, ecg_raw)
plt.plot(t_ecg[r_gt], ecg_raw[r_gt], 'go', ms=10, mfc='none', label=f'ground truth ({len(r_gt)})')
plt.plot(t_ecg[r_nk], ecg_raw[r_nk], 'rx', ms=10, label=f'neurokit2 ({len(r_nk)})')
plt.xlabel('t [s]'); plt.ylabel('mV'); plt.legend()
plt.title('MIT-BIH "100" — R-peaks detectados vs anotados')
plt.show()

# %%
# ### Helper · generador de ECG sintético
# 
# Función provista para los ejercicios. **No es parte del ejercicio**, solo úsenla.
# %%
def synthetic_ecg(duration_s, fs, hr_bpm):
    """ECG sintético: suma de gaussianas P-Q-R-S-T repetidas a hr_bpm."""
    t = np.arange(0, duration_s, 1/fs); rr = 60.0/hr_bpm
    ecg = np.zeros_like(t)
    waves = [(0.00,0.025,0.15),(0.15,0.010,-0.10),(0.17,0.008,1.20),(0.19,0.010,-0.25),(0.35,0.04,0.35)]
    for k in range(int(duration_s/rr)+1):
        t0 = k * rr
        for off, sigma, amp in waves:
            ecg += amp * np.exp(-((t-t0-off)**2)/(2*sigma**2))
    return t, ecg

# %%
# ### Ejercicio 1.1 — estimar bpm con `neurokit2`
# 
# 1. Generá un ECG sintético de 30 s con frecuencia cardíaca aleatoria entre 50 y 110 bpm (sin imprimir cuál).
# 2. Usá `nk.ecg_peaks` para detectar los R-peaks.
# 3. Estimá la frecuencia cardíaca a partir de los intervalos R-R y comparala con el valor real.
# %%
# Tu código acá
import neurokit2 as nk

# 1. Generar ECG sintético de 30 s con HR aleatoria entre 50 y 110 bpm
rng_1_1 = np.random.default_rng(42)
hr_real = rng_1_1.uniform(50, 110)
fs_ecg = 100
ecg_sig = synthetic_ecg(duration_s=30.0, fs=fs_ecg, hr_bpm=hr_real)

# 2. Detectar R-peaks usando neurokit2
signals, info = nk.ecg_peaks(ecg_sig, sampling_rate=fs_ecg)
r_peaks = info['ECG_R_Peaks']

# 3. Estimación de bpm
rr_intervals = np.diff(r_peaks) / fs_ecg
hr_estimado = 60.0 / np.mean(rr_intervals)

print(f'Picos detectados en los índices: {r_peaks}')
print(f'Frecuencia cardíaca estimada: {hr_estimado:.2f} bpm')
print(f'Frecuencia cardíaca real: {hr_real:.2f} bpm')
print(f'Error de estimación: {abs(hr_estimado - hr_real):.4f} bpm')


# %%
# ### Dataset de arritmias (provisto)
# 
# Tres clases (slide 13): **normal** (RR regular), **AFIB** (RR aleatorio + morfología sucia), **bigeminismo** (RR alternante corto-largo). El dataset y el split train/val vienen hechos; ustedes se ocupan del modelo.
# %%
FS = 100; WIN = 500  # 5 s a 100 Hz

def make_rr_window(label, rng):
    t = np.arange(WIN) / FS
    sig = np.zeros(WIN, dtype=np.float32)
    if label == 0:
        hr = rng.uniform(60, 80)
        rrs = np.full(15, 60/hr) + rng.normal(0, 0.02, 15)
        amps, widths, noise = np.full(15, 1.0), np.full(15, 0.015), 0.05
    elif label == 1:
        rrs = rng.uniform(0.4, 1.4, 15)
        amps = np.full(15, 0.9) + rng.normal(0, 0.10, 15)
        widths = rng.uniform(0.018, 0.025, 15); noise = 0.10
    else:
        s = rng.uniform(0.45, 0.55); l = rng.uniform(1.05, 1.20)
        rrs = np.array([s, l] * 8)[:15] + rng.normal(0, 0.02, 15)
        amps = np.array([1.0, 0.6] * 8)[:15]
        widths = np.array([0.015, 0.025] * 8)[:15]; noise = 0.07
    t_pos = 0.3
    for rr, amp, w in zip(rrs, amps, widths):
        if t_pos >= WIN / FS: break
        sig += amp * np.exp(-((t - t_pos) ** 2) / (2 * w ** 2))
        t_pos += rr
    return sig + noise * rng.standard_normal(WIN).astype(np.float32)

rng = np.random.default_rng(0); n_per = 300
labels_arr = ['normal', 'AFIB', 'bigeminismo']
X = np.stack([make_rr_window(c, rng) for c in [0]*n_per + [1]*n_per + [2]*n_per])
y = np.array([0]*n_per + [1]*n_per + [2]*n_per, dtype=np.int64)
idx = rng.permutation(len(X)); X, y = X[idx], y[idx]
split = int(0.8*len(X))
X_tr, X_va, y_tr, y_va = X[:split], X[split:], y[:split], y[split:]
print(f'Train: {X_tr.shape}  Val: {X_va.shape}  Clases: {labels_arr}')

fig, axes = plt.subplots(1, 3, figsize=(13, 2.5))
for ax, c in zip(axes, [0, 1, 2]):
    ax.plot(np.arange(WIN)/FS, X[y == c][0])
    ax.set_title(labels_arr[c])
plt.tight_layout(); plt.show()

# %%
# ### Ejercicio 1.2 — implementar y entrenar la toy CNN (slide 21)
# 
# Implementá **exactamente** la arquitectura de la slide 21 y entrenala sobre el dataset de arritmias.
# 
# **Arquitectura:**
# ```
# Conv1D(in=1, out=16, kernel_size=5)  →  ReLU  →  MaxPool1D(kernel=2)
# Conv1D(in=16, out=32, kernel_size=5) →  ReLU
# GlobalAvgPool1D  →  Flatten  →  Linear(32, 3)
# ```
# 
# **Entrenamiento:** 15 epochs, optimizer Adam (lr=1e-3), loss `CrossEntropyLoss`, batch size 32.
# 
# **Reportar:**
# - `val_acc` final
# - matriz de confusión 3×3
# - **sensibilidad** (recall) y **especificidad** por clase (las dos métricas clínicas clave)
# 
# *Pista para GlobalAvgPool1D:* en PyTorch se logra con `nn.AdaptiveAvgPool1d(1)` seguido de `nn.Flatten()`.
# %%
# Tu código acá
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import confusion_matrix

# 1. Definir la arquitectura de la Toy CNN
class ToyCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential( 
            nn.Conv1d(in_channels=1, out_channels=16, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=5),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten()
        )
        self.fc = nn.Linear(32, 3)
        
    def forward(self, x):
        return self.fc(self.features(x))

# 2. Configurar Dataloaders
train_dataset = TensorDataset(torch.from_numpy(X_tr).unsqueeze(1).float(), torch.from_numpy(y_tr))
val_dataset = TensorDataset(torch.from_numpy(X_va).unsqueeze(1).float(), torch.from_numpy(y_va))

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# Instanciar el modelo, optimizador y pérdida
torch.manual_seed(42)
model = ToyCNN().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

# 3. Entrenamiento (15 epochs)
n_epochs = 15
for epoch in range(n_epochs):
    model.train()
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        preds = model(xb)
        loss = criterion(preds, yb)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

# 4. Evaluación y Métricas
model.eval()
all_preds = []
all_targets = []
with torch.no_grad():
    for xb, yb in val_loader:
        xb = xb.to(device)
        preds = model(xb).argmax(dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_targets.extend(yb.numpy())

all_preds = np.array(all_preds)
all_targets = np.array(all_targets)

val_acc = (all_preds == all_targets).mean()
print(f'val_acc final: {val_acc:.4f}\n')

cm = confusion_matrix(all_targets, all_preds)
print('Matriz de Confusión 3x3:')
print(cm)
print()

for i in range(3):
    clase_label = labels_arr[i]
    tp = cm[i, i]
    fn = cm[i, :].sum() - tp
    fp = cm[:, i].sum() - tp
    tn = cm.sum() - (tp + fn + fp)
    
    sensibilidad = tp / (tp + fn) if (tp + fn) > 0 else 0
    especificidad = tn / (tn + fp) if (tn + fp) > 0 else 0
    print(f"Clase '{clase_label}':")
    print(f'  - Sensibilidad (Recall): {sensibilidad:.4f}')
    print(f'  - Especificidad: {especificidad:.4f}')


# %%
# ### Preguntas conceptuales — toy CNN (Ej 1.2)
# 
# Justificá brevemente cada decisión de la arquitectura y del entrenamiento. Una o dos oraciones por pregunta.
# 
# 1. ¿Por qué usamos `Conv1D` en vez de `Linear` para procesar la señal?
# 2. `kernel_size=5` a `fs=100 Hz`: ¿cuántos milisegundos cubre el kernel? ¿Por qué ese tamaño tiene sentido para un ECG?
# 3. ¿Para qué sirve `MaxPool1d(2)`?
# 4. ¿Por qué la segunda Conv1D pasa de 16 a 32 filtros (el número crece al profundizar)?
# 5. ¿Qué hace `AdaptiveAvgPool1d(1)`? ¿Qué se gana y qué se pierde al colapsar toda la dimensión temporal?
# 6. ¿Por qué la última capa `Linear` tiene **3** outputs y no hay activación softmax después?
# 7. ¿Por qué `CrossEntropyLoss` es la loss apropiada y por qué usamos `Adam` y no `SGD`?
# %%
# 1. **¿Por qué usamos `Conv1D` en vez de `Linear`?**: Las capas `Linear` tratan cada muestra de la señal de manera independiente, ignorando la estructura secuencial y de vecindad local, y requieren un número enorme de parámetros. En cambio, `Conv1D` comparte pesos a lo largo del tiempo, permitiendo la extracción de características invariantes a la traslación temporal (como la morfología del QRS sin importar en qué segundo ocurra) y con muchos menos parámetros.
# 2. **`kernel_size=5` a `fs=100 Hz`**: Cubre $5 \times 10\text{ ms} = 50\text{ ms}$. Tiene sentido para un ECG porque es una ventana temporal lo suficientemente fina como para captar transiciones rápidas y morfologías locales del ciclo cardíaco (como el pico R o las pendientes de las ondas Q y S, cuyas duraciones individuales rondan las decenas de milisegundos).
# 3. **¿Para qué sirve `MaxPool1d(2)`?**: Reduce a la mitad la dimensión temporal (submuestreo), lo que disminuye el costo computacional de las capas posteriores y, de forma más importante, proporciona invariancia local a pequeñas traslaciones de la señal y aumenta de forma progresiva el campo receptivo de los siguientes kernels.
# 4. **¿Por qué la segunda `Conv1D` pasa de 16 a 32 filtros?**: Al profundizar en la red, la resolución temporal disminuye pero se busca que la red extraiga patrones cada vez más abstractos y complejos (combinaciones de características de bajo nivel). Al aumentar el número de filtros (canales), la red puede capturar una mayor diversidad de estas combinaciones semánticas.
# 5. **¿Qué hace `AdaptiveAvgPool1d(1)`?**: Computa el promedio de cada canal a lo largo de todo el eje temporal restante, colapsándolo a 1 muestra por canal. Con esto se gana invariancia absoluta al tamaño de la entrada (el clasificador puede procesar señales de cualquier longitud) y robustez contra el ruido localizado, pero se pierde toda la información de posicionamiento o secuencia temporal exacta de las características (por ejemplo, no sabremos en qué momento exacto ocurrió una anomalía, solo que ocurrió).
# 6. **¿Por qué la última capa `Linear` tiene 3 outputs y no hay activación softmax?**: Tiene 3 outputs porque corresponden a los *logits* no normalizados para las 3 clases del dataset. No hay softmax al final porque en PyTorch, la función de pérdida `nn.CrossEntropyLoss` realiza de forma interna y optimizada numéricamente la operación Softmax antes de calcular la entropía cruzada.
# 7. **¿Por qué `CrossEntropyLoss` y `Adam`?**: `CrossEntropyLoss` es la pérdida estándar para clasificación multiclase exclusiva. Usamos `Adam` porque adapta dinámicamente la tasa de aprendizaje para cada parámetro basándose en el primer y segundo momento de los gradientes, lo que acelera y estabiliza significativamente la convergencia en comparación con `SGD` clásico, el cual es sensible a la tasa de aprendizaje global y puede quedar atrapado en mínimos locales planos en señales.

# %%
# ## 2. EEG
# 
# **Idea central (slides 25, 27, 29):** EEG es una **matriz** `[canales × tiempo]`, no un vector. **EEGNet** (Lawhern 2018) es el baseline estándar. Lo implementamos siguiendo la slide 29 del PDF.
# 
# Dataset sintético con bandas EEG realistas: carga cognitiva **baja** (predominio α) vs **alta** (predominio β).
# %%
FS_EEG = 128; N_CHANS = 14; N_SAMPLES = 128  # 1 segundo
BANDS = {'delta':(0.5,4), 'theta':(4,8), 'alpha':(8,12), 'beta':(13,30)}

def make_eeg_epoch(label, rng):
    if label == 0:
        amps = {'delta':0.5, 'theta':0.5, 'alpha':1.5, 'beta':0.3}
    else:
        amps = {'delta':0.5, 'theta':0.5, 'alpha':0.3, 'beta':1.5}
    t = np.arange(N_SAMPLES) / FS_EEG
    eeg = np.zeros((N_CHANS, N_SAMPLES), dtype=np.float32)
    for c in range(N_CHANS):
        for band, amp in amps.items():
            lo, hi = BANDS[band]
            eeg[c] += amp * np.sin(2*np.pi*rng.uniform(lo, hi)*t + rng.uniform(0, 2*np.pi))
        eeg[c] += 0.2 * rng.standard_normal(N_SAMPLES)
    return eeg

rng = np.random.default_rng(5); n_per = 300
X_eeg = np.stack([make_eeg_epoch(c, rng) for c in [0]*n_per + [1]*n_per])
y_eeg = np.array([0]*n_per + [1]*n_per, dtype=np.int64)
idx = rng.permutation(len(X_eeg)); X_eeg, y_eeg = X_eeg[idx], y_eeg[idx]
split = int(0.8*len(X_eeg))
X_tr_e, X_va_e = X_eeg[:split], X_eeg[split:]
y_tr_e, y_va_e = y_eeg[:split], y_eeg[split:]
print('Train:', X_tr_e.shape, 'Val:', X_va_e.shape)

# %%
# ### Ejercicio 2.1 — implementar y entrenar EEGNet (slide 29)
# 
# Implementá EEGNet siguiendo la slide 29 del PDF y entrenala para clasificar carga cognitiva (baja vs alta).
# 
# **Arquitectura (slide 29):**
# 
# ```
# Input [batch, 1, C=14, T=128]
# 
# BLOQUE 1 — temporal conv
#   Conv2D(in=1, out=F1=8, kernel=(1, 64), padding=(0, 32), bias=False)
#   BatchNorm2d(F1)
# 
# BLOQUE 2 — depthwise spatial conv (combina canales)
#   Conv2D(in=F1, out=F1*D=16, kernel=(C=14, 1), groups=F1, bias=False)
#   BatchNorm2d(F1*D)  →  ELU  →  AvgPool2D(kernel=(1, 4))  →  Dropout(0.25)
# 
# BLOQUE 3 — separable conv
#   Conv2D depthwise: kernel=(1, 16), groups=F1*D, padding=(0, 8), bias=False
#   Conv2D pointwise: kernel=(1, 1), out=F2=16, bias=False
#   BatchNorm2d(F2)  →  ELU  →  AvgPool2D(kernel=(1, 8))  →  Dropout(0.25)
# 
# HEAD
#   Flatten  →  Linear(→ 2 clases)
# ```
# 
# **Entrenamiento:** 10 epochs, Adam (lr=1e-3), `CrossEntropyLoss`, batch size 32.
# 
# **Tip:** el tensor entra como `[B, C, T]`. Tenés que agregarle la dimensión de canal con `x.unsqueeze(1)` antes del primer Conv2D para que quede `[B, 1, C, T]`.
# %%
# Tu código acá
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

class EEGNet(nn.Module):
    def __init__(self, F1=8, D=2, F2=16, C=14):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=F1, kernel_size=(1, 64), padding=(0, 32), bias=False),
            nn.BatchNorm2d(F1)
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(in_channels=F1, out_channels=F1 * D, kernel_size=(C, 1), groups=F1, bias=False),
            nn.BatchNorm2d(F1 * D),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, 4)),
            nn.Dropout(0.25)
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(in_channels=F1 * D, out_channels=F1 * D, kernel_size=(1, 16), padding=(0, 8), groups=F1 * D, bias=False),
            nn.Conv2d(in_channels=F1 * D, out_channels=F2, kernel_size=(1, 1), bias=False),
            nn.BatchNorm2d(F2),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, 8)),
            nn.Dropout(0.25)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(F2 * 4, 2)
        )
        
    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        return self.classifier(x)

# Configurar Dataloaders
train_dataset_e = TensorDataset(torch.from_numpy(X_tr_e).float(), torch.from_numpy(y_tr_e))
val_dataset_e = TensorDataset(torch.from_numpy(X_va_e).float(), torch.from_numpy(y_va_e))

train_loader_e = DataLoader(train_dataset_e, batch_size=32, shuffle=True)
val_loader_e = DataLoader(val_dataset_e, batch_size=32, shuffle=False)

# Instanciar el modelo, optimizador y pérdida
torch.manual_seed(42)
eeg_model = EEGNet().to(device)
optimizer_e = optim.Adam(eeg_model.parameters(), lr=1e-3)
criterion_e = nn.CrossEntropyLoss()

# Entrenamiento (10 epochs)
n_epochs_e = 10
for epoch in range(n_epochs_e):
    eeg_model.train()
    for xb, yb in train_loader_e:
        xb, yb = xb.to(device), yb.to(device)
        preds = eeg_model(xb)
        loss = criterion_e(preds, yb)
        optimizer_e.zero_grad()
        loss.backward()
        optimizer_e.step()

# Evaluación
eeg_model.eval()
all_preds_e = []
all_targets_e = []
with torch.no_grad():
    for xb, yb in val_loader_e:
        xb = xb.to(device)
        preds = eeg_model(xb).argmax(dim=1)
        all_preds_e.extend(preds.cpu().numpy())
        all_targets_e.extend(yb.numpy())

val_acc_e = (np.array(all_preds_e) == np.array(all_targets_e)).mean()
print(f'EEGNet val_acc final: {val_acc_e:.4f}')


# %%
# ### Preguntas conceptuales — EEGNet (Ej 2.1)
# 
# Justificá brevemente cada decisión de la arquitectura. Una o dos oraciones por pregunta.
# 
# 1. La primera conv tiene kernel `(1, 64)`. ¿Por qué solo en el eje temporal (alto 1) y no en el espacial?
# 2. A `fs=128 Hz`, ¿cuántos segundos cubre un kernel de 64? ¿Por qué tiene sentido ese tamaño para EEG?
# 3. La segunda conv tiene kernel `(C=14, 1)` (todo el eje de canales) y `groups=F1` (depthwise). ¿Qué aprende esa conv? ¿Por qué es depthwise?
# 4. ¿Qué representa el **depth multiplier** `D=2`?
# 5. ¿Por qué `AvgPool2d` en vez de `MaxPool2d`? ¿Y por qué `ELU` en vez de `ReLU`?
# 6. ¿Qué es una **separable convolution** (depthwise + pointwise)? ¿Por qué se usa en vez de una conv estándar?
# 7. El pool del bloque 3 es `(1, 8)`, más grande que el del bloque 2 `(1, 4)`. ¿Por qué se vuelve más agresivo al final?
# 8. ¿Para qué sirve `Dropout(0.25)`?
# %%
# 1. **¿Por qué kernel `(1, 64)` solo en el eje temporal?**: Las señales de cada canal de EEG representan series de tiempo individuales. Aplicar una convolución espacial conjunta al inicio mezclaría la información de los diferentes canales prematuramente. Por tanto, se realiza primero una convolución puramente temporal para extraer filtros de frecuencia individuales para cada canal.
# 2. **Duración de kernel 64 a `fs=128 Hz`**: Cubre $64 / 128 = 0.5$ segundos (500 ms). Tiene sentido para EEG porque ritmos fisiológicos relevantes como las ondas Alfa (8–12 Hz) o Theta (4–8 Hz) ocurren en ciclos que duran entre 80 ms y 250 ms. Un kernel de 500 ms contiene suficiente contexto temporal para capturar varios ciclos completos de estas oscilaciones.
# 3. **¿Qué aprende la convolución espacial `(C=14, 1)`?**: Aprende **filtros espaciales**, es decir, combinaciones lineales de la actividad de los diferentes electrodos del cuero cabelludo (canales) que optimizan la relación señal/ruido para la tarea. Es `depthwise` (groups=F1) para aplicar estos filtros espaciales a cada uno de los filtros temporales aprendidos de forma independiente, evitando una explosión en el número de parámetros.
# 4. **¿Qué representa el depth multiplier `D=2`?**: Indica el número de filtros espaciales que se aprenden para cada filtro temporal del Bloque 1. Es decir, expande el número de canales de 8 (F1) a 16 (F1*D) para permitir que la red extraiga múltiples patrones espaciales por cada patrón temporal.
# 5. **¿Por qué `AvgPool2d` y `ELU`?**:
#    * `AvgPool2d` se prefiere en EEGNet porque suaviza la señal promediando la energía a lo largo del tiempo, lo que preserva información sobre la envolvente de la señal, mientras que `MaxPool2d` solo captaría el pico máximo de ruido.
#    * `ELU` (Exponential Linear Unit) se usa porque produce gradientes más suaves y tiene valores negativos de salida para entradas negativas, lo que ayuda a mantener la media de las activaciones cercana a cero y acelera el entrenamiento en bioseñales oscilatorias.
# 6. **¿Qué es una convolución separable?**: Es una técnica que divide una convolución estándar en dos pasos: primero una convolución espacial `depthwise` (un solo filtro por canal de entrada) y luego una convolución `pointwise` 1x1 que mezcla los canales. Se usa porque reduce drásticamente el número de parámetros y el costo computacional de la red, actuando además como un regularizador que previene el sobreajuste.
# 7. **¿Por qué el pool del bloque 3 es `(1, 8)` (más grande)?**: Al avanzar en las capas, las características extraídas son más abstractas y de menor variación temporal rápida. Realizar un pooling más agresivo permite condensar la información a lo largo del tiempo para preparar los datos para la clasificación final reduciendo el riesgo de overfitting en el clasificador lineal.
# 8. **¿Para qué sirve `Dropout(0.25)`?**: Apaga aleatoriamente el 25% de las neuronas durante el entrenamiento. Sirve como regularizador para forzar a la red a no co-depender de características específicas de los datos sintéticos, mejorando la generalización sobre el validation set.

# %%
# ## 3. CNN ↔ Transformer — campo receptivo y atención (slides 31–33)
# 
# **Lo que el PDF quiere mostrar:** una CNN tiene **campo receptivo finito** (slide 31). Un Transformer mira **toda la secuencia** de una sola pasada (slide 32) usando **self-attention** (slide 33).
# 
# Tarea: ventanas de 30 s con dos *spike trains* — uno al segundo 1, otro al segundo 25 — cada uno a 5 Hz o 12 Hz. Preguntamos si matchean. Para responder hay que comparar dos eventos separados **24 segundos** — fuera del campo receptivo de una CNN razonable.
# %%
FS_LC = 100; DUR_LC = 30.0; T_LC = int(DUR_LC * FS_LC)

def make_long_context(label, rng):
    t = np.arange(T_LC) / FS_LC
    sig = 0.15 * rng.standard_normal(T_LC).astype(np.float32)
    f1 = rng.choice([5, 12])
    sig[(t >= 1) & (t < 2)] += 0.5 * np.sin(2*np.pi*f1*t[(t >= 1) & (t < 2)])
    f2 = f1 if label == 0 else (12 if f1 == 5 else 5)
    sig[(t >= 25) & (t < 26)] += 0.5 * np.sin(2*np.pi*f2*t[(t >= 25) & (t < 26)])
    return sig.astype(np.float32)

rng = np.random.default_rng(11); n_per = 400
X_lc = np.stack([make_long_context(c, rng) for c in [0]*n_per + [1]*n_per])
y_lc = np.array([0]*n_per + [1]*n_per, dtype=np.int64)
idx = rng.permutation(len(X_lc)); X_lc, y_lc = X_lc[idx], y_lc[idx]
split = int(0.8*len(X_lc))
X_tr_lc, X_va_lc = X_lc[:split], X_lc[split:]
y_tr_lc, y_va_lc = y_lc[:split], y_lc[split:]

fig, axes = plt.subplots(2, 1, figsize=(13, 4), sharex=True)
axes[0].plot(np.arange(T_LC)/FS_LC, X_lc[y_lc == 0][0]); axes[0].set_title('match (misma freq en t=1 y t=25)')
axes[1].plot(np.arange(T_LC)/FS_LC, X_lc[y_lc == 1][0]); axes[1].set_title('mismatch (frecuencias distintas)')
axes[-1].set_xlabel('t [s]')
plt.tight_layout(); plt.show()

# %%
# ### Demo · una CNN no puede resolver esta tarea
# 
# Definimos una CNN razonable (3 conv layers con stride=2) y la entrenamos. Su campo receptivo no llega a abarcar los 30 s — queda en *chance level*.
# %%
class SmallCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(1, 16, 7, stride=2, padding=3), nn.ReLU(),
            nn.Conv1d(16, 32, 5, stride=2, padding=2), nn.ReLU(),
            nn.Conv1d(32, 32, 5, stride=2, padding=2), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1), nn.Flatten(),
            nn.Linear(32, 2))
    def forward(self, x): return self.net(x)

def train_clf(model, X_tr, y_tr, X_va, y_va, epochs=10, lr=1e-3):
    opt = torch.optim.Adam(model.parameters(), lr=lr); crit = nn.CrossEntropyLoss()
    loader = DataLoader(TensorDataset(torch.from_numpy(X_tr).float(), torch.from_numpy(y_tr)),
                        batch_size=32, shuffle=True)
    for _ in range(epochs):
        model.train()
        for xb, yb in loader:
            loss = crit(model(xb.to(device).unsqueeze(1)), yb.to(device))
            opt.zero_grad(); loss.backward(); opt.step()
    model.eval()
    with torch.no_grad():
        pred = model(torch.from_numpy(X_va).float().to(device).unsqueeze(1)).argmax(1).cpu().numpy()
    return (pred == y_va).mean()

torch.manual_seed(0)
acc_cnn = train_clf(SmallCNN().to(device), X_tr_lc, y_tr_lc, X_va_lc, y_va_lc)
print(f'SmallCNN val_acc = {acc_cnn:.3f}  (chance = 0.50, CNN limitada por campo receptivo)')

# %%
# ### Ejercicio 3.1 — implementar y entrenar un mini-Transformer 1D (slide 33)
# 
# Implementá un Transformer simple que pueda resolver la tarea. La ventaja sobre la CNN es que **self-attention conecta cualquier par de instantes en una sola capa** — sin importar la distancia.
# 
# **Arquitectura:**
# 
# ```
# 1. Patch embedding
#    Conv1D(in=1, out=64, kernel_size=50, stride=50)   # cada patch = 0.5 s
#    transpose para quedar [B, N_patches=60, dim=64]
# 
# 2. Positional embedding (parámetro aprendido)
#    nn.Parameter de shape [1, N_patches=60, dim=64], inicializado random pequeño
#    Se suma a la salida del patch embedding
# 
# 3. Bloque de atención (con conexión residual)
#    LayerNorm  →  MultiheadAttention(dim=64, num_heads=4, batch_first=True)
#    suma residual con la entrada del bloque
# 
# 4. Bloque feed-forward (con conexión residual)
#    LayerNorm  →  Linear(64 → 128)  →  GELU  →  Linear(128 → 64)
#    suma residual
# 
# 5. Clasificador
#    promedio sobre patches (mean dim=1)  →  Linear(64 → 2)
# ```
# 
# **Entrenamiento:** 10 epochs, Adam (lr=3e-4), `CrossEntropyLoss`.
# 
# Después de entrenar, comparé el `val_acc` con `acc_cnn`. El Transformer debería resolver la tarea (~1.0), mientras que la CNN se queda en chance.
# %%
# Tu código acá
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

class MiniTransformer1D(nn.Module):
    def __init__(self, n_patches=60, dim=64, num_heads=4):
        super().__init__()
        self.patch_embed = nn.Conv1d(in_channels=1, out_channels=dim, kernel_size=50, stride=50)
        self.pos_embed = nn.Parameter(torch.randn(1, n_patches, dim) * 0.02)
        self.ln1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, batch_first=True)
        self.ln2 = nn.LayerNorm(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, 128),
            nn.GELU(),
            nn.Linear(128, dim)
        )
        self.classifier = nn.Linear(dim, 2)
        
    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)
        p = self.patch_embed(x)
        p = p.transpose(1, 2)
        x = p + self.pos_embed
        
        norm_x1 = self.ln1(x)
        attn_out, _ = self.attn(norm_x1, norm_x1, norm_x1)
        x = x + attn_out
        
        norm_x2 = self.ln2(x)
        ff_out = self.ff(norm_x2)
        x = x + ff_out
        
        x_mean = x.mean(dim=1)
        return self.classifier(x_mean)

# Configurar Dataloaders
train_dataset_lc = TensorDataset(torch.from_numpy(X_tr_lc).float(), torch.from_numpy(y_tr_lc))
val_dataset_lc = TensorDataset(torch.from_numpy(X_va_lc).float(), torch.from_numpy(y_va_lc))

train_loader_lc = DataLoader(train_dataset_lc, batch_size=32, shuffle=True)
val_loader_lc = DataLoader(val_dataset_lc, batch_size=32, shuffle=False)

# Instanciar el modelo, optimizador y pérdida
torch.manual_seed(0)
tf_model = MiniTransformer1D().to(device)
optimizer_lc = optim.Adam(tf_model.parameters(), lr=3e-4)
criterion_lc = nn.CrossEntropyLoss()

# Entrenamiento (10 epochs)
n_epochs_lc = 10
for epoch in range(n_epochs_lc):
    tf_model.train()
    for xb, yb in train_loader_lc:
        xb, yb = xb.to(device), yb.to(device)
        preds = tf_model(xb)
        loss = criterion_lc(preds, yb)
        optimizer_lc.zero_grad()
        loss.backward()
        optimizer_lc.step()

# Evaluación
tf_model.eval()
all_preds_lc = []
all_targets_lc = []
with torch.no_grad():
    for xb, yb in val_loader_lc:
        xb = xb.to(device)
        preds = tf_model(xb).argmax(dim=1)
        all_preds_lc.extend(preds.cpu().numpy())
        all_targets_lc.extend(yb.numpy())

val_acc_lc = (np.array(all_preds_lc) == np.array(all_targets_lc)).mean()
print(f'Transformer val_acc final: {val_acc_lc:.4f}')


# %%
# ### Preguntas conceptuales — mini-Transformer (Ej 3.1)
# 
# Justificá brevemente cada decisión de la arquitectura. Una o dos oraciones por pregunta.
# 
# 1. El **patch embedding** es una `Conv1D(kernel=50, stride=50)`. ¿Para qué partir la señal en patches? ¿Qué pasaría si pusiéramos todo el tensor directamente en la atención?
# 2. ¿Cuántos segundos cubre cada patch (a `fs=100 Hz`) y cuántos patches totales hay en una ventana de 30 s?
# 3. ¿Para qué hace falta el **positional embedding**? ¿Qué pasaría sin él en una tarea como esta (match/mismatch)?
# 4. En `MultiheadAttention(dim=64, num_heads=4)`: ¿qué representa cada **cabeza** y por qué se usan varias en paralelo?
# 5. ¿Por qué hay **conexiones residuales** (`e = e + attn_out`)? ¿Qué problema previenen?
# 6. ¿Para qué sirve el **bloque feed-forward** después de la atención si ya hicimos atención? ¿Por qué la dimensión interna es `2 × EMBED`?
# 7. ¿Por qué `LayerNorm` y no `BatchNorm` (como sí usamos en EEGNet)?
# 8. ¿Por qué hacemos `e.mean(dim=1)` al final antes del clasificador?
# %%
# 1. **¿Para qué partir la señal en patches?**: Partir la señal reduce la longitud de la secuencia efectiva que entra al bloque de atención de 3000 muestras a 60 patches. Si pusiéramos la señal original de 3000 puntos directamente en la atención, el costo computacional de la autoatención (que escala cuadráticamente, $O(T^2)$) sería prohibitivo y muy difícil de optimizar para aprender patrones significativos.
# 2. **Duración y cantidad de patches**: A $f_s = 100\text{ Hz}$, un patch de 50 muestras cubre $50 / 100 = 0.5$ segundos. En una ventana de 30 segundos (3000 muestras), hay un total de $3000 / 50 = 60$ patches.
# 3. **¿Para qué hace falta el positional embedding?**: La autoatención es una operación de conjunto (permutation invariant) que procesa los elementos de forma paralela sin noción intrínseca de orden. Sin `positional embedding`, el modelo no sabría si una oscilación ocurrió al principio (t=1 s) o al final (t=25 s), imposibilitando la resolución de la tarea de correspondencia temporal (match/mismatch).
# 4. **¿Qué representa cada cabeza en `MultiheadAttention`?**: Cada cabeza aprende a proyectar los datos en diferentes subespacios de representación y a buscar distintas relaciones de correlación temporal. Tener múltiples cabezas en paralelo permite que una de ellas aprenda a enfocarse en emparejar las frecuencias del inicio con el final, mientras otra cabeza puede evaluar la morfología o descartar el ruido.
# 5. **¿Por qué hay conexiones residuales?**: Permiten que el gradiente fluya directamente a través del bloque sin atenuación durante la retropropagación (backpropagation), previniendo el desvanecimiento del gradiente (vanishing gradient) y facilitando el entrenamiento de modelos de atención.
# 6. **¿Para qué sirve el bloque feed-forward después de la atención?**: La atención es una operación de mezcla lineal (combinación de embeddings). El bloque `feed-forward` añade no-linealidades (mediante GELU) a cada representación de patch de manera independiente, permitiendo proyectar y aprender interacciones no lineales complejas entre las características. Su dimensión interna se expande a $2 \times \text{EMBED}$ (128) para proporcionar capacidad de modelado adicional.
# 7. **¿Por qué `LayerNorm` y no `BatchNorm`?**: En los Transformers, los datos son procesados secuencialmente como colecciones de tokens (patches). `LayerNorm` normaliza a lo largo del canal/dimensión latente para cada elemento individualmente, lo cual es inmune a las variaciones del tamaño de lote (batch size) y mantiene estables las magnitudes de las representaciones a través del tiempo, mientras que `BatchNorm` en secuencias largas puede verse afectado negativamente por la correlación temporal entre lotes.
# 8. **¿Por qué hacemos `e.mean(dim=1)`?**: Para resumir o "colapsar" las representaciones de los 60 patches en un único vector de características global de tamaño 64. Esto proporciona una representación uniforme de toda la señal que puede alimentar directamente a la capa lineal del clasificador binario.

# %%
# ## Cierre — takeaways de la clase
# 
# 1. **Una señal 1D = lista de números + `fs`.** ECG, EEG y PPG son instancias del mismo concepto.
# 2. **El filtro deslizante es el throughline.** Pan-Tompkins → Conv1D → self-attention — lo que cambia es **cuántos parámetros**    y **cuánto contexto** ve cada operación.
# 3. **AI extiende al clásico, no lo reemplaza.** Pan-Tompkins de 1985 sigue siendo baseline en producción.
# 4. **Dimensionalidad escala el problema.** ECG (1D) → PPG (1D + movimiento) → EEG (2D = canales × tiempo). El framework es el mismo;    la arquitectura se adapta.
# 5. **Las CNNs tienen campo receptivo finito.** Cuando la tarea requiere contexto largo, hay que ir a Transformers.
