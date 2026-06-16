# -*- coding: utf-8 -*-
# Generated from: Entrega 3 - Parte 1 - Computer Vision con MedMNIST.ipynb

# %%
# # Clase 7 — Visión Artificial en Biomedicina
# ## Notebook de práctica para alumnos
# 
# Esta notebook tiene **dos partes**:
# 
# - **Parte 1 — Referencia (PathMNIST, resuelto):** es la notebook completa que vimos en clase. Pueden correrla, leerla y usarla como molde. **No hace falta modificarla.**
# - **Parte 2 — Práctica (PneumoniaMNIST, a completar):** replican el pipeline entero sobre otro dataset de MedMNIST que tiene dimensiones distintas (grayscale en vez de RGB, binario en vez de multiclase). Eso los obliga a adaptar shapes, `in_channels`, `n_classes`, y a experimentar con modificaciones de arquitectura e hiperparámetros.
# - **Parte 3 — Desafío opcional:** entrenar un modelo sobre un dataset **3D** de MedMNIST (volúmenes, no imágenes planas). No está guiado — lo hacen si quieren ir más lejos.
# 
# Las celdas con huecos están marcadas con `# TODO`. Donde dice `raise NotImplementedError(...)` tienen que reemplazar esa línea por su implementación.
# 
# > **Sugerencia:** corré esta notebook en Google Colab con GPU (Runtime → Change runtime type → GPU).

# %%
# ## Setup

# %%
!pip install -q medmnist torch torchvision matplotlib seaborn scikit-learn tqdm

# %%
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
import torchvision.transforms as T
from torchvision.models import resnet18, ResNet18_Weights

from sklearn.metrics import accuracy_score, confusion_matrix

import medmnist
from medmnist import INFO

# Reproducibilidad
torch.manual_seed(0)
np.random.seed(0)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'MedMNIST version: {medmnist.__version__}')
print(f'PyTorch version:  {torch.__version__}')
print(f'Device:           {device}')

# %%
# ---
# ## 1. Repaso breve de la clase
# 
# En la clase 7 vimos los fundamentos conceptuales de visión artificial. Resumen de lo que vamos a usar:
# 
# - **Una imagen es un tensor.** Escala de grises → matriz `(H × W)`; RGB → tensor `(H × W × 3)`. Para una red neuronal, es solo un arreglo de números.
# - **Aplanar y meter en un MLP no escala.** Una imagen de 224×224×3 tiene 150.528 valores. Conectar eso a una sola capa de 1000 neuronas ya son 150M de parámetros, y además **se pierde la estructura espacial**: píxeles vecinos dejan de ser vecinos después del flatten.
# - **Las convoluciones resuelven dos problemas a la vez:** (1) **localidad** — un kernel 3×3 solo mira su vecindario, (2) **equivarianza traslacional** — el mismo kernel detecta el mismo patrón en cualquier parte de la imagen. Consecuencia: muchos menos parámetros y mejor generalización.
# - **Historia corta de las CNN:** LeNet (1998) → AlexNet (2012, el *big bang* del DL) → VGG (2014) → **ResNet (2015, el que vamos a usar hoy)** → Vision Transformers (2020).
# - **En biomedicina, la arquitectura casi nunca se inventa.** Se toma una red probada en ImageNet (ResNet, DenseNet, EfficientNet) y se entrena o ajusta sobre el dataset médico. El trabajo duro está en **curar el dataset**.
# 
# Con eso, vamos a la práctica.

# %%
# ---
# ## 2. El dataset: MedMNIST
# 
# **MedMNIST** ([medmnist.com](https://medmnist.com/)) es una colección de 18 datasets de imágenes médicas, estandarizados al formato MNIST (imágenes pequeñas, 28×28 o 64/128/224). Cubre modalidades muy distintas: radiografías, dermatoscopía, histopatología, retinografía, ecografía, OCT, tomografías 3D.
# 
# Su objetivo es servir como **benchmark introductorio y reproducible** para clasificación biomédica — el equivalente a MNIST/CIFAR, pero con datos reales de medicina.
# 
# Para esta clase vamos a usar **PathMNIST**: imágenes de histopatología de cáncer colorrectal. Es multi-clase (9 tipos de tejido), a color (3 canales) y tiene ~100k muestras.

# %%
# Elegimos el dataset
data_flag = 'pathmnist'
info = INFO[data_flag]

print(f'Dataset:     {info["python_class"]}')
print(f'Descripción: {info["description"][:200]}...')
print(f'Tarea:       {info["task"]}')
print(f'Canales:     {info["n_channels"]}')
print(f'# clases:    {len(info["label"])}')
print(f'Clases:')
for k, v in info['label'].items():
    print(f'   {k}: {v}')

# %%
# Cargamos la clase específica del dataset
DataClass = getattr(medmnist, info['python_class'])
n_classes = len(info['label'])
class_names = [info['label'][str(i)] for i in range(n_classes)]

# Transformación básica: pasar a tensor y normalizar a [-1, 1]
transform = T.Compose([
    T.ToTensor(),
    T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

train_ds = DataClass(split='train', transform=transform, download=True, size=28)
val_ds   = DataClass(split='val',   transform=transform, download=True, size=28)
test_ds  = DataClass(split='test',  transform=transform, download=True, size=28)

print(f'\nTrain: {len(train_ds):>6}  |  Val: {len(val_ds):>6}  |  Test: {len(test_ds):>6}')

# Forma de una imagen
img, label = train_ds[0]
print(f'Imagen: shape={tuple(img.shape)}, dtype={img.dtype}, rango=[{img.min():.2f}, {img.max():.2f}]')
print(f'Label : {label} (tipo {type(label).__name__})')

# %%
# ### Visualización: ejemplos por clase
# 
# Antes de entrenar cualquier modelo, **siempre** conviene mirar los datos.

# %%
def denormalize(img_tensor):
    """De (C,H,W) normalizado a [-1,1] a (H,W,C) en [0,1] para mostrar."""
    return (img_tensor.permute(1, 2, 0) * 0.5 + 0.5).clamp(0, 1).cpu().numpy()

n_per_class = 3
fig, axes = plt.subplots(n_per_class, n_classes, figsize=(n_classes * 1.4, n_per_class * 1.6))
shown = {c: 0 for c in range(n_classes)}

for img, label in train_ds:
    c = int(np.asarray(label).flatten()[0])
    if shown[c] < n_per_class:
        row = shown[c]
        ax = axes[row, c]
        ax.imshow(denormalize(img))
        ax.axis('off')
        if row == 0:
            ax.set_title(f'{c}: {class_names[c][:12]}', fontsize=8)
        shown[c] += 1
    if all(v >= n_per_class for v in shown.values()):
        break

plt.suptitle(f'PathMNIST — {n_per_class} ejemplos por clase', fontsize=12)
plt.tight_layout()
plt.show()

# %%
# Distribución de clases en el train set
train_labels = np.asarray(train_ds.labels).flatten()
counts = np.bincount(train_labels, minlength=n_classes)

plt.figure(figsize=(10, 4))
bars = plt.bar(range(n_classes), counts, color=sns.color_palette('husl', n_classes))
plt.xticks(range(n_classes), [f'{i}\n{class_names[i][:8]}' for i in range(n_classes)], fontsize=8)
plt.ylabel('# muestras en train')
plt.title('Distribución de clases — PathMNIST')
for bar, c in zip(bars, counts):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height(), str(c),
             ha='center', va='bottom', fontsize=8)
plt.tight_layout()
plt.show()

# %%
# ### Subset para entrenamiento rápido
# 
# PathMNIST tiene ~90k imágenes de entrenamiento. Para que esta notebook corra en tiempo razonable (sobre todo sin GPU), usamos un subconjunto. En un proyecto real usaríamos todas las imágenes y más epochs.

# %%
BATCH_SIZE = 128
TRAIN_SUBSET = 8000   # de los ~90k disponibles
TEST_SUBSET  = 3000

rng = np.random.RandomState(0)
train_idx = rng.choice(len(train_ds), TRAIN_SUBSET, replace=False)
test_idx  = rng.choice(len(test_ds),  TEST_SUBSET,  replace=False)

train_sub = Subset(train_ds, train_idx)
test_sub  = Subset(test_ds,  test_idx)

train_loader = DataLoader(train_sub, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
val_loader   = DataLoader(val_ds,    batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
test_loader  = DataLoader(test_sub,  batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

print(f'Train subset: {len(train_sub)} | Val: {len(val_ds)} | Test subset: {len(test_sub)}')

# %%
# ### Funciones auxiliares de entrenamiento y evaluación
# 
# Las mismas funciones las vamos a reusar con los tres modelos.

# %%
def to_target(y):
    """MedMNIST devuelve labels como tensor (B, 1). Los queremos como (B,) long."""
    return y.squeeze(-1).long()

def train_model(model, train_loader, val_loader, epochs=3, lr=1e-3, name='modelo'):
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    history = {'train_loss': [], 'val_acc': []}

    for epoch in range(epochs):
        model.train()
        running = 0.0
        for x, y in train_loader:
            x, y = x.to(device), to_target(y).to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
            running += loss.item() * x.size(0)
        train_loss = running / len(train_loader.dataset)

        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), to_target(y).to(device)
                correct += (model(x).argmax(1) == y).sum().item()
                total += y.size(0)
        val_acc = correct / total

        history['train_loss'].append(train_loss)
        history['val_acc'].append(val_acc)
        print(f'[{name}] Epoch {epoch+1}/{epochs}  loss={train_loss:.4f}  val_acc={val_acc:.4f}')
    return history


def evaluate(model, loader):
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), to_target(y).to(device)
            preds.append(model(x).argmax(1).cpu())
            trues.append(y.cpu())
    preds = torch.cat(preds).numpy()
    trues = torch.cat(trues).numpy()
    return accuracy_score(trues, preds), preds, trues


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def show_predictions(model, dataset, n=10, title=''):
    """Muestra n ejemplos con predicción del modelo. Verde = correcto, rojo = error."""
    model.eval()
    cols = 5
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.2, rows * 2.4))
    axes = axes.flatten()
    rng_local = np.random.RandomState(42)
    idxs = rng_local.choice(len(dataset), n, replace=False)

    for ax, idx in zip(axes, idxs):
        img, label = dataset[idx]
        true = int(np.asarray(label).flatten()[0])
        with torch.no_grad():
            logits = model(img.unsqueeze(0).to(device))
            probs = F.softmax(logits, dim=1)[0]
            pred = int(probs.argmax().item())
            conf = float(probs[pred].item())
        ok = (pred == true)
        ax.imshow(denormalize(img))
        ax.axis('off')
        ax.set_title(
            f'Real: {class_names[true][:10]}\nPred: {class_names[pred][:10]} ({conf:.0%})',
            fontsize=8, color=('green' if ok else 'red')
        )
    for ax in axes[n:]:
        ax.axis('off')
    plt.suptitle(title, fontsize=12)
    plt.tight_layout()
    plt.show()


def plot_confusion(preds, trues, title=''):
    cm = confusion_matrix(trues, preds, labels=list(range(n_classes)))
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=[c[:8] for c in class_names],
                yticklabels=[c[:8] for c in class_names], cbar=False)
    plt.xlabel('Predicho'); plt.ylabel('Real')
    plt.title(title)
    plt.tight_layout()
    plt.show()

# %%
# ---
# ## 3. Clasificador básico: MLP (Multi-Layer Perceptron)
# 
# **Idea:** aplanamos la imagen de 28×28×3 = 2.352 valores y la pasamos por una red *fully-connected*. Es el enfoque "ingenuo" que discutimos en la clase.
# 
# **Qué esperamos:** funciona, pero:
# - Tiene muchos parámetros por capa.
# - **Pierde la estructura espacial** — para la red, un píxel en la esquina superior y su vecino a la derecha son dos entradas sin relación.
# - **No es invariante a traslaciones** — si el mismo patrón aparece corrido, la red lo "ve" distinto.

# %%
class MLP(nn.Module):
    def __init__(self, in_shape=(3, 28, 28), n_classes=9):
        super().__init__()
        in_features = int(np.prod(in_shape))
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, n_classes),
        )

    def forward(self, x):
        return self.net(x)

mlp = MLP(in_shape=(3, 28, 28), n_classes=n_classes)
print(mlp)
print(f'\nParámetros entrenables: {count_params(mlp):,}')

# %%
hist_mlp = train_model(mlp, train_loader, val_loader, epochs=3, lr=1e-3, name='MLP')

acc_mlp, preds_mlp, trues_mlp = evaluate(mlp, test_loader)
print(f'\n==> Accuracy en test (MLP): {acc_mlp:.4f}')

# %%
# Ejemplos de predicción
show_predictions(mlp, test_sub, n=10, title=f'MLP — test acc = {acc_mlp:.3f}')

# %%
# ---
# ## 4. Agregando convoluciones: CNN simple
# 
# Ahora usamos **capas convolucionales**. Cada filtro es una matriz pequeña (aquí 3×3) que se desliza sobre la imagen detectando un patrón local. La red aprende los pesos de los filtros por sí sola.
# 
# Arquitectura: tres bloques `Conv → ReLU → MaxPool`, seguidos de *global average pooling* y una capa lineal final.
# 
# **Qué esperamos frente al MLP:**
# - **Menos parámetros** (los filtros son chicos y se comparten en toda la imagen).
# - **Mejor accuracy** porque la red ahora respeta la estructura espacial.
# - **Jerarquía de features:** primeras capas aprenden bordes/texturas, capas profundas combinan en patrones más complejos.

# %%
class SimpleCNN(nn.Module):
    def __init__(self, in_channels=3, n_classes=9):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),                     # 28 -> 14
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),                     # 14 -> 7
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),             # 7 -> 1 (global avg pool)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.2),
            nn.Linear(128, n_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))

cnn = SimpleCNN(in_channels=3, n_classes=n_classes)
print(cnn)
print(f'\nParámetros entrenables: {count_params(cnn):,}  (vs MLP: {count_params(mlp):,})')

# %%
# ### Bonus conceptual: una CNN acepta imágenes de tamaño variable, un MLP no
# 
# Mirá la arquitectura que acabamos de definir. Fijate que **ninguna capa depende explícitamente del tamaño espacial de la entrada**:
# 
# - `Conv2d` es un filtro pequeño (3×3) que se desliza sobre la imagen — funciona igual sobre `28×28`, `64×64` o `224×224`.
# - `MaxPool2d(2)` divide por dos sin saber cuánto mide la entrada.
# - `AdaptiveAvgPool2d(1)` es **la pieza clave**: colapsa cualquier mapa espacial que reciba a un único valor por canal. Eso fija la dimensión de entrada de la `Linear` final en 128, sin importar qué tamaño tenía la imagen original.
# 
# Comparemos con el MLP. Ahí hicimos `Flatten → Linear(in_features, 512)`. El problema: `in_features` se calculó como `3 × 28 × 28 = 2352`. Si mañana la imagen viene en `64×64`, el flatten produce `3 × 64 × 64 = 12288` valores — y la primera `Linear` **no matchea la forma** y tira error.
# 
# **Consecuencia práctica en visión:**
# 
# - Por eso casi todos los pipelines de visión incluyen un `Resize` — para llevar todas las imágenes al mismo tamaño que espera el MLP (o la última capa densa de una CNN con flatten).
# - Las CNN *totalmente convolucionales* (con global pooling) pueden procesar imágenes de cualquier tamaño sin modificar pesos. Eso es útil en medicina: una radiografía puede venir en `1024×1024` y un parche de histopatología en `256×256`, y la misma red los procesa.
# - Es la misma razón por la que en la sección de ResNet resizeamos a `64×64`: no porque la red lo exija estrictamente (la ResNet también tiene global avg pool antes de la `fc`), sino porque `28×28` es tan chico que después de varios strides la resolución espacial se vuelve `0` y rompe.
# 
# Abajo lo vemos en acción: pasamos tensores de **tres tamaños distintos** por la misma CNN y por el mismo MLP.

# %%
# Forward pass con imágenes sintéticas de tres tamaños distintos
# (nos quedamos con batch=1 y usamos torch.randn — solo queremos verificar si la forward pasa)

sizes = [(3, 28, 28), (3, 64, 64), (3, 100, 100)]

print('CNN (con AdaptiveAvgPool):')
print('-' * 50)
cnn.eval()
for shape in sizes:
    x = torch.randn(1, *shape).to(device)
    try:
        with torch.no_grad():
            out = cnn(x)
        print(f'  input {tuple(x.shape)}  ->  output {tuple(out.shape)}  OK')
    except Exception as e:
        print(f'  input {tuple(x.shape)}  ->  ERROR: {type(e).__name__}: {str(e)[:80]}')

print('\nMLP (con Flatten + Linear de tamaño fijo):')
print('-' * 50)
mlp.eval()
for shape in sizes:
    x = torch.randn(1, *shape).to(device)
    try:
        with torch.no_grad():
            out = mlp(x)
        print(f'  input {tuple(x.shape)}  ->  output {tuple(out.shape)}  OK')
    except Exception as e:
        print(f'  input {tuple(x.shape)}  ->  ERROR: {type(e).__name__}: {str(e)[:80]}')

print('\n==> La CNN procesa cualquier tamaño; el MLP solo funciona con el tamaño exacto')
print('    para el que se definieron sus capas Linear.')

# %%
hist_cnn = train_model(cnn, train_loader, val_loader, epochs=3, lr=1e-3, name='CNN')

acc_cnn, preds_cnn, trues_cnn = evaluate(cnn, test_loader)
print(f'\n==> Accuracy en test (CNN): {acc_cnn:.4f}')

# %%
show_predictions(cnn, test_sub, n=10, title=f'CNN simple — test acc = {acc_cnn:.3f}')

# %%
# ### Matriz de confusión
# 
# Un número único como la *accuracy* resume todo el desempeño en un solo valor y **oculta dónde se equivoca el modelo**. La **matriz de confusión** soluciona eso: es una tabla de `n_clases × n_clases` donde
# 
# - Cada **fila** es la clase **real**.
# - Cada **columna** es la clase **predicha**.
# - La celda `(i, j)` cuenta cuántas muestras de clase `i` fueron clasificadas como `j`.
# 
# Interpretación rápida:
# 
# - La **diagonal** son los aciertos (predicho = real).
# - Todo lo que está **fuera de la diagonal** son errores, y su posición indica *con qué clase se confundió*.
# - En medicina esto es clave: no es lo mismo confundir dos tejidos sanos que confundir un tumor con tejido sano. La accuracy no distingue entre esos errores, la matriz de confusión sí.

# %%
plot_confusion(preds_cnn, trues_cnn, title='Matriz de confusión — CNN simple')

# %%
# ---
# ## 4.bis — La misma CNN, pero con más datos
# 
# Antes de cambiar de arquitectura, hacemos una pregunta clave: **¿cuánto mejora si simplemente entrenamos con más imágenes?**
# 
# Reentrenamos la CNN (mismo modelo, mismas epochs, misma tasa de aprendizaje) pero con un subset **5 veces más grande**. Esto muestra un principio general del deep learning: **los datos importan al menos tanto como la arquitectura**.
# 
# > Nota: esta celda tarda más que las anteriores porque procesa más imágenes por epoch. En CPU puede llevar un par de minutos.

# %%
# Subset más grande: 40.000 imágenes (vs 8.000 anteriores)
TRAIN_SUBSET_BIG = 40000

rng_big = np.random.RandomState(1)
train_idx_big = rng_big.choice(len(train_ds), TRAIN_SUBSET_BIG, replace=False)
train_sub_big = Subset(train_ds, train_idx_big)
train_loader_big = DataLoader(train_sub_big, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

print(f'Train subset anterior: {len(train_sub):>6} imágenes')
print(f'Train subset nuevo:    {len(train_sub_big):>6} imágenes (x{len(train_sub_big) // len(train_sub)})')

# %%
# Re-instanciamos la CNN desde cero (pesos aleatorios) para comparación limpia
cnn_big = SimpleCNN(in_channels=3, n_classes=n_classes)
hist_cnn_big = train_model(cnn_big, train_loader_big, val_loader, epochs=3, lr=1e-3, name='CNN-big')

acc_cnn_big, preds_cnn_big, trues_cnn_big = evaluate(cnn_big, test_loader)
print(f'\n==> Accuracy en test:')
print(f'    CNN (8k train):  {acc_cnn:.4f}')
print(f'    CNN (40k train): {acc_cnn_big:.4f}')
print(f'    Mejora absoluta: {(acc_cnn_big - acc_cnn) * 100:+.2f} puntos porcentuales')

# %%
show_predictions(cnn_big, test_sub, n=10, title=f'CNN + más datos — test acc = {acc_cnn_big:.3f}')

# %%
plot_confusion(preds_cnn_big, trues_cnn_big, title='Matriz de confusión — CNN con más datos')

# %%
# **Observación:** la arquitectura es exactamente la misma, el número de epochs también. Lo único que cambió es la cantidad de ejemplos. En deep learning este patrón es casi universal — **más datos de calidad suele superar a una arquitectura más lujosa con pocos datos**. Por eso en medicina el cuello de botella no suele ser elegir el modelo, sino curar y anotar el dataset.

# %%
# ---
# ## 5. ResNet: backbone preentrenado
# 
# ### ¿Qué es ResNet?
# 
# **ResNet** (He et al., 2015) fue un avance clave: permitió entrenar redes **muy profundas** (18, 34, 50, 101, 152 capas) sin que la señal del gradiente se degrade. Su idea central son las **skip connections** (o *residual connections*):
# 
# ```
#         x  ────────────────────────┐
#         │                          │
#    [Conv → BN → ReLU → Conv → BN]  │  (residual F(x))
#         │                          │
#         └────────────── (+) ───────┘
#                         │
#                    ReLU(x + F(x))
# ```
# 
# En vez de que cada bloque aprenda la salida directamente, aprende el **residuo** `F(x)` que se suma a la entrada. Eso deja pasar el gradiente sin atenuación y permite entrenar redes profundas de forma estable.
# 
# ### ¿Qué es un *backbone*?
# 
# Un **backbone** es una red preentrenada — típicamente sobre **ImageNet** (1.28M imágenes, 1000 clases) — cuyas capas convolucionales aprenden features visuales genéricas (bordes, texturas, partes de objetos). Esas features se **reutilizan** como punto de partida para tareas nuevas:
# 
# - Se reemplaza la última capa por una adaptada al número de clases de nuestro problema.
# - Se entrena sobre el dataset nuevo (*fine-tuning*), arrancando desde pesos ya informados.
# 
# Esto es **el patrón dominante en visión biomédica**: CheXNet (Stanford) es una DenseNet-121 preentrenada en ImageNet y ajustada para detectar neumonía en Rx de tórax. Modelos de retinopatía diabética, de lesiones cutáneas, de histopatología — casi todos siguen el mismo esquema.
# 
# > **Punto clave:** no es obvio que features aprendidas sobre perros, autos y objetos cotidianos sirvan para tejidos o radiografías. Pero en la práctica sí sirven — son *representaciones visuales de bajo y medio nivel* que transfieren bien entre dominios.

# %%
# ### 5.a — ResNet-18 sin preentrenar (desde cero)
# 
# Antes de usar el backbone preentrenado, es instructivo entrenar la **misma arquitectura** pero con los pesos inicializados aleatoriamente. Esto aísla el efecto del preentrenamiento: cualquier diferencia de performance contra la versión preentrenada viene puramente de haber visto ImageNet antes.
# 
# Pasos:
# 1. Redimensionar las imágenes a 64×64 (ResNet está pensada para inputs más grandes; 28×28 es demasiado chico para su stride inicial).
# 2. Instanciar `resnet18(weights=None)` — arquitectura sin pesos.
# 3. Reemplazar la capa final `fc` por una adaptada a nuestras `n_classes`.
# 4. Entrenar desde cero.

# %%
# Transformación y dataloaders para ResNet (resize a 64x64 + normalización ImageNet)
# Usamos las mismas estadísticas que la versión preentrenada para que la única diferencia sean los pesos iniciales
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

transform_resnet = T.Compose([
    T.Resize(64),
    T.ToTensor(),
    T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

train_ds_r = DataClass(split='train', transform=transform_resnet, download=True, size=28)
val_ds_r   = DataClass(split='val',   transform=transform_resnet, download=True, size=28)
test_ds_r  = DataClass(split='test',  transform=transform_resnet, download=True, size=28)

train_sub_r = Subset(train_ds_r, train_idx)
test_sub_r  = Subset(test_ds_r,  test_idx)

train_loader_r = DataLoader(train_sub_r, batch_size=BATCH_SIZE, shuffle=True)
val_loader_r   = DataLoader(val_ds_r,    batch_size=BATCH_SIZE, shuffle=False)
test_loader_r  = DataLoader(test_sub_r,  batch_size=BATCH_SIZE, shuffle=False)

# %%
# ResNet-18 SIN pesos preentrenados
resnet_scratch = resnet18(weights=None)
resnet_scratch.fc = nn.Linear(resnet_scratch.fc.in_features, n_classes)

print(f'Parámetros entrenables: {count_params(resnet_scratch):,}')
print(f'Pesos iniciales: aleatorios (NO preentrenados)')

# %%
# Entrenamiento desde cero — lr estándar (no hace falta ser conservador porque no hay pesos buenos que romper)
hist_resnet_scratch = train_model(resnet_scratch, train_loader_r, val_loader_r, epochs=3, lr=1e-3, name='ResNet-scratch')

acc_resnet_scratch, preds_resnet_scratch, trues_resnet_scratch = evaluate(resnet_scratch, test_loader_r)
print(f'\n==> Accuracy en test (ResNet-18 desde cero): {acc_resnet_scratch:.4f}')

# %%
# **Observación preliminar:** la ResNet desde cero suele quedar en un rango parecido a la CNN simple — a veces incluso un poco peor — porque es una red mucho más grande y 3 epochs sobre 8k imágenes no alcanzan para que aproveche su capacidad. El problema típico de las redes profundas es que *necesitan muchos datos* para no sobreajustar ni quedarse subentrenadas.

# %%
# ### 5.b — ResNet-18 preentrenada (*transfer learning*)
# 
# Ahora repetimos el experimento, pero arrancando desde pesos **preentrenados en ImageNet** en lugar de pesos aleatorios.
# 
# Pasos:
# 1. Cargar `resnet18` con pesos de ImageNet (`ResNet18_Weights.IMAGENET1K_V1`).
# 2. Reemplazar la capa final `fc` (originalmente 1000 clases de ImageNet) por una con `n_classes` salidas.
# 3. Entrenar (*fine-tuning* de toda la red) con una tasa de aprendizaje más baja.
# 
# Lo interesante va a ser comparar esta accuracy con la de la misma arquitectura sin preentrenamiento.

# %%
# Cargamos ResNet-18 preentrenada y reemplazamos la capa final
resnet = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
resnet.fc = nn.Linear(resnet.fc.in_features, n_classes)

print(f'ResNet-18 — última capa: {resnet.fc}')
print(f'Parámetros entrenables: {count_params(resnet):,}')
print(f'   (MLP: {count_params(mlp):,}  |  CNN: {count_params(cnn):,})')

# %%
# Fine-tuning: lr más bajo porque arrancamos de pesos ya informados (no queremos romperlos)
hist_resnet = train_model(resnet, train_loader_r, val_loader_r, epochs=3, lr=3e-4, name='ResNet-pretrained')

acc_resnet, preds_resnet, trues_resnet = evaluate(resnet, test_loader_r)
print(f'\n==> Accuracy en test:')
print(f'    ResNet-18 desde cero:    {acc_resnet_scratch:.4f}')
print(f'    ResNet-18 preentrenada:  {acc_resnet:.4f}')
print(f'    Ganancia del transfer:   {(acc_resnet - acc_resnet_scratch) * 100:+.2f} puntos porcentuales')

# %%
# Ejemplos de predicción — usamos el dataset con resize
def show_predictions_resnet(model, dataset, n=10, title=''):
    """Versión para ResNet: la imagen está normalizada con stats de ImageNet, la desnormalizamos distinto."""
    model.eval()
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std  = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    cols = 5
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.2, rows * 2.4))
    axes = axes.flatten()
    idxs = np.random.RandomState(42).choice(len(dataset), n, replace=False)
    for ax, idx in zip(axes, idxs):
        img, label = dataset[idx]
        true = int(np.asarray(label).flatten()[0])
        with torch.no_grad():
            probs = F.softmax(model(img.unsqueeze(0).to(device)), dim=1)[0]
            pred = int(probs.argmax().item()); conf = float(probs[pred].item())
        img_show = (img * std + mean).permute(1, 2, 0).clamp(0, 1).cpu().numpy()
        ax.imshow(img_show); ax.axis('off')
        ax.set_title(f'Real: {class_names[true][:10]}\nPred: {class_names[pred][:10]} ({conf:.0%})',
                     fontsize=8, color=('green' if pred == true else 'red'))
    for ax in axes[n:]:
        ax.axis('off')
    plt.suptitle(title, fontsize=12); plt.tight_layout(); plt.show()

show_predictions_resnet(resnet, test_sub_r, n=10, title=f'ResNet-18 — test acc = {acc_resnet:.3f}')

# %%
# Y la matriz de confusión, para ver si los errores se distribuyen distinto que con los modelos anteriores:

# %%
plot_confusion(preds_resnet, trues_resnet, title='Matriz de confusión — ResNet-18')

# %%
# ---
# ## 6. Benchmarking
# 
# Comparamos todos nuestros modelos entre sí y contra el **benchmark oficial de PathMNIST** publicado en la página de MedMNIST ([medmnist.com](https://medmnist.com/)).
# 
# Los números oficiales fueron obtenidos con:
# - Dataset completo (~90k train).
# - Entrenamiento más largo (100 epochs).
# - Selección de modelo por mejor AUC en validación.
# 
# Los nuestros corren sobre subsets y solo 3 epochs — así que es esperable que estén más abajo. Lo interesante es **el patrón relativo** entre variantes: qué aporta cambiar la arquitectura, qué aporta agregar datos, y qué aporta el preentrenamiento.

# %%
# Nuestros resultados
our_results = {
    'MLP':                    acc_mlp,
    'CNN (8k train)':         acc_cnn,
    'CNN (40k train)':        acc_cnn_big,
    'ResNet-18 desde cero':   acc_resnet_scratch,
    'ResNet-18 preentrenada': acc_resnet,
}

# Benchmark oficial de MedMNIST para PathMNIST (28x28), accuracy en test
# Fuente: https://medmnist.com/ (MedMNIST v2, Yang et al., 2023)
official_results = {
    'auto-sklearn':         0.716,
    'Google AutoML Vision': 0.728,
    'AutoKeras':            0.834,
    'ResNet-18 (28)':       0.907,
    'ResNet-18 (224)':      0.909,
    'ResNet-50 (28)':       0.911,
    'ResNet-50 (224)':      0.911,
}

fig, ax = plt.subplots(figsize=(10, 7))
all_items = [(k, v, 'ours')      for k, v in our_results.items()] + \
            [(k, v, 'benchmark') for k, v in official_results.items()]
all_items.sort(key=lambda t: t[1])

names  = [t[0] for t in all_items]
values = [t[1] for t in all_items]
colors = ['#E74C3C' if t[2] == 'ours' else '#3498DB' for t in all_items]

bars = ax.barh(names, values, color=colors, edgecolor='black')
for bar, v in zip(bars, values):
    ax.text(v + 0.005, bar.get_y() + bar.get_height()/2, f'{v:.3f}',
            va='center', fontsize=9)
ax.set_xlabel('Accuracy en test')
ax.set_xlim(0.4, 1.0)
ax.set_title('Benchmark — PathMNIST (28x28)\n' +
             'Rojo: nuestros modelos   Azul: benchmark oficial MedMNIST (90k train, 100 epochs)',
             fontsize=11)
ax.grid(True, axis='x', alpha=0.3)
plt.tight_layout()
plt.show()

# %%
# Curvas de entrenamiento de los cinco modelos
fig, axes = plt.subplots(1, 2, figsize=(13, 4))

histories = [
    ('MLP',                    hist_mlp),
    ('CNN (8k)',               hist_cnn),
    ('CNN (40k)',              hist_cnn_big),
    ('ResNet-18 scratch',      hist_resnet_scratch),
    ('ResNet-18 pretrained',   hist_resnet),
]

for name, hist in histories:
    axes[0].plot(hist['train_loss'], 'o-', label=name)
    axes[1].plot(hist['val_acc'],    'o-', label=name)

axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('Train loss')
axes[0].set_title('Evolución del loss'); axes[0].legend(fontsize=8); axes[0].grid(alpha=0.3)

axes[1].set_xlabel('Epoch'); axes[1].set_ylabel('Val accuracy')
axes[1].set_title('Evolución de val accuracy'); axes[1].legend(fontsize=8); axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.show()

# %%
# ---
# ## Conclusiones
# 
# Qué mostró la práctica:
# 
# 1. **MLP vs CNN:** a igualdad de entrenamiento, la CNN supera al MLP con **menos parámetros**. La diferencia es consecuencia directa de respetar la estructura espacial (localidad + equivarianza traslacional).
# 2. **Datos > arquitectura (a veces).** La misma CNN entrenada con 5× más imágenes mejora de forma clara, sin tocar ni un hiperparámetro. En medicina, dedicar esfuerzo a curar más datos suele rendir más que perseguir la arquitectura del momento.
# 3. **Arquitectura sola no alcanza.** La ResNet-18 *sin preentrenar* tiene millones de parámetros pero, con poco datos y pocas epochs, no logra superar claramente a una CNN simple. Una arquitectura potente necesita un entrenamiento acorde.
# 4. **Transfer learning gana casi gratis.** La misma ResNet-18, inicializada con pesos de ImageNet, mejora fuerte respecto a la versión desde cero con el mismo presupuesto de cómputo. Esto es el principio detrás de CheXNet y casi todos los modelos clínicos de visión: *reusar backbones entrenados en datos naturales*.
# 5. **Escalabilidad al benchmark oficial.** Con 90k imágenes y 100 epochs, los modelos de MedMNIST llegan a ~91%. Nuestros números son más bajos, pero el orden entre modelos es el esperado.
# 
# ### Ideas para seguir experimentando
# 
# - Probar **otro dataset** de MedMNIST (`pneumoniamnist`, `dermamnist`, `chestmnist`, `breastmnist`). Cambiar una línea (`data_flag`) y todo sigue funcionando.
# - **Congelar el backbone** de la ResNet preentrenada (`requires_grad = False` en todo salvo `fc`) y comparar contra fine-tuning completo.
# - Aumentar a **size=64 o 224** en MedMNIST y ver si la ResNet mejora (son imágenes más grandes → más detalle).
# - Probar **ResNet-50**, **DenseNet-121** o **EfficientNet-B0** — todas disponibles en `torchvision.models` con pesos preentrenados.
# - Entrenar más epochs sobre el **dataset completo** y reproducir el benchmark oficial.

# %%
# # ──────────────────────────────────────────────────────────
# # Parte 2 — Práctica: replicar el pipeline en PneumoniaMNIST
# # ──────────────────────────────────────────────────────────
# 
# Ahora les toca a ustedes. Vamos a usar otro dataset de MedMNIST: **PneumoniaMNIST** — radiografías de tórax pediátricas clasificadas como *normal* o *neumonía*.
# 
# **Diferencias clave con PathMNIST que hay que adaptar:**
# 
# | Aspecto             | PathMNIST        | PneumoniaMNIST          |
# |---------------------|------------------|-------------------------|
# | Tarea               | 9-clase          | **binaria**             |
# | Canales             | 3 (RGB)          | **1 (grayscale)**       |
# | Dominio             | Histopatología   | Radiografía de tórax    |
# | # imágenes (train)  | ~90k             | ~4.7k                   |
# 
# Eso obliga a:
# - Cambiar `n_classes = 2` y `in_channels = 1`.
# - Revisar toda la cadena de shapes: `(3, 28, 28)` ahora es `(1, 28, 28)`.
# - Adaptar la ResNet preentrenada (espera 3 canales) — hint: replicar el canal grayscale tres veces con un transform.
# 
# **Al final de la parte 2 van a entregar:**
# 1. Los tres modelos adaptados y entrenados (MLP, CNN modificada, ResNet preentrenada).
# 2. Un mini-experimento de hiperparámetros (ver sección 2.5).
# 3. Una comparación con el benchmark oficial de PneumoniaMNIST.
# 4. Una breve discusión sobre la matriz de confusión (sección 2.7).

# %%
# ## 2.1 — Carga y exploración del nuevo dataset
# 
# **Ejercicio 2.1.a:** cambiá el `data_flag` y cargá el `INFO`. Imprimí los metadatos relevantes (tarea, canales, número de clases, nombres de las clases).

# %%
data_flag_p = 'pneumoniamnist'
info_p = INFO[data_flag_p]

print(f'Dataset:     {info_p["python_class"]}')
print(f'Descripción: {info_p["description"][:200]}...')
print(f'Tarea:       {info_p["task"]}')
print(f'Canales:     {info_p["n_channels"]}')
print(f'# clases:    {len(info_p["label"])}')
print(f'Clases:')
for k, v in info_p['label'].items():
    print(f'   {k}: {v}')

# %%
# **Ejercicio 2.1.b:** cargá los tres splits (train, val, test) con un transform apropiado. **Ojo:** ahora las imágenes son grayscale — el `Normalize` espera un solo valor de media y desvío, no tres.

# %%
transform_p = T.Compose([
    T.ToTensor(),
    T.Normalize([0.5], [0.5])
])

DataClass_p = getattr(medmnist, info_p['python_class'])

train_ds_p = DataClass_p(split='train', transform=transform_p, download=True, size=28)
val_ds_p   = DataClass_p(split='val',   transform=transform_p, download=True, size=28)
test_ds_p  = DataClass_p(split='test',  transform=transform_p, download=True, size=28)

n_classes_p = len(info_p['label'])
class_names_p = [info_p['label'][str(i)] for i in range(n_classes_p)]
img, label = train_ds_p[0]
print(f'Shape: {tuple(img.shape)}  ->  ¿qué cambió vs PathMNIST?')
print(f'Train: {len(train_ds_p)} | Val: {len(val_ds_p)} | Test: {len(test_ds_p)}')

# %%
# **Ejercicio 2.1.c:** visualizá algunos ejemplos por clase y la distribución del train set.
# 
# Podés reusar las funciones que ya vimos; solo cuidado con `imshow` — una imagen grayscale tiene shape `(H, W)` (sin dimensión de canal) o `(H, W, 1)` — en ambos casos matplotlib espera `cmap='gray'` para mostrarla bien.

# %%
fig, axes = plt.subplots(1, 10, figsize=(15, 2))
for i in range(10):
    img, label = train_ds_p[i]
    img = img.squeeze(0) * 0.5 + 0.5 # desnormalizar
    axes[i].imshow(img, cmap='gray')
    axes[i].set_title(class_names_p[label.item()])
    axes[i].axis('off')
plt.show()

# %%
labels = [label.item() for _, label in train_ds_p]
unique, counts = np.unique(labels, return_counts=True)
plt.bar([class_names_p[i] for i in unique], counts, color=['lightblue', 'salmon'])
plt.title('Distribución de clases en Train')
plt.ylabel('Cantidad')
plt.show()
print('Conclusión: no están balanceadas. Hay mucha más neumonía que normal.')

# %%
# ## 2.2 — Subset y DataLoaders
# 
# PneumoniaMNIST es más chico que PathMNIST (~4.7k train), así que pueden usar el dataset completo sin subsetear. Armá los `DataLoader` de train, val y test.

# %%
BATCH_SIZE_P = 64

train_loader_p = DataLoader(train_ds_p, batch_size=BATCH_SIZE_P, shuffle=True)
val_loader_p   = DataLoader(val_ds_p,   batch_size=BATCH_SIZE_P, shuffle=False)
test_loader_p  = DataLoader(test_ds_p,  batch_size=BATCH_SIZE_P, shuffle=False)

# %%
# ## 2.3 — Adaptar el MLP
# 
# **Ejercicio 2.3:** diseñá un MLP para PneumoniaMNIST. Mínimos cambios esperables: `in_features` nuevo (¿cuánto es?) y `n_classes=2`. Podés usar la misma cantidad de capas que el de referencia o probar algo distinto.
# 
# Después entrenalo por unas pocas epochs y reportá la accuracy en test.

# %%
class MLP_P(nn.Module):
    def __init__(self, in_shape=(1, 28, 28), n_classes=2):
        super().__init__()
        # 1*28*28 = 784
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, n_classes)
        )

    def forward(self, x):
        return self.net(x)

mlp_p = MLP_P(in_shape=(1, 28, 28), n_classes=2).to(device)
hist_mlp_p = train_model(mlp_p, train_loader_p, val_loader_p, epochs=5, lr=1e-3, name='MLP-P')
acc_mlp_p, preds_mlp_p, trues_mlp_p = evaluate(mlp_p, test_loader_p)
print(f'==> Accuracy en test (MLP-P): {acc_mlp_p:.4f}')

# %%
# ## 2.4 — CNN modificada (requisitos obligatorios)
# 
# Diseñá y entrená una CNN adaptada a PneumoniaMNIST. **No copies la CNN de la Parte 1 — queremos ver una arquitectura distinta.** Tu red tiene que cumplir, como mínimo, los siguientes puntos:
# 
# 1. **`in_channels=1`** y **`n_classes=2`** (adaptación obligatoria al nuevo dataset).
# 2. **Al menos 4 capas convolucionales** (la de clase tenía 3).
# 3. **`BatchNorm2d` después de cada `Conv2d`** — estabiliza el entrenamiento, especialmente con redes más profundas.
# 4. **Reemplazar el bloque final `AdaptiveAvgPool2d(1) → Linear`** por `Flatten` + **al menos una capa `Linear` oculta** antes de la salida (estilo "cabeza densa"). Cuidado con el tamaño de entrada al `Flatten` — depende de los strides/poolings que pongas.
# 5. Agregá **al menos un `Dropout`** en algún lugar de la red para regularizar.
# 
# **Tip:** una forma limpia es usar `nn.Sequential` con bloques `Conv2d → BatchNorm2d → ReLU → MaxPool2d` repetidos, y recién al final poner el clasificador denso.

# %%
class MyCNN(nn.Module):
    def __init__(self, in_channels=1, n_classes=2):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(16, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 1 * 1, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, n_classes)
        )

    def forward(self, x):
        return self.classifier(self.features(x))

my_cnn = MyCNN(in_channels=1, n_classes=2).to(device)
print(my_cnn)
def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f'\nParámetros: {count_params(my_cnn):,}')

# %%
hist_my_cnn = train_model(my_cnn, train_loader_p, val_loader_p, epochs=5, lr=1e-3, name='MyCNN')
acc_my_cnn, preds_my_cnn, trues_my_cnn = evaluate(my_cnn, test_loader_p)
print(f'\n==> Accuracy en test (MyCNN): {acc_my_cnn:.4f}')

# %%
# ## 2.5 — Experimento de hiperparámetros
# 
# Fijá la arquitectura `MyCNN` (la que acabás de diseñar) y **barré uno o más hiperparámetros**. El mínimo es:
# 
# - **Learning rate**: probá al menos **3 valores** (sugerencia: `1e-2`, `1e-3`, `1e-4`).
# 
# Opcionalmente, barré también uno de estos:
# - **Batch size**: `32`, `64`, `128`.
# - **Número de epochs**: `3`, `5`, `10`.
# - **Optimizer**: `Adam` vs `SGD(momentum=0.9)`.
# - **Dropout rate**: `0.0`, `0.2`, `0.5`.
# 
# **Entregable:** una tabla o un gráfico con la accuracy de validación para cada configuración, y **una línea de conclusión** diciendo cuál combinación ganó y por qué pensás que fue así.
# 
# > **Importante:** para cada configuración, **reinstanciá el modelo desde cero** (si reentrenás el mismo objeto, los pesos ya están ajustados y la comparación no es limpia).

# %%
import pandas as pd
results = []
for lr in [1e-2, 1e-3, 1e-4]:
    model = MyCNN(in_channels=1, n_classes=2).to(device)
    hist = train_model(model, train_loader_p, val_loader_p, epochs=5, lr=lr, name=f'MyCNN (lr={lr})')
    acc, _, _ = evaluate(model, test_loader_p)
    results.append({'lr': lr, 'val_acc_final': hist['val_acc'][-1], 'test_acc': acc})
print(pd.DataFrame(results))

# %%
# **Conclusión del experimento (escribí acá):**
# 
# _Reemplazá este texto con tu análisis. ¿Qué configuración ganó? ¿Te sorprendió algún resultado? ¿Vieron evidencia de underfitting (loss alto, acc baja) o overfitting (gap grande entre train y val)?_

# %%
# ## 2.6 — Adaptar la ResNet-18 preentrenada
# 
# La ResNet-18 preentrenada espera imágenes **RGB** (3 canales), pero PneumoniaMNIST es grayscale (1 canal). Tienen dos opciones:
# 
# 1. **Replicar el canal grayscale tres veces** (con un transform tipo `Grayscale(num_output_channels=3)` o directamente `lambda x: x.repeat(3, 1, 1)`). **Recomendada** — deja intacta la primera capa preentrenada.
# 2. Reemplazar `resnet.conv1` por un `Conv2d(1, 64, ...)` nuevo. Pierden el preentrenamiento de esa capa.
# 
# Usen la opción **1**. Además: resize a 64×64 y normalización con estadísticas de ImageNet, igual que en la Parte 1.

# %%
# Asumo que IMAGENET_MEAN e IMAGENET_STD existen, sino usamos los valores default de ImageNet
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

transform_resnet_p = T.Compose([
    T.Resize(64),
    T.ToTensor(),
    T.Lambda(lambda x: x.repeat(3, 1, 1)),
    T.Normalize(IMAGENET_MEAN, IMAGENET_STD)
])

train_ds_pr = DataClass_p(split='train', transform=transform_resnet_p, download=True, size=28)
val_ds_pr   = DataClass_p(split='val',   transform=transform_resnet_p, download=True, size=28)
test_ds_pr  = DataClass_p(split='test',  transform=transform_resnet_p, download=True, size=28)

train_loader_pr = DataLoader(train_ds_pr, batch_size=BATCH_SIZE_P, shuffle=True)
val_loader_pr   = DataLoader(val_ds_pr,   batch_size=BATCH_SIZE_P, shuffle=False)
test_loader_pr  = DataLoader(test_ds_pr,  batch_size=BATCH_SIZE_P, shuffle=False)

# %%
resnet_p = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
resnet_p.fc = nn.Linear(resnet_p.fc.in_features, n_classes_p)
resnet_p = resnet_p.to(device)

hist_resnet_p = train_model(resnet_p, train_loader_pr, val_loader_pr, epochs=3, lr=3e-4, name='ResNet-P')
acc_resnet_p, preds_resnet_p, trues_resnet_p = evaluate(resnet_p, test_loader_pr)
print(f'\n==> Accuracy en test (ResNet-P): {acc_resnet_p:.4f}')

# %%
# ## 2.7 — Matriz de confusión y análisis clínico
# 
# Una vez entrenado el mejor modelo, graficá la matriz de confusión sobre el test set y **respondé por escrito** las siguientes preguntas:
# 
# **Q1.** ¿Cuántos **falsos negativos** hay? (predicción *normal* cuando en realidad es *neumonía*). En términos clínicos, ¿es un error más grave que el inverso? Justificá.
# 
# **Q2.** El dataset está desbalanceado (hay más casos de neumonía que de normal). ¿La **accuracy** es una métrica suficiente en este caso? ¿Qué métrica(s) adicional(es) recomendarías mirar? (pista: sensibilidad/recall, especificidad, F1).
# 
# **Q3.** Si tuvieran que decidir el *threshold* de decisión (el valor de probabilidad por encima del cual predecimos "neumonía"), ¿lo moverían del 0.5 por defecto? ¿En qué dirección y por qué?

# %%
# Elegimos MyCNN (o la mejor que dio)
from sklearn.metrics import confusion_matrix
import seaborn as sns
cm_p = confusion_matrix(trues_my_cnn, preds_my_cnn)
plt.figure(figsize=(5,4))
sns.heatmap(cm_p, annot=True, fmt='d', cmap='Blues', xticklabels=class_names_p, yticklabels=class_names_p)
plt.xlabel('Predicción')
plt.ylabel('Real')
plt.title('Matriz de Confusión - MyCNN en PneumoniaMNIST')
plt.show()

# %%
# **Respuestas a Q1, Q2, Q3 (escribí acá):**
# 
# _Reemplazá este texto con tu análisis clínico._

# %%
# ## 2.8 — Comparación con el benchmark oficial de PneumoniaMNIST
# 
# Graficá una comparación entre tus tres modelos (MLP, MyCNN, ResNet preentrenada) y los números oficiales del paper de MedMNIST para **PneumoniaMNIST** (accuracy en test, tamaño 28×28).
# 
# Números de referencia (fuente: [medmnist.com](https://medmnist.com/), MedMNIST v2, Yang et al. 2023 — valores aproximados, verificá la tabla oficial):
# 
# | Método                  | Accuracy |
# |-------------------------|----------|
# | auto-sklearn            | 0.855    |
# | AutoKeras               | 0.878    |
# | Google AutoML Vision    | 0.889    |
# | ResNet-18 (28)          | 0.854    |
# | ResNet-18 (224)         | 0.864    |
# | ResNet-50 (28)          | 0.854    |
# | ResNet-50 (224)         | 0.884    |

# %%
# Según medmnist, acc en test para PneumoniaMNIST puede andar por ~0.85 o más.
bench_acc = 0.854 # Aproximado de paper de MedMNISTv2 para ResNet18

models = ['MLP-P', 'MyCNN', 'ResNet-P', 'Benchmark']
accs = [acc_mlp_p, acc_my_cnn, acc_resnet_p, bench_acc]

plt.figure(figsize=(8, 4))
sns.barplot(x=accs, y=models, palette='viridis')
plt.xlim(0, 1.0)
plt.title('Comparación de Accuracy en Test - PneumoniaMNIST')
for i, acc in enumerate(accs):
    plt.text(acc + 0.01, i, f'{acc:.3f}', va='center')
plt.show()

# %%
# # ──────────────────────────────────────────────────────────
# # Parte 3 — Desafío opcional: dataset 3D
# # ──────────────────────────────────────────────────────────
# 
# **Totalmente opcional.** Para quienes quieran ir un paso más allá: MedMNIST también incluye datasets **3D** (volúmenes, no imágenes planas). Un volumen es un tensor `(D, H, W)` — por ejemplo un recorte de CT de 28×28×28 voxeles.
# 
# **Opciones disponibles:**
# 
# | Dataset           | Tarea                                        | Clases | Tamaño aprox. |
# |-------------------|----------------------------------------------|--------|---------------|
# | OrganMNIST3D      | Clasificar órgano a partir de corte de CT    | 11     | ~1.7k         |
# | NoduleMNIST3D     | Nódulo pulmonar maligno sí/no                | 2      | ~1.6k         |
# | FractureMNIST3D   | Tipo de fractura vertebral                   | 3      | ~1.4k         |
# | AdrenalMNIST3D    | Masa adrenal benigna vs maligna              | 2      | ~1.6k         |
# | VesselMNIST3D     | Vaso cerebral aneurismático sí/no            | 2      | ~1.9k         |
# | SynapseMNIST3D    | Sinapsis excitatoria vs inhibitoria          | 2      | ~1.7k         |
# 
# **Qué cambia respecto al caso 2D:**
# 
# 1. El input es `(B, 1, D, H, W)` en vez de `(B, C, H, W)`.
# 2. Hay que usar **`nn.Conv3d`**, **`nn.MaxPool3d`**, **`nn.BatchNorm3d`** en vez de las versiones 2D.
# 3. El costo computacional sube fuerte — más parámetros y más operaciones por sample. **Usen GPU**.
# 4. Las ResNets de `torchvision` son 2D. Para 3D pueden: (a) implementar una CNN 3D desde cero (recomendado para este desafío), o (b) explorar `torchvision.models.video.r3d_18` (pensada para video pero sirve como *backbone* 3D), o (c) usar MONAI — framework específico de DL médico: [monai.io](https://monai.io/).
# 
# **Entregable sugerido:**
# - Elegir un dataset 3D de la tabla.
# - Entrenar una CNN 3D básica (al menos 2 bloques `Conv3d → BN3d → ReLU → MaxPool3d` + cabeza densa).
# - Reportar accuracy en test.
# - Comparar con el benchmark oficial de MedMNIST para ese dataset.

# %%
# TODO (opcional): cargar un dataset 3D
# Ejemplo con OrganMNIST3D:
# data_flag_3d = 'organmnist3d'
# info_3d = INFO[data_flag_3d]
# DataClass_3d = getattr(medmnist, info_3d['python_class'])
#
# IMPORTANTE: los datasets 3D no aceptan el argumento size=28 en todas las versiones. Revisá la doc:
# https://medmnist.com/  ->  Documentation / Quickstart
#
# Transform para 3D: a diferencia del 2D, no hay T.ToTensor() directo para volúmenes.
# MedMNIST 3D devuelve arrays numpy shape (D, H, W). Hay que convertirlos a tensor y agregar canal:
# transform_3d = lambda v: torch.from_numpy(v).float().unsqueeze(0) / 255.0  # (1, D, H, W)

# train_ds_3d = DataClass_3d(split='train', transform=transform_3d, download=True)
# val_ds_3d   = DataClass_3d(split='val',   transform=transform_3d, download=True)
# test_ds_3d  = DataClass_3d(split='test',  transform=transform_3d, download=True)

pass  # opcional, reemplazá si querés arrancar con el desafío

# %%
# TODO (opcional): CNN 3D simple
# class Simple3DCNN(nn.Module):
#     def __init__(self, in_channels=1, n_classes=11):
#         super().__init__()
#         self.features = nn.Sequential(
#             nn.Conv3d(in_channels, 16, kernel_size=3, padding=1),
#             nn.BatchNorm3d(16),
#             nn.ReLU(),
#             nn.MaxPool3d(2),
#             nn.Conv3d(16, 32, kernel_size=3, padding=1),
#             nn.BatchNorm3d(32),
#             nn.ReLU(),
#             nn.MaxPool3d(2),
#             nn.AdaptiveAvgPool3d(1),
#         )
#         self.classifier = nn.Sequential(
#             nn.Flatten(),
#             nn.Linear(32, n_classes),
#         )
#     def forward(self, x):
#         return self.classifier(self.features(x))

# Cuidado: las labels 3D también vienen en shape (1,) igual que 2D — to_target() les sirve.
# Pero el train_model() que definimos va a funcionar sin cambios.

pass  # opcional

# %%
# ---
# ## Entrega y cierre
# 
# Lo que tienen que entregar de esta notebook:
# 
# 1. **Parte 2 completa** — todas las celdas marcadas `# TODO` resueltas y corridas, con sus outputs visibles.
# 2. **Respuestas en prosa** a las preguntas de las secciones 2.5 y 2.7.
# 3. **Gráfico de comparación** con el benchmark oficial (sección 2.8).
# 4. **Parte 3 (opcional)** — si lo hacen, nota de honor.
# 
# **Recursos útiles:**
# - Documentación de MedMNIST: [medmnist.com](https://medmnist.com/)
# - Paper de MedMNIST v2 (Yang et al., 2023) con benchmarks oficiales: [arxiv.org/abs/2110.14795](https://arxiv.org/abs/2110.14795)
# - Torchvision models (arquitecturas preentrenadas): [pytorch.org/vision/stable/models.html](https://pytorch.org/vision/stable/models.html)
# - MONAI (DL médico): [monai.io](https://monai.io/)

