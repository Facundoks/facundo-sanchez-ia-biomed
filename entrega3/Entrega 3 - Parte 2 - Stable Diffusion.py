# -*- coding: utf-8 -*-
# Generated from: Entrega 3 - Parte 2 - Stable Diffusion.ipynb

# %%
# # **Materia: IA para Ingeniería Biomédica**
# ## Clase 9: Stable Diffusion 🎨
# 
# **Profesores:** Paulo Veiga · Marco Sanchez Sorondo
# 
# *...usando `🧨diffusers`*
# 
# Stable Diffusion es un modelo de difusión latente de texto a imagen creado por investigadores e ingenieros de [CompVis](https://github.com/CompVis), [Stability AI](https://stability.ai/) y [LAION](https://laion.ai/). Está entrenado con imágenes de 512x512 de un subconjunto de la base de datos [LAION-5B](https://laion.ai/blog/laion-5b/). Este modelo utiliza un codificador de texto CLIP ViT-L/14 congelado para condicionar el modelo en las indicaciones de texto. Con su UNet de 860M y su codificador de texto de 123M, el modelo es relativamente ligero y puede ejecutarse en muchas GPUs de consumo.
# Consulte la [tarjeta del modelo](https://huggingface.co/CompVis/stable-diffusion) para más información.
# 
# Este cuaderno de Colab muestra cómo usar Stable Diffusion con la biblioteca 🤗 Hugging Face [🧨 Diffusers](https://github.com/huggingface/diffusers).
# 
# ¡Comencemos!

# %%
# ## 1. Cómo usar `StableDiffusionPipeline`
# 
# Antes de profundizar en los aspectos teóricos de cómo funciona Stable Diffusion,
# ¡probémoslo un poco! 🤗
# 
# En esta sección, mostramos cómo puede ejecutar inferencia de texto a imagen en solo unas pocas líneas de código.

# %%
# ### Configuración
# 
# Primero, asegúrese de estar usando un entorno de ejecución con GPU para ejecutar este cuaderno, de modo que la inferencia sea mucho más rápida. Si el siguiente comando falla, use el menú `Runtime` arriba y seleccione `Change runtime type`.

# %%
import torch

# %%
if torch.cuda.is_available():
    # Shows the nVidia GPUs, if this system has any
    !nvidia-smi

# %%
# A continuación, debe instalar `diffusers` así como `scipy`, `ftfy` y `transformers`. `accelerate` se utiliza para lograr una carga mucho más rápida.

# %%
!pip install diffusers==0.11.1
!pip install transformers scipy ftfy accelerate

# %%
# ### Pipeline de Stable Diffusion
# 
# `StableDiffusionPipeline` es un pipeline de inferencia de extremo a extremo que puede usar para generar imágenes a partir de texto con solo unas pocas líneas de código.
# 
# Primero, cargamos los pesos preentrenados de todos los componentes del modelo. En este cuaderno usamos Stable Diffusion versión 1.4 ([CompVis/stable-diffusion-v1-4](https://huggingface.co/CompVis/stable-diffusion-v1-4)), pero hay otras variantes que quizás desee probar:
# * [runwayml/stable-diffusion-v1-5](https://huggingface.co/runwayml/stable-diffusion-v1-5)
# * [stabilityai/stable-diffusion-2-1-base](https://huggingface.co/stabilityai/stable-diffusion-2-1-base)
# * [stabilityai/stable-diffusion-2-1](https://huggingface.co/stabilityai/stable-diffusion-2-1). Esta versión puede producir imágenes con una resolución de 768x768, mientras que las otras funcionan a 512x512.
# 
# Además del id del modelo [CompVis/stable-diffusion-v1-4](https://huggingface.co/CompVis/stable-diffusion-v1-4), también estamos pasando una `revision` específica y `torch_dtype` al método `from_pretrained`.
# 
# Queremos asegurarnos de que cada Google Colab gratuito pueda ejecutar Stable Diffusion, por lo que estamos cargando los pesos de la rama de media precisión [`fp16`](https://huggingface.co/CompVis/stable-diffusion-v1-4/tree/fp16) y también le decimos a `diffusers` que espere los pesos en precisión float16 pasando `torch_dtype=torch.float16`.
# 
# Si desea asegurar la máxima precisión posible, asegúrese de eliminar `torch_dtype=torch.float16` a costa de un mayor uso de memoria.

# %%
# This is added to get around some issues of Torch not loading models correctly (test on Mac OS X and Kubuntu Linux)
!pip install --upgrade huggingface-hub==0.26.2 transformers==4.46.1 tokenizers==0.20.1 diffusers==0.31.0

# %%
from diffusers import StableDiffusionPipeline

pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", torch_dtype=torch.float16)

# %%
# A continuación, movamos el pipeline a la GPU para tener una inferencia más rápida.

# %%
if torch.cuda.is_available():
    device=torch.device("cuda")
elif torch.backends.mps.is_available():
    device=torch.device("mps")

pipe = pipe.to(device)

# %%
# Y estamos listos para generar imágenes:

# %%
prompt = "a photograph of an astronaut riding a horse"
image = pipe(prompt).images[0]  # image here is in [PIL format](https://pillow.readthedocs.io/en/stable/)

# Now to display an image you can either save it such as:
image.save(f"astronaut_rides_horse.png")

# or if you're in a google colab you can directly display it with
image

# %%
prompt = "an astronaut with a dog in the moon"
image = pipe(prompt).images[0]  # image here is in [PIL format](https://pillow.readthedocs.io/en/stable/)

# Now to display an image you can either save it such as:
image.save(f"astronaut_rides_horse.png")

# or if you're in a google colab you can directly display it with
image

# %%
# Ejecutar la celda anterior varias veces le dará una imagen diferente cada vez. Si desea una salida determinista, puede pasar una semilla aleatoria al pipeline. Cada vez que use la misma semilla, obtendrá el mismo resultado de imagen.

# %%
generator = torch.Generator(device).manual_seed(1024)

image = pipe(prompt, generator=generator).images[0]

image

# %%
# Puede cambiar el número de pasos de inferencia usando el argumento `num_inference_steps`. En general, los resultados son mejores cuantos más pasos use. Stable Diffusion, siendo uno de los últimos modelos, funciona muy bien con un número relativamente pequeño de pasos, por lo que recomendamos usar el valor predeterminado de `50`. Si desea resultados más rápidos, puede usar un número menor.
# 
# La siguiente celda usa la misma semilla que antes, pero con menos pasos. Observe cómo algunos detalles, como la cabeza del caballo o el casco, son menos realistas y menos definidos que en la imagen anterior:

# %%
generator = torch.Generator(device).manual_seed(1024)

image = pipe(prompt, num_inference_steps=15, generator=generator).images[0]

image

# %%
# El otro parámetro en la llamada al pipeline es `guidance_scale`. Es una forma de aumentar la adherencia a la señal condicional que en este caso es texto, así como la calidad general de la muestra. En términos simples, la guía sin clasificador obliga a la generación a coincidir mejor con la indicación. Números como `7` u `8.5` dan buenos resultados, si usa un número muy grande, las imágenes pueden verse bien, pero serán menos diversas.
# 
# Puede aprender sobre los detalles técnicos de este parámetro en [la última sección](https://colab.research.google.com/drive/1ALXuCM5iNnJDNW5vqBm5lCtUQtZJHN2f?authuser=1#scrollTo=UZp-ynZLrS-S) de este cuaderno.

# %%
# Para generar múltiples imágenes para la misma indicación, simplemente usamos una lista con la misma indicación repetida varias veces. Enviaremos la lista al pipeline en lugar de la cadena que usamos antes.

# %%
# 
# Primero escribamos una función auxiliar para mostrar una cuadrícula de imágenes. Simplemente ejecute la siguiente celda para crear la función `image_grid`, o divulgue el código si está interesado en cómo se hace.

# %%
from PIL import Image

def image_grid(imgs, rows, cols):
    assert len(imgs) == rows*cols

    w, h = imgs[0].size
    grid = Image.new('RGB', size=(cols*w, rows*h))
    grid_w, grid_h = grid.size

    for i, img in enumerate(imgs):
        grid.paste(img, box=(i%cols*w, i//cols*h))
    return grid

# %%
# Ahora, podemos generar una imagen de cuadrícula una vez que se haya ejecutado el pipeline con una lista de 3 indicaciones.

# %%
num_images = 3
prompt = ["a photograph of an astronaut riding a horse", "a photograph of an astronaut riding a bull"]

images = pipe(prompt).images

grid = image_grid(images, rows=1, cols=2)
grid

# %%
# Y así es como generar una cuadrícula de imágenes de `n × m`.

# %%
num_cols = 3
num_rows = 4

prompt = ["a photograph of an astronaut riding a horse"] * num_cols

all_images = []
for i in range(num_rows):
  images = pipe(prompt).images
  all_images.extend(images)

grid = image_grid(all_images, rows=num_rows, cols=num_cols)
grid

# %%
# ### Generar imágenes no cuadradas
# 
# Stable Diffusion produce imágenes de `512 × 512` píxeles por defecto. Pero es muy fácil anular el valor predeterminado usando los argumentos `height` y `width`, por lo que puede crear imágenes rectangulares en proporciones de retrato o paisaje.
# 
# Estas son algunas recomendaciones para elegir buenos tamaños de imagen:
# - Asegúrese de que `height` y `width` sean múltiplos de `8`.
# - Bajar de 512 puede resultar en imágenes de menor calidad.
# - Superar 512 en ambas direcciones repetirá áreas de imagen (se pierde la coherencia global).
# - La mejor manera de crear imágenes no cuadradas es usar `512` en una dimensión y un valor mayor que ese en la otra.

# %%
prompt = "a photograph of an astronaut riding a horse"

image = pipe(prompt, height=512, width=768).images[0]
image

# %%
# ## 🎯 Tarea 1 — Prompting libre
# 
# Tu turno. **Modificá el prompt y ejecutá la celda varias veces.** El objetivo es desarrollar intuición sobre cómo el contenido del prompt afecta la imagen.
# 
# ### Pautas
# 
# Probá al menos **tres variaciones** de las siguientes:
# 
# 1. **Cambiar el sujeto** principal (de `astronaut` a `doctor`, `scientist`, `musician`, `robot`, etc.).
# 2. **Sumar atributos** específicos: edad, ropa, expresión, postura.
# 3. **Sumar estilo visual**: `oil painting`, `pencil sketch`, `cyberpunk`, `Studio Ghibli style`, `black and white photography`.
# 4. **Sumar condiciones de luz**: `dramatic shadows`, `soft natural lighting`, `golden hour`, `neon backlight`.
# 5. **Sumar un escenario**: `in a hospital corridor`, `on Mars`, `in a 1920s café`.
# 
# ### Plantilla recomendada para construir buenos prompts
# 
# ```
# [sujeto principal] + [atributos] + [acción/postura] + [escenario] + [estilo visual] + [condiciones de luz]
# ```
# 
# ### Reflexión
# 
# Después de probar varios prompts, contestá mentalmente (o en una celda de texto):
# 
# - ¿Qué tipo de cambios afectan **más** la imagen — sujeto, estilo o luz?
# - ¿Hay palabras o frases que el modelo parece **ignorar**? ¿Por qué creés que pasa?
# - ¿Qué pasa si escribís el prompt en español?

# %%
# 🎯 TU PROMPT ACÁ — completá y modificá libremente
my_prompt = "A high-resolution photograph of a doctor in a futuristic hospital corridor holding a digital tablet, soft volumetric lighting, cinematic composition, photorealistic, 8k"   # ← escribí tu prompt acá

image = pipe(my_prompt).images[0]
image.save("tarea1_prompt_libre.png")
image

# %%
# ## 🎯 Tarea 2 — Análisis de hiperparámetros
# 
# Diseñá tu propio experimento para entender cómo se comportan los hiperparámetros con tus prompts.
# 
# ### Mínimo a entregar
# 
# 1. **Elegí un prompt** propio y fijalo (no lo cambies durante la tarea).
# 2. **Fijá una seed** y mantenela.
# 3. **Generá 4 imágenes** variando solo `num_inference_steps`. ¿A partir de cuántos pasos la calidad deja de mejorar para tu prompt?
# 4. **Generá 4 imágenes más** variando solo `guidance_scale`. ¿Cuál es el rango "natural"?
# 5. **Generá 4 imágenes** variando solo la `seed` (mantené el resto). ¿Qué cambia y qué se mantiene entre seeds?
# 
# ### Reflexión
# 
# - ¿Qué pasa cuando subís el `guidance_scale` muy alto (≥ 20)? Describí el artefacto visual que aparece.
# - ¿Cuándo conviene usar 25 pasos en lugar de 50? Pensá en deployment a escala (costo por imagen, latencia).
# - Si tu equipo tuviera que generar 100.000 imágenes por día, ¿qué hiperparámetros priorizarías y por qué?
# 
# ### Pista
# 
# Recordá la sintaxis de los ejemplos anteriores:
# ```python
# generator = torch.Generator(device).manual_seed(SEED)
# image = pipe(prompt,
#              num_inference_steps=STEPS,
#              guidance_scale=SCALE,
#              generator=generator).images[0]
# ```

# %%
# 🎯 TU EXPERIMENTO ACÁ
# Sugerencia: armá un loop como los anteriores con tus propios valores

my_prompt = "A high-resolution photograph of a doctor in a futuristic hospital corridor holding a digital tablet, soft volumetric lighting, cinematic composition, photorealistic, 8k"        # ← completá con tu prompt
my_seed = 42          # ← elegí una seed

# Sub-experimento 1: variar num_inference_steps con seed y guidance_scale fijos
print("--- Experimento 1: Variando num_inference_steps ---")
steps_list = [5, 15, 30, 50]
images_steps = []
for steps in steps_list:
    generator = torch.Generator(device).manual_seed(my_seed)
    image = pipe(my_prompt, num_inference_steps=steps, guidance_scale=7.5, generator=generator).images[0]
    image.save(f"experiment_steps_{steps}.png")
    images_steps.append(image)

grid_steps = image_grid(images_steps, rows=1, cols=4)
grid_steps.save("grid_steps.png")
grid_steps

# Sub-experimento 2: variar guidance_scale con seed y steps fijos
print("--- Experimento 2: Variando guidance_scale ---")
scales_list = [1.0, 3.0, 7.5, 20.0]
images_scales = []
for scale in scales_list:
    generator = torch.Generator(device).manual_seed(my_seed)
    image = pipe(my_prompt, num_inference_steps=30, guidance_scale=scale, generator=generator).images[0]
    image.save(f"experiment_scale_{scale}.png")
    images_scales.append(image)

grid_scales = image_grid(images_scales, rows=1, cols=4)
grid_scales.save("grid_scales.png")
grid_scales

# Sub-experimento 3: variar seed con steps y guidance_scale fijos
print("--- Experimento 3: Variando seed ---")
seeds_list = [10, 42, 123, 999]
images_seeds = []
for s in seeds_list:
    generator = torch.Generator(device).manual_seed(s)
    image = pipe(my_prompt, num_inference_steps=30, guidance_scale=7.5, generator=generator).images[0]
    image.save(f"experiment_seed_{s}.png")
    images_seeds.append(image)

grid_seeds = image_grid(images_seeds, rows=1, cols=4)
grid_seeds.save("grid_seeds.png")
grid_seeds

# %%
# ## 2. Qué es Stable Diffusion
# 
# Ahora, pasemos a la parte teórica de Stable Diffusion 👩‍🎓.
# 
# Stable Diffusion se basa en un tipo particular de modelo de difusión llamado **Difusión Latente**, propuesto en [High-Resolution Image Synthesis with Latent Diffusion Models](https://arxiv.org/abs/2112.10752).
# 

# %%
# Los modelos de difusión generales son sistemas de aprendizaje automático que están entrenados para *eliminar ruido* del ruido gaussiano aleatorio paso a paso, para llegar a una muestra de interés, como una *imagen*. Para obtener una descripción más detallada de cómo funcionan, consulte [este colab](https://colab.research.google.com/github/huggingface/notebooks/blob/main/diffusers/diffusers_intro.ipynb).
# 
# Los modelos de difusión han demostrado lograr resultados de vanguardia para generar datos de imagen. Pero una desventaja de los modelos de difusión es que el proceso inverso de eliminación de ruido es lento. Además, estos modelos consumen mucha memoria porque operan en el espacio de píxeles, lo que se vuelve excesivamente costoso al generar imágenes de alta resolución. Por lo tanto, es difícil entrenar estos modelos y también usarlos para inferencia.

# %%
# 
# <br>
# 
# La difusión latente puede reducir la complejidad de memoria y cómputo aplicando el proceso de difusión sobre un espacio _latente_ de menor dimensión, en lugar de usar el espacio de píxeles real. Esta es la diferencia clave entre la difusión estándar y los modelos de difusión latente: **en la difusión latente, el modelo está entrenado para generar representaciones latentes (comprimidas) de las imágenes.**
# 
# Hay tres componentes principales en la difusión latente.
# 
# 1. Un autocodificador (VAE).
# 2. Una [U-Net](https://colab.research.google.com/github/huggingface/notebooks/blob/main/diffusers/diffusers_intro.ipynb#scrollTo=wW8o1Wp0zRkq).
# 3. Un codificador de texto, *p. ej.* [Codificador de texto de CLIP](https://huggingface.co/docs/transformers/model_doc/clip#transformers.CLIPTextModel).

# %%
# **1. El autocodificador (VAE)**
# 
# El modelo VAE tiene dos partes, un codificador y un decodificador. El codificador se usa para convertir la imagen en una representación latente de baja dimensión, que servirá como entrada al modelo *U-Net*.
# El decodificador, a la inversa, transforma la representación latente de nuevo en una imagen.
# 
# Durante el _entrenamiento_ de difusión latente, el codificador se usa para obtener las representaciones latentes (_latentes_) de las imágenes para el proceso de difusión hacia adelante, que aplica cada vez más ruido en cada paso. Durante la _inferencia_, los latentes eliminados de ruido generados por el proceso de difusión inversa se convierten nuevamente en imágenes usando el decodificador VAE. Como veremos durante la inferencia, **solo necesitamos el decodificador VAE**.

# %%
# **2. La U-Net**
# 
# La U-Net tiene una parte codificadora y una parte decodificadora, ambas compuestas por bloques ResNet.
# El codificador comprime una representación de imagen en una representación de imagen de menor resolución y el decodificador decodifica la representación de imagen de menor resolución de vuelta a la representación de imagen original de mayor resolución que supuestamente tiene menos ruido.
# Más específicamente, la salida de U-Net predice el residuo de ruido que se puede usar para calcular la representación de imagen desruidada predicha.
# 
# Para evitar que la U-Net pierda información importante durante el muestreo descendente, generalmente se agregan conexiones de acceso directo entre los ResNets de muestreo descendente del codificador a los ResNets de muestreo ascendente del decodificador.
# Además, la U-Net de difusión estable puede condicionar su salida en incrustaciones de texto a través de capas de atención cruzada. Las capas de atención cruzada se agregan tanto a la parte del codificador como del decodificador de la U-Net, generalmente entre bloques ResNet.

# %%
# **3. El codificador de texto**
# 
# El codificador de texto es responsable de transformar la indicación de entrada, *p. ej.* "Un astronauta montando a caballo" en un espacio de incrustación que puede ser entendido por la U-Net. Por lo general, es un codificador simple *basado en transformador* que mapea una secuencia de tokens de entrada a una secuencia de incrustaciones de texto latentes.
# 
# Inspirado por [Imagen](https://imagen.research.google/), Stable Diffusion **no** entrena el codificador de texto durante el entrenamiento y simplemente usa un codificador de texto ya entrenado de CLIP, [CLIPTextModel](https://huggingface.co/docs/transformers/model_doc/clip#transformers.CLIPTextModel).

# %%
# **¿Por qué la difusión latente es rápida y eficiente?**
# 
# Dado que la U-Net de los modelos de difusión latente opera en un espacio de baja dimensión, reduce en gran medida los requisitos de memoria y cómputo en comparación con los modelos de difusión en el espacio de píxeles. Por ejemplo, el autocodificador utilizado en Stable Diffusion tiene un factor de reducción de 8. Esto significa que una imagen de forma `(3, 512, 512)` se convierte en `(3, 64, 64)` en el espacio latente, lo que requiere `8 × 8 = 64` veces menos memoria.
# 
# ¡Por eso es posible generar imágenes de `512 × 512` tan rápido, incluso en GPUs de Colab de 16GB!

# %%
# **Stable Diffusion durante la inferencia**
# 
# Juntando todo, veamos ahora más de cerca cómo funciona el modelo en la inferencia ilustrando el flujo lógico.

# %%
# <p align="left">
# <img src="https://raw.githubusercontent.com/patrickvonplaten/scientific_images/master/stable_diffusion.png" alt="sd-pipeline" width="500"/>
# </p>
# 
# El modelo de difusión estable toma tanto una semilla latente como una indicación de texto como entrada. La semilla latente se utiliza luego para generar representaciones de imagen latente aleatorias de tamaño $64 \times 64$, mientras que la indicación de texto se transforma en incrustaciones de texto de tamaño $77 \times 768$ a través del codificador de texto de CLIP.
# 
# A continuación, la U-Net *elimina el ruido* iterativamente de las representaciones de imagen latente aleatorias mientras está condicionada en las incrustaciones de texto. La salida de la U-Net, siendo el residuo de ruido, se usa para calcular una representación de imagen latente desruidada a través de un algoritmo de planificador. Se pueden usar muchos algoritmos de planificador diferentes para este cálculo, cada uno con sus pros y sus contras. Para Stable Diffusion, recomendamos usar uno de:
# 
# - [Planificador PNDM](https://github.com/huggingface/diffusers/blob/main/src/diffusers/schedulers/scheduling_pndm.py) (usado por defecto).
# - [Planificador K-LMS](https://github.com/huggingface/diffusers/blob/main/src/diffusers/schedulers/scheduling_lms_discrete.py).
# - [Planificador Heun Discreto](https://github.com/huggingface/diffusers/blob/main/src/diffusers/schedulers/scheduling_heun_discrete.py).
# - [Planificador DPM Solver Multipaso](https://github.com/huggingface/diffusers/blob/main/src/diffusers/schedulers/scheduling_dpmsolver_multistep.py). Este planificador puede lograr una gran calidad en menos pasos. ¡Puede probar con 25 en lugar de los 50 predeterminados!
# 
# La teoría sobre cómo funciona el algoritmo del planificador está fuera del alcance de este cuaderno, pero en resumen, uno debe recordar que calculan la representación de imagen desruidada predicha a partir de la representación de ruido anterior y el residuo de ruido predicho.
# Para obtener más información, recomendamos consultar [Elucidating the Design Space of Diffusion-Based Generative Models](https://arxiv.org/abs/2206.00364)
# 
# El proceso de *eliminación de ruido* se repite *aprox.* 50 veces para recuperar paso a paso mejores representaciones de imagen latente.
# Una vez completo, la representación de imagen latente es decodificada por la parte decodificadora del autocodificador variacional.

# %%
# 
# Después de esta breve introducción a la Difusión Latente y Estable, ¡veamos cómo hacer un uso avanzado de 🤗 Hugging Face Diffusers!

# %%
# ## 3. Cómo escribir su propio pipeline de inferencia con `diffusers`
# 
# Finalmente, mostramos cómo puede crear pipelines de difusión personalizados con `diffusers`.
# Esto suele ser muy útil para profundizar un poco más en ciertas funcionalidades del sistema y para potencialmente intercambiar ciertos componentes.
# 
# En esta sección, demostraremos cómo usar Stable Diffusion con un planificador diferente, es decir, el planificador K-LMS de [Katherine Crowson](https://github.com/crowsonkb) que se agregó en [este PR](https://github.com/huggingface/diffusers/pull/185#pullrequestreview-1074247365).

# %%
# Repasemos el `StableDiffusionPipeline` paso a paso para ver cómo podríamos haberlo escrito nosotros mismos.
# 
# Comenzaremos cargando los modelos individuales involucrados.

# %%
# El [modelo preentrenado](https://huggingface.co/CompVis/stable-diffusion-v1-3-diffusers/tree/main) incluye todos los componentes necesarios para configurar un pipeline de difusión completo. Se almacenan en las siguientes carpetas:
# - `text_encoder`: Stable Diffusion usa CLIP, pero otros modelos de difusión pueden usar otros codificadores como `BERT`.
# - `tokenizer`. Debe coincidir con el utilizado por el modelo `text_encoder`.
# - `scheduler`: El algoritmo de planificación utilizado para agregar progresivamente ruido a la imagen durante el entrenamiento.
# - `unet`: El modelo utilizado para generar la representación latente de la entrada.
# - `vae`: Módulo de autocodificador que usaremos para decodificar representaciones latentes en imágenes reales.
# 
# Podemos cargar los componentes haciendo referencia a la carpeta en la que se guardaron, usando el argumento `subfolder` en `from_pretrained`.

# %%
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import AutoencoderKL, UNet2DConditionModel, PNDMScheduler

# 1. Load the autoencoder model which will be used to decode the latents into image space.
vae = AutoencoderKL.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="vae")

# 2. Load the tokenizer and text encoder to tokenize and encode the text.
tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14")

# 3. The UNet model for generating the latents.
unet = UNet2DConditionModel.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="unet")

# %%
# Ahora, en lugar de cargar el planificador predefinido, usaremos el planificador K-LMS en su lugar.

# %%
from diffusers import LMSDiscreteScheduler

scheduler = LMSDiscreteScheduler.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="scheduler")

# %%
# A continuación, movemos los modelos a la GPU.

# %%
vae = vae.to(device)
text_encoder = text_encoder.to(device)
unet = unet.to(device)

# %%
# Ahora definimos los parámetros que usaremos para generar imágenes.
# 
# Tenga en cuenta que `guidance_scale` se define de manera análoga al peso de guía `w` de la ecuación (2) en el [artículo de Imagen](https://arxiv.org/pdf/2205.11487.pdf). `guidance_scale == 1` corresponde a no hacer guía sin clasificador. Aquí lo establecemos en 7.5 como también se hizo anteriormente.
# 
# A diferencia de los ejemplos anteriores, establecemos `num_inference_steps` en 100 para obtener una imagen aún más definida.

# %%
prompt = ["a photograph of an astronaut riding a horse"]

height = 512                        # default height of Stable Diffusion
width = 512                         # default width of Stable Diffusion

num_inference_steps = 100            # Number of denoising steps

guidance_scale = 7.5                # Scale for classifier-free guidance

generator = torch.manual_seed(32)   # Seed generator to create the inital latent noise

batch_size = 1

# %%
# Primero, obtenemos las text_embeddings para la indicación. Estas incrustaciones se usarán para condicionar el modelo U-Net.

# %%
text_input = tokenizer(prompt, padding="max_length", max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt")

with torch.no_grad():
  text_embeddings = text_encoder(text_input.input_ids.to(device))[0]

# %%
# También obtendremos las incrustaciones de texto incondicionales para la guía sin clasificador, que son solo las incrustaciones para el token de relleno (texto vacío). Deben tener la misma forma que las `text_embeddings` condicionales (`batch_size` y `seq_length`)

# %%
max_length = text_input.input_ids.shape[-1]
uncond_input = tokenizer(
    [""] * batch_size, padding="max_length", max_length=max_length, return_tensors="pt"
)
with torch.no_grad():
  uncond_embeddings = text_encoder(uncond_input.input_ids.to(device))[0]

# %%
# Para la guía sin clasificador, necesitamos hacer dos pases hacia adelante. Uno con la entrada condicionada (`text_embeddings`), y otro con las incrustaciones incondicionales (`uncond_embeddings`). En la práctica, podemos concatenar ambos en un solo lote para evitar hacer dos pases hacia adelante.

# %%
text_embeddings = torch.cat([uncond_embeddings, text_embeddings])

# %%
# Genere el ruido aleatorio inicial.

# %%
latents = torch.randn(
  (batch_size, unet.in_channels, height // 8, width // 8),
  generator=generator,
)
latents = latents.to(device)

# %%
latents.shape

# %%
# Genial, se esperaba $64 \times 64$. El modelo transformará esta representación latente (ruido puro) en una imagen de `512 × 512` más adelante.
# 
# A continuación, inicializamos el planificador con nuestro `num_inference_steps` elegido.
# Esto calculará los `sigmas` y los valores exactos de pasos de tiempo que se usarán durante el proceso de eliminación de ruido.

# %%
scheduler.set_timesteps(num_inference_steps)

# %%
# El planificador K-LMS necesita multiplicar los `latents` por sus valores `sigma`. Hagamos esto aquí

# %%
latents = latents * scheduler.init_noise_sigma

# %%
# Estamos listos para escribir el bucle de eliminación de ruido.

# %%
from tqdm.auto import tqdm
from torch import autocast

for t in tqdm(scheduler.timesteps):
  # expand the latents if we are doing classifier-free guidance to avoid doing two forward passes.
  latent_model_input = torch.cat([latents] * 2)

  latent_model_input = scheduler.scale_model_input(latent_model_input, t)

  # predict the noise residual
  with torch.no_grad():
    noise_pred = unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample

  # perform guidance
  noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
  noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

  # compute the previous noisy sample x_t -> x_t-1
  latents = scheduler.step(noise_pred, t, latents).prev_sample

# %%
# Ahora usamos el `vae` para decodificar los `latents` generados de nuevo en la imagen.

# %%
# scale and decode the image latents with vae
latents = 1 / 0.18215 * latents

with torch.no_grad():
  image = vae.decode(latents).sample

# %%
# Y finalmente, convirtamos la imagen a PIL para que podamos mostrarla o guardarla.

# %%
image = (image / 2 + 0.5).clamp(0, 1)
image = image.detach().cpu().permute(0, 2, 3, 1).numpy()
images = (image * 255).round().astype("uint8")
pil_images = [Image.fromarray(image) for image in images]
pil_images[0]

# %%
# Ahora tiene todas las piezas para construir sus propios pipelines o usar componentes de diffusers como desee 🔥.

# %%
# ## 🎯 Tarea 3 — Aplicación al dominio biomédico (opcional)
# 
# > 📝 **Aunque esta tarea sea opcional, suma puntos a la nota final.** Hacerla muestra interés y compromiso, y queda registrada en la entrega.
# 
# En clase discutimos que Stable Diffusion fue entrenado con **LAION-5B** — un dataset de imágenes de internet con sus descripciones. Eso plantea preguntas interesantes para el dominio biomédico:
# 
# - ¿Qué tan bien puede SD generar imágenes con **estilo médico**? (microscopía, radiografía, eco, dermatología.)
# - ¿Las imágenes generadas son **anatómicamente correctas** o solo estéticamente convincentes?
# - ¿Qué **sesgos** demográficos o de presentación introduce el modelo cuando le pedís personajes médicos?
# - ¿Sirve para **aumentar datasets sintéticos** en aplicaciones biomédicas, o las limitaciones del modelo lo descalifican?
# 
# ### Tarea
# 
# Elegí **uno o más** ejes y generá imágenes que te ayuden a responder. Algunos prompts sugeridos:
# 
# #### Estilo médico
# 
# ```
# "a high magnification microscopy image of liver tissue, hematoxylin and eosin stain"
# "a chest X-ray showing pneumonia, black and white, clinical photograph"
# "a dermoscopy image of a melanocytic skin lesion"
# "a fundus photograph of a healthy retina with visible blood vessels"
# ```
# 
# #### Anatomía / fidelidad
# 
# ```
# "a detailed anatomical drawing of a human hand showing all bones, medical illustration"
# "a labeled diagram of the human heart with all chambers and valves"
# ```
# 
# #### Análisis de sesgos
# 
# ```
# "a photograph of a doctor"          # ¿qué sale por defecto? género, etnia, edad
# "a photograph of a nurse"           # ¿hay estereotipo de género?
# "a photograph of a patient in a hospital bed"   # ¿edad y demografía?
# "a photograph of a surgeon performing surgery"  # ¿composición típica?
# ```
# 
# > **Pista:** generá varias imágenes con el mismo prompt y distintas seeds (usá `image_grid` que ya definiste antes). Una sola imagen no alcanza para detectar un sesgo.
# 
# ### Reflexión final
# 
# Después de tus experimentos, contestá:
# 
# - ¿Qué tan **realistas** son las imágenes médicas generadas? ¿Pasarían como reales para un especialista?
# - ¿Son **clínicamente válidas** o solo estéticamente convincentes?
# - ¿Cuándo **sí** y cuándo **no** usarías estas imágenes para entrenar otro modelo?
# - ¿Qué cuidados éticos o regulatorios habría que tomar?

# %%
# 🎯 TU EXPERIMENTO BIOMÉDICO ACÁ

biomedical_prompts = [
    # Tu prompt 1 (ej: estilo médico)
    "",
    # Tu prompt 2 (ej: anatomía)
    "",
    # Tu prompt 3 (ej: análisis de sesgo)
    "",
]

for p in biomedical_prompts:
    if not p:
        continue
    image = pipe(p).images[0]
    print(f"\nPrompt: {p}")
    display(image)

# %%
# ## 🎯 Tareas avanzadas (opcional)
# 
# > 📝 **Aunque esta tarea sea opcional, suma puntos a la nota final.** Hacerla muestra interés y compromiso, y queda registrada en la entrega.
# 
# Si terminaste lo anterior y querés profundizar, acá hay tres direcciones interesantes — cada una corresponde a una capacidad distinta de Stable Diffusion.
# 
# ### A. Negative prompts
# 
# Hasta ahora le dijimos al modelo **qué queremos**. También podemos decirle **qué NO queremos** mediante un *negative prompt*. Esto se hace reemplazando el embedding incondicional (el del prompt vacío) por el embedding del *negative prompt*.
# 
# > **Implementación:** modificá el pipeline manual de la sección 3. El cambio mínimo es reemplazar
# >
# > ```python
# > uncond_input = tokenizer([""] * batch_size, ...)
# > ```
# >
# > por
# >
# > ```python
# > uncond_input = tokenizer(["bad anatomy, extra fingers, blurry, low quality, distorted face"] * batch_size, ...)
# > ```
# >
# > Compará la imagen con y sin negative prompt — el efecto es notable, sobre todo en manos y caras.
# 
# ### B. Visualización del proceso de denoising
# 
# Modificá el pipeline manual para **guardar el latente cada N pasos** y decodificarlo con el VAE. Vas a poder ver cómo emerge la imagen progresivamente desde el ruido.
# 
# > **Implementación:** dentro del loop de denoising, cada 5 pasos copiá el latente actual, decodificalo con el VAE y guardalo. Al final armá un grid con todos los snapshots.
# >
# > Es una de las visualizaciones más reveladoras de cómo funciona la difusión. Vas a ver el ruido transformarse gradualmente en una imagen reconocible.
# 
# ### C. Image-to-image: edición controlada
# 
# `StableDiffusionImg2ImgPipeline` permite tomar una imagen existente y reformularla con un prompt.
# 
# **Caso biomédico imaginario:** usar img2img para generar variantes estilísticas de una radiografía existente, evaluando si SD preserva la estructura anatómica relevante.
# 
# ```python
# from diffusers import StableDiffusionImg2ImgPipeline
# 
# img2img = StableDiffusionImg2ImgPipeline.from_pretrained(
#     "CompVis/stable-diffusion-v1-4", torch_dtype=torch.float16,
# ).to(device)
# 
# result = img2img(
#     prompt="a stylized version of this image",
#     image=initial_image,
#     strength=0.5,
# ).images[0]
# ```
# 
# > El parámetro `strength` (0-1) controla cuánto se aleja del original. Probá con strengths 0.3, 0.5, 0.7, 0.9 sobre la misma imagen base.

# %%
# 🎯 TAREAS AVANZADAS — TU CÓDIGO ACÁ

# Sugerencia: empezá por A (negative prompts) reutilizando las celdas
# del pipeline manual de la sección 3.

# %%
# ## Cierre
# 
# En esta notebook recorrimos Stable Diffusion en tres niveles:
# 
# 1. **Alto nivel** — `pipe(prompt)` te da una imagen en una línea.
# 2. **Hiperparámetros** — `num_inference_steps`, `guidance_scale`, `seed` son las perillas que controlan calidad, fidelidad y reproducibilidad.
# 3. **Bajo nivel** — desarmaste el pipeline y viste el bucle de denoising operando con VAE + UNET + CLIP por separado.
# 
# ### Lo que conviene llevarse
# 
# - Stable Diffusion **no es magia**: son tres redes pre-entrenadas orquestadas por un loop de ~50 pasos. En cada paso, la UNET predice ruido y el scheduler lo resta.
# - La **UNET** que vimos en segmentación (Sección 2 de la clase) es la **misma** que talla la imagen acá — un patrón arquitectónico muy reutilizable.
# - En aplicaciones biomédicas, **los sesgos del dataset y la falta de fidelidad anatómica** son las dos limitaciones principales. Saber dónde falla el modelo es tan importante como saber qué puede hacer.
# 
# ### Para seguir explorando
# 
# - **DreamBooth / LoRA** — cómo *fine-tunear* SD con un puñado de imágenes propias (útil cuando SD no conoce tu dominio, como pasaba con Slanted Land).
# - **ControlNet** — cómo guiar SD con una imagen estructural (un boceto, un mapa de profundidad, una pose).
# - **Stable Diffusion XL / FLUX / SD 3** — modelos más recientes con mejor calidad, mejor texto, mejor anatomía.
# - **Diffusion para imágenes médicas** — papers de difusión condicionada para síntesis MRI→CT, augmentación de patologías raras, anonimización con realismo clínico.
# 
# ---
# 
# **Profesores:** Paulo Veiga · Marco Sanchez Sorondo
# **Materia: IA para Ingeniería Biomédica**

# %%


