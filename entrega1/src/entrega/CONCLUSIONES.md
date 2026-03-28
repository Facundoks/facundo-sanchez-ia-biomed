# Conclusiones - Práctica LLMs para Biomedicina

**Nombre:** Facundo Sanchez y Felix Vischi
**Fecha:** 27/03/2026

---

## Ejercicio 1: Primera Llamada

### 1. Diferencia entre respuesta sin y con system instruction
Sin un system instruction la IA toma un rol mas generalista y trata de no alejarse de la informacion que se le da. Pero si se le agrega un system instruction se puede moldear la respuesta de la IA para que sea mas especifica y acorde a lo que se busca, en el caso de ejemplo se le da un rol de medico pero con un lenguaje mas sencillo, que es lo mejor para dar a un paciente que quiere saber lo que tiene pero no entiende terminos medicos.

### 2. ¿Pudiste modificar los parámetros internos del modelo? ¿Qué sí controlaste?
No, solo controle lo que es el promnt y el sustem intruction, parametros como la temperatura, topk, topP y maxOutputTokens no los pude modificar.

### 3. ¿Qué pasaría si cambiaras el rol en el system instruction?
El modelo se adaptara al nuevo rol en el system instruction

### 4. ¿Qué system instruction sería útil para tu campo de estudio?
Si tengo una aplicacion para medicos que use IA, puedo darle al modelo un system promnt que indique que siemrpe trabaja con medicos, asi que no debe esplicar temas basicos de medicina, sino que debe ir directo a lo que interesa al medico.

---

## Ejercicio 2: Hiperparámetros

### 1. ¿Qué temperature usarías para un informe médico? ¿Y para brainstorming?
Para un informe médico usaria una temperatura 0 o 0.1, para que el modelo sea lo mas preciso posible sin que empiece a dar respuestas pueden desguiar al medico. Para brainstorming usaria una temperatura 1 o 2, para que el modelo sea mas creativo y pueda generar mas ideas.

### 2. ¿Qué pasó con maxOutputTokens=50? ¿Fue útil?
En los casos de ejemplo no, debido a que el modelo no pudo generar la respuesta completa, sino que lo termina cortando en medio y al final no da una respuesta coherente. Puede servir para casos mas especificos donde se sabe que la respuesta no va a ser muy larga, que se deberia poder logar con un system intruction que le diga al modelo que sea mas conciso o limite las palabras/tokens.

### 3. Diferencia entre topP bajo y alto
En terminos generales, no hay muchas diferencias entre topP bajo y alto, ambos generan respuestas similares, aunque con topP se ve que el modelo genera el texto como si fuera un informe medico, mientras que con topP alto genera el texto mas personalisado (refiriendose a que no es tan formal y habla con "vos").

### 4. ¿Las respuestas con temperature=0 fueron idénticas? Implicancias para reproducibilidad
En este caso fueron respuestas identicas (habiendolo corrido 3 veces, no hay cambios), hasta aumentandolo a 0.1 no hay cambios. Para la reproducibilidad esto es bueno, debido a que podemos tener cierta certeza a que el modelo en una situacion con misma memoria y promnt, realizara una respuesta identica/casi identica.

### 5. Hiperparámetros ideales para un chatbot médico. Justificá.
Para un chatbot medico usaria una temperatura 0.1, topP 0.3, topK 40 y maxOutputTokens 2000. La temperatura baja para que el modelo sea mas preciso, topP y topK bajo para que sea mas conciso (y si se trata de un caso bien estudiado, deberia ser mas preciso) y maxOutputTokens alto para que el modelo pueda generar la respuesta completa.

---

## Ejercicio 3: Prompt Engineering

### 1. Ranking de técnicas (peor a mejor) con justificación
1. Role + Constraints: Es el que mejores resultados dio, ademas que otorga las respuestas en un formato que se puede usar facilmente en una aplicacion, e incluso en casos que simplemente se requiera de respuestas con un mismo formato entre distintos promnts.
2 o 3. Few-shot: Al otorgar ejemplos, el modelo puede entender mejor lo que se espera de el, aunque no siempre da el formato deseado, al menos eso se vio en el caso B donde fue la respuesta mas corta de todas las tecnicas (aunque tambien puede ser algo bueno segun como se lo mire)
2 o 3. Chain-of-Thought: Al marcar los pasos que debe seguir el modelo lo obligas a generar un razonamiento mas profuendo, pero es el que menos formato posee respecto a los dos anteriores
4. Zero-shot: Sin una guia el modelo hace lo que le parece mejor, lo cual no siempre es lo que se busca, ademas que no otorga un formato especifico, lo que lo hace poco util. Ademas que depende mucho de la memoria del modelo para dar una respuesta coherente.

### 2. ¿La respuesta JSON fue clínicamente correcta? Ventajas del output estructurado
No, en los 3 casos no otorgo un diagnostico correcto, aunque si otorgo un diagnostico plausible. La ventaja del output estructurado es que se puede usar facilmente en una aplicacion, e incluso en casos que simplemente se requiera de respuestas con un mismo formato entre distintos promnts.

### 3. ¿El chain-of-thought cambió el diagnóstico o solo el razonamiento?
No cambio el diagnostico solo el razonamiento, aunque si hizo que el modelo sea mas preciso en su razonamiento, ya que al obligarlo a seguir pasos, este se ve forzado a pensar mas en lo que esta haciendo (o al menos se evidencia que camino tomo para llegar a esa respuesta)

### 4. ¿Encontraste información incorrecta presentada con confianza? ¿Cómo mitigarlo?
En la parte D, el modelo otorga lo siguiente: "Anemia microcítica (VCM elevado: 112 fL)" lo cual es incorrecto, ya que el VCM es de 112 fL lo cual es macrocitosis, no microcitosis (este error no lo vimos en los otros casos). Para mitigarlo se podria agregar mas reglas que indique los pasos a seguir o que ante los valores que son otorgados, explicar el porque del diagnostico dado, para que el medico pueda verificar si es correcto o no.

### 5. Tu diseño ideal de asistente diagnóstico
El diseño ideal que se diseño incluye de todos un poco, el Chain-of-Thought para poder ver el razonamiento y que el modelo se explaye (asi si hay un error, ver donde empezo a fallar), el Role + Constraints para que el modelo sepa que rol debe tomar y que formato debe usar, y el Few-shot para que el modelo tenga ejemplos de como debe responder. Ademas de un system instruction que le indique al modelo que debe ser preciso y que debe explicar el porque de sus respuestas.
No se incluyo un formato especifico como en la Parte D porque no vimos necesario o al menos pensamos que podiamos limitar al modelo al no otrogar campos especificos.

---

## Reflexión Final

### ¿Qué aprendiste que no esperabas?
Los distintos tipos de parametros que se pueden modificar para que el modelo de una respuesta mas precisa o creativa, o poder limitar que palabras usa (o mejor dicho que use tokens de mayor probabilidad)

### ¿Qué riesgos ves en el uso de LLMs en medicina?
No se puede castigar a un modelo por dar una respuesta incorrecta, y tampoco se puede garantizar que el modelo de una respuesta correcta en todos los casos. No es lo mismo que una ia de una respuesta erroena en un codigo, donde simplemente puede dar un nuevo bug y corregirlo, sino que estamos hablando de la salud de las personas, y una respuesta incorrecta puede tener consecuencias fatales.

### ¿Qué oportunidades ves para tu área de especialización?
Una segunda voz, cuando un medico otorga un diagnostico, generalmente otorga segun las posibilidades, es decir que ante sintomas especificos, otorga el diagnostico a X enfermedad que es la mas comun, pero puede existir el caso que con esos sintomas mas otros que son mas dificiles de notar, se tenga una enfermedad Y. Con una IA que tenga la instrucción de dar varios diagnosticos distintos, puede mostrar que la enfermedad Y es una posibilidad, y el medico puede reconocerlo y volver a analizar el caso.
Tambien para todos temas de bioinformatica, donde se pueden usar LLMs para analizar grandes cantidades de datos y encontrar patrones que los humanos no pueden ver.