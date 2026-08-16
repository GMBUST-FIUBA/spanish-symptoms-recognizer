# Entrenamiento de modelo de reconocimiento de entidades (NER)

Esta sección se usa para poder entrenar el modelo NER de reconocimiento de entidades (fenotipos).

## Preparación

Se debe utilizar un ambiente virtual de Python (se utilizó `pyenv`) y una versión de Python antigua (se utilizó la 3.9.0) para ejecutar las librerías utilizadas por el BSC. Los modelos pueden luego utilizarse con otras versiones, pero para el entrenamiento deben usarse las versiones dadas en el archivo de requerimientos.

## Ejecución del entrenamiento

Para utilizarlo se dejaron los archivos necesarios en las distintas carpetas:

- `data`: para poder crear los datasets usados (naturales y sintéticos).
- `ner`: contiene el código de Python provisto por el BSC para hacer el *fine-tuning* al modelo base.
- `ner.sh`: ejecutar para iniciar el proceso de entrenamiento.