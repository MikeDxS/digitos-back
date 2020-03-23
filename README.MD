# Información
API que hace uso de un modelo de aprendizaje automático entrenado con los datos MNIST para la clasificación de dígitos escritos a mano.

## Antes de ejecutar
Para ejecutarlo se debe primero correr el siguiente comando:
`pip install -r requirements.txt`

## Ejecutar API
Para correr el programa sería:
`python back.py`

## Ejecutar entrenamiento
Para entrenar la red sería:
`python dnn.py`

## Consideraciones
Las petición de la clasificación se hace a través de la URL:
`http://localhost:5000/prediccion`
Y el body es un JSON de la forma:
{
	"url": "<url de la imagen>"
}
Y retorna la información como:
{
	"url": "<url de la imagen>",
	"prediccion": "número que representa que digito es"
}