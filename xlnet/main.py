from transformers import pipeline

# Cargar el modelo de XLNet para preguntas y respuestas
model_name = "xlnet-base-cased"
model = pipeline("question-answering", model=model_name)

# Contexto
context = """
El huracán María fue un ciclón tropical extremadamente potente que azotó Puerto Rico en septiembre de 2017, siendo uno de los huracanes más catastróficos que hayan afectado a la isla. María fue el décimo huracán de la temporada de huracanes en el Atlántico de 2017, el cuarto huracán de categoría 5 de la temporada y el primer huracán de categoría 5 en afectar a Puerto Rico desde el huracán Hugo en 1989. María tocó tierra en Yabucoa el 20 de septiembre a las 6:15 a.m. con vientos máximos sostenidos de 250 km/h (155 mph) y rachas de 296 km/h (184 mph). 
"""

# Hacer una pregunta
question = "¿Cuándo tocó tierra el huracán María en Puerto Rico?"

# Obtener la respuesta
answer = model(question=question, context=context)

# Imprimir la respuesta
print(answer)
