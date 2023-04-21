from transformers import AutoTokenizer, AutoModelForQuestionAnswering, pipeline
from textwrap import wrap

the_model = 'mrm8488/distill-bert-base-spanish-wwm-cased-finetuned-spa-squad2-es'
tokenizer = AutoTokenizer.from_pretrained(the_model, do_lower_case=False)
model = AutoModelForQuestionAnswering.from_pretrained(the_model)


# Ejemplo tokenización
contexto = 'Yo soy Marcos'
pregunta = '¿cómo me llamo?'

encode = tokenizer.encode_plus(pregunta, contexto, return_tensors='pt')
input_ids = encode['input_ids'].tolist()
tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
for id, token in zip(input_ids[0], tokens):
  print('{:<12} {:>6}'.format(token, id))
  print('')


# Ejemplo de inferencia (pregunta-respuesta)
nlp = pipeline('question-answering', model=model, tokenizer=tokenizer)
salida = nlp({'question':pregunta, 'context':contexto})
print(salida)


def pregunta_respuesta(model, contexto, nlp):

  # Imprimir contexto
  print('Contexto:')
  print('-----------------')
  print('\n'.join(wrap(contexto)))

  # Loop preguntas-respuestas:
  continuar = True
  while continuar:
    print('\nPregunta:')
    print('-----------------')
    pregunta = str(input())

    continuar = pregunta!=''

    if continuar:
      salida = nlp({'question':pregunta, 'context':contexto})
      print('\nRespuesta:')
      print('-----------------')
      print(salida['answer'])
