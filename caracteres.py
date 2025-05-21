import easyocr

#Carga del modelo de lengua
reader = easyocr.Reader(['es']) 

#Reconocimiento de una imagen
res = reader.readtext('C:/Users/Vladimir/Documents/concurso/yes/yolo-plate-recognition-main/blanco.png')

for (bbox, text, prob) in res:
    # Coordenadas en orden 
    (top_left, top_right, bottom_right, bottom_left) = bbox
    print(f'\nTexto: {text}\nProbabilidad: {prob:.2f}\nContenedor: {tuple(map(int, top_left)),tuple(map(int, bottom_right))}')


#Con restricción de caracteres reconocibles
#result = reader.readtext('toy.tif', allowlist ='0123456789')