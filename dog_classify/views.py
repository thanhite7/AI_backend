from django.shortcuts import render
from . import engine
from django.http import HttpResponse
import numpy as np
import cv2
from django.http import JsonResponse
from PIL import Image
from io import BytesIO
from keras.preprocessing.image import load_img
import random
def predict(request):
    if request.method == 'POST':
        image_file = request.FILES.get('image')

        if image_file:
            image_file.seek(0)
            image_file = BytesIO(image_file.read())
            image_file = Image.open(image_file)
            image_file = image_file.resize((331,331))
            img_g = np.expand_dims(image_file, axis=0)
            test_features = engine.extract_features(img_g)
            predg = engine.model.predict(test_features)

            predglabel = np.argsort(predg[0])[::-1]
            predgaccuracy = np.sort(predg[0])[::-1]
            lb1 = engine.classes[predglabel[0]]
            lb2 = engine.classes[predglabel[1]]
            lb3 = engine.classes[predglabel[2]]

            return render(request, 'index.html', {  'prediction1': lb1,
                                                    'accuracy1':round(predgaccuracy[0]* 100,3) ,
                                                    'prediction2': lb2,
                                                    'accuracy2':round(predgaccuracy[1]* 100,3) ,
                                                    'prediction3': lb3,
                                                    'accuracy3':round(predgaccuracy[2]* 100,3),     
                                                    'list1': pred_image_generate(engine.classes[predglabel[0]]),
                                                    'list2': pred_image_generate(engine.classes[predglabel[1]]),
                                                    'list3': pred_image_generate(engine.classes[predglabel[2]]),
            })
    return render(request, 'index.html')

def pred_image_generate(label):
    image_list = []
    for i in range(len(engine.labels['id'])):
        if(engine.labels['breed'][i]==label):
            image_list.append(engine.labels['id'][i]+'.jpg')
    selected_images = random.sample(image_list, min(10, len(image_list)))
    return selected_images
