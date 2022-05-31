import sys, os
# print(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), "../nlp"))
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), "../ip"))


from flask import Flask, request, render_template
from nlp.train import BertClassifier
from nlp.predict import predict
from nlp.utils import clean_question, find_objects, list_to_string
import speech_recognition as sr
import torch
from ip.demo import detect_cv2, color_result, object_result, object_count_result

app = Flask(__name__, template_folder='templates', static_folder='static')
app.debug = True

r = sr.Recognizer()
model = BertClassifier()
model.load_state_dict(torch.load('../nlp/model/bert.pt', map_location=torch.device('cpu')))
cfg_file = '../ip/cfg/yolov4.cfg'
weight_file = '../ip/weights/yolov4.weights'


def speech_to_text():
    with sr.Microphone() as source:
        # read the audio data from the default microphone
        audio_data = r.record(source, duration=5)
        print("Recognizing...")
        # convert speech to text
        text = r.recognize_google(audio_data)
        return text


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        text = speech_to_text()
        cl = clean_question(text)

        try:
            obj = find_objects(list_to_string(cl))
            obj = obj[0]
        except Exception as err:
            obj = None
            pass

        try:
            key, label_id = predict(model, text=text)
            image_file = request.form['uploaded']
            image_path = f"./static/images/{image_file}"
            print('img: ', image_file)
            print('path:', image_path)
            all_list = detect_cv2(cfgfile=cfg_file, weightfile=weight_file, imgfile=image_path)
        except Exception as err:
            raise err

        if key == 'number':
            if obj == 'people':
                obj = 'person'
            result = object_count_result(all_list, object_name=obj)
        elif key == 'object':
            result = object_result(all_list)
        else:
            if obj == 'people':
                obj = 'person'
            result = color_result(all_list, object_name=obj)
        return render_template('index.html', cls=str(result))
    else:
        return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True)
