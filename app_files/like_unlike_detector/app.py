import gradio as gr
import tensorflow as tf

IMG_SIZE = (128, 128)

# load keras model
model_path = 'my_model.h5'
model = tf.keras.models.load_model(model_path)

categories = ('like', 'unlike')

def classify_image(img):
    img_array_expanded_dims = img.reshape((-1, 128, 128, 3))
    prediction = model.predict(img_array_expanded_dims)
    prediction_prob = float(tf.nn.sigmoid(prediction))
    probs = [1-prediction_prob, prediction_prob]
    return dict(zip(categories, probs))

gr_image = gr.inputs.Image(shape=IMG_SIZE)
label = gr.outputs.Label()
examples = ['like.png', 'unlike.png']

iface = gr.Interface(fn=classify_image, inputs=gr_image, outputs=label, examples=examples)
iface.launch(inline=False)