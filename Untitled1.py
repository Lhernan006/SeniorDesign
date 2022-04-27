#!/usr/bin/env python
# coding: utf-8

# In[20]:


from flask import Flask,request,send_from_directory,render_template
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import cv2
import numpy as np
from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw
from Music21 import *
us = environment.UserSettings()
us['musicxmlPath'] = 'C:\\Program Files\\MuseScore 3\\bin\\MuseScore3.exe'
us['musescoreDirectPNGPath'] = 'C:\\Program Files\\MuseScore 3\\bin\\MuseScore3.exe'
us['musicxmlPath']
app = Flask(__name__, static_url_path='')
def sparse_tensor_to_strs(sparse_tensor):
    indices = sparse_tensor[0][0]
values = sparse_tensor[0][1]
dense_shape = sparse_tensor[0][2]
strs = [ [] for i in range(dense_shape[0]) ]
string = []
ptr = 0
b = 0
for idx in range(len(indices)):
    if indices[idx][0] != b:
        strs[b] = string
        string = []
        b = indices[idx][0]
    string.append(values[ptr])
    ptr = ptr + 1
strs[b] = string


return strs

def normalize(image):
    return (255. - image)/255.

def resize(image, height):
    width = int(float(height * image.shape[1]) / image.shape[0])
    sample_img = cv2.resize(image, (width, height))
    return sample_img

voc_file = "vocabulary_semantic.txt"
model = "Semantic-Model/semantic_model.meta"
tf.reset_default_graph()
sess = tf.InteractiveSession()
# Read the dictionary
dict_file = open(voc_file,'r')
dict_list = dict_file.read().splitlines()
int2word = dict()
for word in dict_list:
    word_idx = len(int2word)
    int2word[word_idx] = word
dict_file.close()
# Restore weights
saver = tf.train.import_meta_graph(model)
saver.restore(sess,model[:-5])
graph = tf.get_default_graph()
input = graph.get_tensor_by_name("model_input:0")
seq_len = graph.get_tensor_by_name("seq_lengths:0")
rnn_keep_prob = graph.get_tensor_by_name("keep_prob:0")
height_tensor = graph.get_tensor_by_name("input_height:0")
width_reduction_tensor = graph.get_tensor_by_name("width_reduction:0")
logits = tf.get_collection("logits")[0]
# Constants that are saved inside the model itself
WIDTH_REDUCTION, HEIGHT = sess.run([width_reduction_tensor, height_tensor])
decoded, _ = tf.nn.ctc_greedy_decoder(logits, seq_len)
@app.route('/img/<filename>')
def send_img(filename):



    return send_from_directory('', filename)
@app.route("/")
def root():
    return render_template('index.html')
@app.route('/predict', methods = ['GET', 'POST'])
def predict():
    if request.method == 'POST':
        f = request.files['file']
    img = f
    image = Image.open(img).convert('L')
    image = np.array(image)
    image = resize(image, HEIGHT)
    image = normalize(image)
    image = np.asarray(image).reshape(1,image.shape[0],image.shape[1],1)
    seq_lengths = [ image.shape[2] / WIDTH_REDUCTION ]
    prediction = sess.run(decoded,
                feed_dict={
                    input: image,
                    seq_len: seq_lengths,
                    rnn_keep_prob: 1.0,
                })

    str_predictions = sparse_tensor_to_strs(prediction)
    array_of_notes = []
    for w in str_predictions[0]:
        array_of_notes.append(int2word[w])
file_header = '\\version "2.22.1"\n {\n \\autoBeamOff \n'
with open('score-read.txt','w') as f:
    for score in array_of_notes:
        f.write(score)
        f.write('\n')
with open('score.ly','w') as f:
    f.write(file_header)
    for score in array_of_notes:
        if (score =="clef-G1"):
            f.write('\t\clef "treble"\n')
        if (score =="clef-G2"):
            f.write('\t\clef "treble"\n')
        if (score =="clef-F3"):
            f.write('\t\clef "bass"\n')
        if (score =="clef-F4"):
            f.write('\t\clef "bass"\n')
        if (score =="clef-F5"):
            f.write('\t\clef "bass"\n')
        if (score =="clef-C1"):
            f.write('\t\clef "alto"\n')
        if (score =="clef-C2"):
            f.write('\t\clef "alto"\n')
        if (score =="clef-C3"):
            f.write('\t\clef "alto"\n')
        if (score =="clef-C4"):
            f.write('\t\clef "tenor"\n')
        if (score =="clef-C5"):
            f.write('\t\clef "tenor"\n')
        if (score =="keySignature-AbM"):
            f.write('\t\key a \major\n')
        if (score =="keySignature-AM"):
            f.write('\t\key a \minor\n')
        if (score =="keySignature-BbM"):
            f.write('\t\key b \major\n')
        if (score =="keySignature-BM"):
            f.write('\t\key b \minor\n')
        if (score =="keySignature-CM"):
            f.write('\t\key c \major\n')
        if (score =="keySignature-C#M"):
            f.write('\t\key c \minor\n')
        if (score =="keySignature-DbM"):
            f.write('\t\key d \major\n')
        if (score =="keySignature-DM"):
            f.write('\t\key d \minor\n')
        if (score =="keySignature-EbM"):
            f.write('\t\key e \major\n')
        if (score =="keySignature-EM"):
            f.write('\t\key e \minor\n')
        if (score =="keySignature-FM"):
            f.write('\t\key f \major\n')
        if (score =="keySignature-F#M"):
            f.write('\t\key f \minor\n')
        if (score =="keySignature-GbM"):
            f.write('\t\key g \major\n')
        if (score =="keySignature-GM"):
            f.write('\t\key g \minor\n')
        if (score =="timeSignature-11/4"):
            f.write('\t\\time 11/4\n')
        if (score =="timeSignature-1/2"):
            f.write('\t\\time 1/2\n')



if (score =="timeSignature-12/16"):
    f.write('\t\\time 12/16\n')
    if (score =="timeSignature-12/4"):
        f.write('\t\\time 12/4\n')
    if (score =="timeSignature-12/8"):
        f.write('\t\\time 12/8\n')
    if (score =="timeSignature-1/4"):
        f.write('\t\\time 1/4\n')
    if (score =="timeSignature-2/1"):
        f.write('\t\\time 2/1\n')
    if (score =="timeSignature-2/2"):
        f.write('\t\\time 2/2\n')
    if (score =="timeSignature-2/3"):
        f.write('\t\\time 2/3\n')
    if (score =="timeSignature-2/4"):
        f.write('\t\\time 2/4\n')
    if (score =="timeSignature-24/16"):
        f.write('\t\\time 24/16\n')
    if (score =="timeSignature-2/48"):
        f.write('\t\\time 2/48\n')
    if (score =="timeSignature-2/8"):
        f.write('\t\\time 2/8\n')
    if (score =="timeSignature-3/1"):
        f.write('\t\\time 3/1\n')
    if (score =="timeSignature-3/2"):
        f.write('\t\\time 3/2\n')
    if (score =="timeSignature-3/4"):
        f.write('\t\\time 3/4\n')
    if (score =="timeSignature-3/6"):
        f.write('\t\\time 3/6\n')
    if (score =="timeSignature-3/8"):
        f.write('\t\\time 3/8\n')
    if (score =="timeSignature-4/1"):
        f.write('\t\\time 4/1\n')
    if (score =="timeSignature-4/2"):
        f.write('\t\\time 4/1\n')
    if (score =="timeSignature-4/4"):
        f.write('\t\\time 4/4\n')
    if (score =="timeSignature-4/8"):
        f.write('\t\\time 4/8\n')
    if (score =="timeSignature-5/4"):
        f.write('\t\\time 5/4\n')
    if (score =="timeSignature-5/8"):
        f.write('\t\\time 5/8\n')
    if (score =="timeSignature-6/16"):
        f.write('\t\\time 6/16\n')



if (score =="timeSignature-6/2"):
    f.write('\t\\time 6/2\n')
    if (score =="timeSignature-6/4"):
        f.write('\t\\time 6/4\n')
    if (score =="timeSignature-6/8"):
        f.write('\t\\time 6/8\n')
    if (score =="timeSignature-7/4"):
        f.write('\t\\time 7/4\n')
    if (score =="timeSignature-8/12"):
        f.write('\t\\time 8/12\n')
    if (score =="timeSignature-8/16"):
        f.write('\t\\time 8/16\n')
    if (score =="timeSignature-8/2"):
        f.write('\t\\time 8/2\n')
    if (score =="timeSignature-8/4"):
        f.write('\t\\time 8/4\n')
    if (score =="timeSignature-8/8"):
        f.write('\t\\time 8/8\n')
    if (score =="timeSignature-9/16"):
        f.write('\t\\time 9/16\n')
    if (score =="timeSignature-9/4"):
        f.write('\t\\time 9/4\n')
    if (score =="timeSignature-9/8"):
        f.write('\t\\time 9/8\n')
    # The last 2 are old notation, with C meaning 4/4 and C/ meaning 2/2
    # This is added for historical reasons, as well as to make it correct for old scores
    # without modern notation
    if (score =="timeSignature-C"):
        f.write('\t\\time 4/4\n')
    if (score =="timeSignature-C/"):
        f.write('\t\\time 2/2\n')
    if score[0:5]=="note-":
        notestring = score
        note_octave = ""
        note_modifier = ""
        note_letter = notestring[5:6]
        f.write('\t')
        f.write(note_letter.lower())
    if notestring[6].isdigit():
        note_octave = notestring[6]
    if not notestring[6].isdigit():
        note_octave = notestring[7]
    if score[0:5]=="note-":
        if not score[6].isdigit():
            note_modifier = score[5:7]


else:
    note_modifier = score[5]
    if note_modifier[-1] == 'b':
        f.write('es')
    if note_modifier[-1] == '#':
        f.write('is')
    if note_octave == "1":
        f.write(",,,")
    if note_octave == "2":
        f.write(",,")
    if note_octave == "3":
        f.write(",")
    if note_octave == "4":
        f.write("'")
    if note_octave == "5":
        f.write("\'\'")
    if note_octave == "6":
        f.write("\'\'\'")
    if note_octave == "7":
        f.write("\'\'\'\'")
    if note_octave == "8":
        f.write("\'\'\'\'\'")
        note_length = ""
        uscore_pos = ""
        uscore_pos = notestring.find("_")
        note_length = notestring[uscore_pos+1:]

    if note_length =="quadruple_whole":
        f.write("1*4")
    if note_length =="quadruple_whole.":
        f.write("1*4.")
    if note_length =="quadruple_whole_fermata":
        f.write("1*4\\fermata")
    if note_length =="whole":
        f.write("1")
    if note_length =="whole.":
        f.write("1.")
    if note_length =="whole_fermata":
        f.write("1\\fermata")
    if note_length =="double_whole":
        f.write("1*2")
    if note_length =="double_whole.":
        f.write("1*2.")
    if note_length =="double_whole_fermata":
        f.write("1*2\\fermata")
    if note_length =="eighth":
        f.write("8")
    if note_length =="eighth.":
        f.write("8.")
    if note_length =="eighth..":
        f.write("8..")
    if note_length =="eighth_fermata":
        f.write("8\\fermata")
    if note_length =="half":
        f.write("2")
    if note_length =="half.":
        f.write("2.")
    if note_length =="half_fermata":
        f.write("2\fermata")
    if note_length =="half._fermata":
        f.write("2.\fermata")
    if note_length =="hundred_twenty_eighth":
        f.write("128")
    if note_length =="quarter":
        f.write("4")
    if note_length =="quarter.":
        f.write("4.")
    if note_length =="quarter..":
        f.write("1")
    if note_length =="quarter_fermata":
        f.write("4..")
    if note_length =="quarter._fermata":
        f.write("4.\fermata")
    if note_length =="sixteenth":
        f.write("16")
    if note_length =="sixteenth.":
        f.write("16.")
    if note_length =="sixty_fourth":
        f.write("64")
    if note_length =="thirty_second":
        f.write("32")
    if note_length =="thirty_second.":
        f.write("32.")
    if note_length =="thirty_second_fermata":
        f.write("32\fermata")
        f.write('\n')
    if score =="rest-eighth":
        f.write('r8\n')
    if score =="rest-eighth.":
        f.write('r8.\n')
    if score =="rest-eighth..":
        f.write('r8..\n')
    if score =="rest-eighth_fermata":
        f.write('r8 \\fermata\n')
    if score =="rest-eighth._fermata":
        f.write('r8. \\fermata\n')
    if score =="rest-half":
        f.write('r2\n')
    if score =="rest-half.":
        f.write('r2.\n')
    if score =="rest-half_fermata":
        f.write('r2 \\fermata\n')
    if score =="rest-half._fermata":
        f.write('r2. \\fermata\n')
    if score =="rest-quadruple_whole":
        f.write('r1*4\n')
    if score =="rest-quarter":
        f.write('r4\n')
    if score =="rest-quarter.":
        f.write('r4.\n')
    if score =="rest-quarter..":
        f.write('r4..\n')
    if score =="rest-quarter_fermata":
        f.write('r4 \\fermata\n')
    if score =="rest-quarter._fermata":
        f.write('r4. \\fermata\n')
    if score =="rest-quarter.._fermata":
        f.write('r4.. \\fermata\n')
    if score =="rest-sixteenth":
        f.write('r16\n')
    if score =="rest-sixteenth.":
        f.write('r16.\n')
    if score =="rest-sixteenth_fermata":
        f.write('r16 \\fermata\n')
    if score =="rest-sixty_fourth":
        f.write('r64\n')
    if score =="rest-thirty_second":
        f.write('r32\n')
    if score =="rest-whole":
        f.write('r1\n')
    if score =="rest-whole.":
        f.write('r1.n')
    if score =="rest-whole_fermata":
        f.write('r1 \\fermata\n')
    if score=="tie":
        f.write('~')
    if score[0:10]=="gracenote-":
        notestring = score
        note_octave = ""
        note_modifier = ""
        note_letter = notestring[10:11]
        f.write('\t \grace {')
        f.write(note_letter.lower())
    if notestring[12].isdigit():
        note_octave = notestring[12]
    if not notestring[12].isdigit():
        note_octave = notestring[13]
    if score[0:10]=="gracenote-":
        if not score[11].isdigit():
            note_modifier = score[10:12]
    else:
        note_modifier = score[10]
        if note_modifier[-1] == 'b':
            f.write('es')
        if note_modifier[-1] == '#':
            f.write('is')
        if note_octave == "1":
            f.write(",,,")
        if note_octave == "2":
            f.write(",,")
        if note_octave == "3":
            f.write(",")
        if note_octave == "4":
            f.write("'")
        if note_octave == "5":
            f.write("\'\'")
        if note_octave == "6":
            f.write("\'\'\'")
        if note_octave == "7":
            f.write("\'\'\'\'")
        if note_octave == "8":
            f.write("\'\'\'\'\'")
            note_length = ""
            uscore_pos = ""
            uscore_pos = notestring.find("_")



            note_length = notestring[uscore_pos+1:]

    if note_length =="quadruple_whole":
        f.write("1*4")
    if note_length =="quadruple_whole.":
        f.write("1*4.")
    if note_length =="quadruple_whole_fermata":
        f.write("1*4\\fermata")
    if note_length =="whole":
        f.write("1")
    if note_length =="whole.":
            f.write("1.")
    if note_length =="whole_fermata":
        f.write("1\\fermata")
    if note_length =="double_whole":
        f.write("1*2")
    if note_length =="double_whole.":
        f.write("1*2.")
    if note_length =="double_whole_fermata":
        f.write("1*2\\fermata")
    if note_length =="eighth":
        f.write("8")
    if note_length =="eighth.":
        f.write("8.")
    if note_length =="eighth..":
        f.write("8..")
    if note_length =="eighth_fermata":
        f.write("8\\fermata")
    if note_length =="half":
        f.write("2")
    if note_length =="half.":
        f.write("2.")
    if note_length =="half_fermata":
        f.write("2\fermata")
    if note_length =="half._fermata":
        f.write("2.\fermata")
    if note_length =="hundred_twenty_eighth":
        f.write("128")
    if note_length =="quarter":
        f.write("4")
    if note_length =="quarter.":
        f.write("4.")
    if note_length =="quarter..":
        f.write("1")
    if note_length =="quarter_fermata":
        f.write("4..")
    if note_length =="quarter._fermata":
        f.write("4.\fermata")
    if note_length =="sixteenth":
        f.write("16")
    if note_length =="sixteenth.":
        f.write("16.")
    if note_length =="sixty_fourth":
        f.write("64")
    if note_length =="thirty_second":
        f.write("32")
    if note_length =="thirty_second.":
        f.write("32.")
    if note_length =="thirty_second_fermata":
        f.write("32\fermata")
        f.write('} \n')
        f.write("}\n")

    return render_template('result.html')
    if __name__=="__main__":
        app.run()


# In[ ]:




