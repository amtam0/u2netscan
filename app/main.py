import io
import os
import sys
from flask import Flask, request, send_file, jsonify
from flask_cors import CORS
from PIL import Image
import numpy as np
import time
import logging
import u2net_test
import json
import base64
import shutil
import subprocess
import re
import pytesseract
from deskew import determine_skew


def PIL_thresh(IMG,threshold = 191):
    """https://stackoverflow.com/a/6490819"""
    return IMG.point(lambda p: p > threshold and 255)

def compose_mask(real_path=None,mask_path=None):
    """
    show content and blacken background
    """
    real_gr = Image.open(real_path).convert(mode='L')
    mask_weak_gr = Image.open(mask_path).convert(mode='L')

    white_backgrd = Image.new("L", real_gr.size, 255)
    black_backgrd = Image.new("L", real_gr.size, 0)

    thresh_mask = PIL_thresh(mask_weak_gr)

    white_bgrd = Image.composite(real_gr,white_backgrd, thresh_mask)
    black_bgrd = Image.composite(real_gr,black_backgrd, thresh_mask)
    white_bgrd.save(os.path.join(os.path.dirname(mask_path),"white.png" ))

    return white_bgrd,black_bgrd

def basic_handler(event=None,MODEL_NAME = "u2net"):
    """
    """
    #json_keys
    body_image64 = event['body64'].encode("utf-8")
    
    if os.path.isdir("/tmp/in_data/"):
        shutil.rmtree("/tmp/in_data/")
    if os.path.isdir("/tmp/out_data/"):
        shutil.rmtree("/tmp/out_data/")
    os.makedirs("/tmp/in_data/")
    os.makedirs("/tmp/out_data/")
    img_path = "/tmp/in_data/in_img.png"
    
    #print("Decode & save inp image to /tmp")
    start = time.time()
    with open(img_path, "wb") as f:
        f.write(base64.b64decode(body_image64))
    logging.info(f'Decode & save inp image to /tmp {time.time() - start:.2f}s')

    #print("run model")
    start = time.time()
    u2net_test.main(model_name=MODEL_NAME,
                       image_dir='/tmp/in_data/',
                       prediction_dir='/tmp/out_data/',)
    logging.info(f'run model {time.time() - start:.2f}s')

    #print("read output")
    start = time.time()
    Mask_path = "/tmp/out_data/in_img.png"
    Real_path = "/tmp/in_data/in_img.png"
    white_bgrd,black_bgrd = compose_mask(real_path=Real_path,mask_path=Mask_path)
    logging.info(f'read output {time.time() - start:.2f}s')
    
    #print("resize image")
    start = time.time()
    file_size_MB = round(os.path.getsize(os.path.join(os.path.dirname(Mask_path), "white.png"))/(1024*1024),2)
    logging.info(f"file_size_MB : {file_size_MB}")
    logging.info(f'resize image  {time.time() - start:.2f}s')

    #print("TEXTCLEANER")
    start = time.time()
    local_img_in = "/tmp/out_data/white_bgrd.png"
    local_img_out = "/tmp/out_data/white_bgrd_cln.png"
    local_img_deskew = "/tmp/out_data/white_bgrd_dsk.png"
    shell_path = "."
    config = "-g -e normalize -f 10 -o 6 -u -s 1 -T -p 10"
    white_bgrd.save(local_img_in)
    cleantext_CMD = """bash "{}/textcleaner.sh" {} "{}" {}""".format(shell_path, config, local_img_in, local_img_out)
    # Process cleaned image
    subprocess_call = subprocess.call([cleantext_CMD], shell=True)
    white_bgrd_cln = Image.open(local_img_out)
    logging.info(f'TEXTCLEANER {time.time() - start:.2f}s')
    
    #print("Deskew")
    start = time.time()
    angle = determine_skew(np.array(white_bgrd_cln))
    logging.info(f"Deskew angle : {angle}")
    white_bgrd_cln = white_bgrd_cln.rotate(angle, expand=1, fillcolor = "#ffffff")
    logging.info(f'DESKEW {time.time() - start:.2f}s')
    
    #print("Tesseract OSD orient")
    start = time.time()
    #TESSERACT FIND ORIENTATION
    # osd = pytesseract.image_to_osd(white_bgrd_cln)
    # angle = re.search('(?<=Rotate: )\d+', osd).group(0)
    try: #sometimes Tesseract return error when image is not identified
        angle = pytesseract.image_to_osd(white_bgrd_cln, output_type="dict", config="-l eng")["orientation"]
    except:
        angle = 0
    logging.info(f"Tesseract angle : {angle}")
    try:
        if float(angle)!=0:
            # white_bgrd_cln = white_bgrd_cln.rotate(360-float(angle),expand=True)
            white_bgrd_cln = white_bgrd_cln.rotate(float(angle),expand=True)
    except:
        pass
    logging.info(f'Tesseract OSD orient {time.time() - start:.2f}s')
    
    return white_bgrd_cln


logging.basicConfig(level=logging.INFO)

# Initialize the Flask application
app = Flask(__name__)
CORS(app)

# Simple probe.
@app.route('/', methods=['GET'])
def hello():
    return 'Hello U2-NET!'

# Route http posts to this method
@app.route('/', methods=['POST'])
def run():
    start = time.time()

    data = request.json

    res = basic_handler(event=data, MODEL_NAME = str(data["model_name"]))

    # Save to buffer
    buff = io.BytesIO()
    res.save(buff, 'PNG')
    # buff.seek(0)
    
    data["body64"] = base64.b64encode(buff.getvalue()).decode("utf-8")

    # #print stats
    logging.info(f'Total time {time.time() - start:.2f}s')

    # Return data
    return jsonify(data)#send_file(buff, mimetype='image/png')


if __name__ == '__main__':
    os.environ['FLASK_ENV'] = 'development'
    port = int(os.environ.get('PORT', 8080))
    app.run(debug=True, host='0.0.0.0', port=port)
