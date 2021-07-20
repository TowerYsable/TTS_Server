import os,sys,io
from pathlib import Path
from flask import Flask,render_template,send_file,request
from mtts.sysnthesize_server import syn_server

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/tts',methods=['GET'])
def tts():
    text = request.args.get('text')
    print('model inpur:{}'.format(text))
    final_wav_path = syn_server(text)
    out = io.BytesIO()
    return send_file(final_wav_path,mimetype='audio/wav')

def main():
    app.run(debug=False, host='0.0.0.0', port=1999)

if __name__ == '__main__':
    main()
