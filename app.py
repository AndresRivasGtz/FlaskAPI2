from flask import Flask, request

app = Flask(__name__)

@app.route('/lector')
def lector():
    return 'Este es el lector'

if __name__ == '__main__':
    app.run(debug = True, port=5000)