from flask import Flask

app = Flask(__name__)

@app.route('/')
def hello():
    return "<h1>Hello! Flask is working!</h1><p>If you see this, the server is running correctly.</p>"

if __name__ == '__main__':
    print("Starting simple Flask app...")
    print("Try opening: http://127.0.0.1:5001")
    app.run(debug=True, host='127.0.0.1', port=5001)