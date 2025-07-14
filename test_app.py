from flask import Flask

app = Flask(__name__)

@app.route('/')
def hello():
    return """
    <h1>NCAAB Predictor Test</h1>
    <p>If you see this, Flask is working!</p>
    <p>Next we'll add your prediction functions.</p>
    """

if __name__ == '__main__':
    print("Starting test app...")
    app.run(debug=True, port=8080)