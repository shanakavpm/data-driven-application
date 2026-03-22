"""Launch the Flask web app."""

from webapp import create_app

app = create_app()

if __name__ == "__main__":
    print("\n  Starting FraudGuard at http://localhost:5000\n")
    app.run(debug=True, host="0.0.0.0", port=5000)
