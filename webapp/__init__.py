"""Flask app factory."""

import os
from flask import Flask


def create_app():
    base = os.path.dirname(__file__)
    app = Flask(__name__,
                static_folder=os.path.join(base, "static"),
                template_folder=os.path.join(base, "templates"))
    app.config["SECRET_KEY"] = "cmp7005-fraud-detection"

    from webapp.routes import main_bp
    app.register_blueprint(main_bp)
    return app
