"""Flask application factory."""
from flask import Flask


def create_app():
    app = Flask(__name__)
    app.config["SECRET_KEY"] = "dev-spirometry-interpreter-change-in-production"

    from app.routes import bp

    app.register_blueprint(bp)

    return app
