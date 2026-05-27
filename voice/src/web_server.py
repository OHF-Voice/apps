import logging
import threading

from flask import Flask, render_template, url_for
from werkzeug.middleware.proxy_fix import ProxyFix

from const import BASE_DIR, AppState


def make_web_server(state: AppState) -> Flask:
    flask_app = Flask(__name__, template_folder=str(BASE_DIR / "templates"))
    flask_app.secret_key = "145ec815-48f0-450a-abcc-374f726655c9"

    flask_app.wsgi_app = ProxyFix(flask_app.wsgi_app, x_proto=1, x_host=1)  # type: ignore[method-assign]
    flask_app.wsgi_app = IngressPrefixMiddleware(flask_app.wsgi_app)  # type: ignore[method-assign]

    @flask_app.context_processor
    def inject_url_for():
        return dict(url_for=url_for)  # pylint: disable=use-dict-literal

    @flask_app.route("/", methods=["GET"])
    def index():
        return render_template("index.html", state=state)

    @flask_app.route("/health")
    def health():
        return {"status": "ok"}, 200

    return flask_app


def run_web_server(state: AppState, flask_app: Flask) -> threading.Thread:
    def run_flask():
        logging.getLogger("werkzeug").setLevel(logging.ERROR)
        flask_app.run(host=state.http_host, port=state.http_port, use_reloader=False)

    flask_thread = threading.Thread(target=run_flask, daemon=True)

    return flask_thread


class IngressPrefixMiddleware:
    """Ingress fix for Home Assistant app web UI."""

    def __init__(self, app):
        self.app = app

    def __call__(self, environ, start_response):
        ingress_path = environ.get("HTTP_X_INGRESS_PATH", "")
        if ingress_path:
            environ["SCRIPT_NAME"] = ingress_path
            path_info = environ.get("PATH_INFO", "")
            if path_info.startswith(ingress_path):
                environ["PATH_INFO"] = path_info[len(ingress_path) :] or "/"
        return self.app(environ, start_response)
