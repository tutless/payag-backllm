from payag_llm.llm_core import LLMCore
from flask_smorest import Api
from flask import Flask
from api.chat import blp as ChatBlueprint
from api.api_tester import blp as ApiTesterBlueprint


class MainApp:
    def __init__(self):
        self.app = Flask(__name__)
        self.app.config["PROPAGATE_EXCEPTIONS"] = True
        self.app.config["API_TITLE"] = "Payag Chatbot"
        self.app.config["API_VERSION"] = "v1"
        self.app.config["OPENAPI_VERSION"] = "3.0.3"
        self.app.config["OPENAPI_URL_PREFIX"] = "/"
        self.app.config["OPENAPI_SWAGGER_UI_PATH"] = "/swagger-ui"
        self.app.config["OPENAPI_SWAGGER_UI_URL"] = (
            "https://cdn.jsdelivr.net/npm/swagger-ui-dist"
        )
        self.registered_blueprints()

    def registered_blueprints(self):
        api = Api(self.app)
        api.register_blueprint(ChatBlueprint)
        api.register_blueprint(ApiTesterBlueprint)

    @classmethod
    def run(cls):
        cls().app.run(debug=True, port=5888, host="0.0.0.0")


if __name__ == "__main__":
    MainApp.run()
