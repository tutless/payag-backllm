from flask import jsonify
from flask_smorest import Blueprint
from flask.views import MethodView
from schema import PlainQuerySchema
from payag_llm.llm_core import LLMCore


blp = Blueprint("Chat", __name__, description="Chat Operation")


@blp.route("/chat")
class Chat(MethodView):

    @blp.response(200)
    def get(self):
        return jsonify({"message": "reponse from chat"})

    @blp.response(201)
    @blp.arguments(PlainQuerySchema)
    def post(self, item):
        llm_core = LLMCore(item["query"])
        llm_answer = llm_core.run_llm()

        return {"answer": llm_answer}
