from flask.views import MethodView
from flask_smorest import Blueprint
from schema import PlainTestSchema

blp = Blueprint("Tester", __name__, description="Test Api from accessing device")


@blp.route("/tester")
class ApiTester(MethodView):

    @blp.response(200)
    def get(self):
        return {"message": "test success"}

    @blp.arguments(PlainTestSchema)
    @blp.response(201)
    def post(self, item):
        return {"message": f"your post test is {item["test"]}"}
