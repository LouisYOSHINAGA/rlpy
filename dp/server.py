import os
import tornado.web
import tornado.escape
import tornado.ioloop
from tornado.options import define, parse_command_line
from environment import Environment, Prob
from planner import StateValueList2d, Planner, ValueIterationPlanner, PolicyIterationPlanner
from typing import Any

define("port", default=8888, help="run on the given port", type=int)


class IndexHandler(tornado.web.RequestHandler):
    def get(self) -> None:
        self.render("index.html")


class PlanningHandler(tornado.web.RequestHandler):
    def post(self, move_prob: Prob =0.8) -> None:
        data = tornado.escape.json_decode(self.request.body)
        try:
            move_prob = float(data["prob"])
        except ValueError:
            pass

        env = Environment(data["grid"], move_prob=move_prob)
        if data["plan"] == "value":
            planner: Planner = ValueIterationPlanner(env)
        elif data["policy"] == "policy":
            planner = PolicyIterationPlanner(env)
        result: StateValueList2d = planner.plan()
        planner.log.append(result)
        self.write({'log': planner.log})


class Application(tornado.web.Application):
    def __init__(self) -> None:
        handlers: list[tuple[str, Any]] = [
            (r"/", IndexHandler),
            (r"/plan", PlanningHandler),
        ]
        settings: dict[str, Any] = dict(
            template_path=os.path.join(os.path.dirname(__file__), "templates"),
            static_path=os.path.join(os.path.dirname(__file__), "static"),
            cookie_secret=os.environ.get("SECRET_TOKEN", "__TODO:_GENERATE_YOUR_OWN_RANDOM_VALUE_HERE__"),
            debug=True,
        )
        super(Application, self).__init__(handlers, **settings)  # type: ignore


def main() -> None:
    parse_command_line()
    app = Application()
    port: int = int(os.environ.get("PORT", 8888))
    app.listen(port)
    print(f"Run server on port: {port}")
    tornado.ioloop.IOLoop.current().start()


if __name__ == "__main__":
    main()