from fastai.vision import load_learner, Path, BytesIO, open_image
from starlette.requests import Request
from starlette.responses import Response
from starlette.applications import Starlette
from starlette.responses import JSONResponse, FileResponse
import uvicorn
import numpy as np


learn = load_learner(Path("."))
app = Starlette(debug=True)


@app.route("/check", methods=["POST"])
async def upload(request):
    data = await request.form()
    bytes = await (data["file"].read())
    return prediction(bytes)


def prediction(bytes):
    try:
        img = open_image(BytesIO(bytes))
    except:
        return JSONResponse({"error": "image error"})
    _, _, losses = learn.predict(img)
    return JSONResponse({"v":
                         sorted(zip(learn.data.classes, map(
                             float, losses)), key=lambda p: p[1], reverse=True)
                         })


@app.route("/")
async def home(request):
    return FileResponse("index.html")

# if __name__ == "__main__":
#    uvicorn.run(app, host='0.0.0.0', port=8000)
