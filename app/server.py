from starlette.applications import Starlette
from starlette.responses import HTMLResponse, JSONResponse
from starlette.staticfiles import StaticFiles
from starlette.middleware.cors import CORSMiddleware
import uvicorn, aiohttp, asyncio
from io import BytesIO

from fastai import *
from fastai.vision import *

export_file_url = 'https://drive.google.com/uc?export=download&id=1C4-599sntLTKiE7gHubUBJuyEz9upRCr'
export_file_name = 'export.pkl'

classes = ['alpine sea holly',
 'anthurium',
 'artichoke',
 'azalea',
 'ball moss',
 'balloon flower',
 'barbeton daisy',
 'bearded iris',
 'bee balm',
 'bird of paradise',
 'bishop of llandaff',
 'black-eyed susan',
 'blackberry lily',
 'blanket flower',
 'bolero deep blue',
 'bougainvillea',
 'bromelia',
 'buttercup',
 'californian poppy',
 'camellia',
 'canna lily',
 'canterbury bells',
 'cape flower',
 'carnation',
 'cautleya spicata',
 'clematis',
 "colt's foot",
 'columbine',
 'common dandelion',
 'corn poppy',
 'cyclamen ',
 'daffodil',
 'desert-rose',
 'english marigold',
 'fire lily',
 'foxglove',
 'frangipani',
 'fritillary',
 'garden phlox',
 'gaura',
 'gazania',
 'geranium',
 'giant white arum lily',
 'globe thistle',
 'globe-flower',
 'grape hyacinth',
 'great masterwort',
 'hard-leaved pocket orchid',
 'hibiscus',
 'hippeastrum ',
 'japanese anemone',
 'king protea',
 'lenten rose',
 'lotus',
 'love in the mist',
 'magnolia',
 'mallow',
 'marigold',
 'mexican aster',
 'mexican petunia',
 'monkshood',
 'moon orchid',
 'morning glory',
 'orange dahlia',
 'osteospermum',
 'oxeye daisy',
 'passion flower',
 'pelargonium',
 'peruvian lily',
 'petunia',
 'pincushion flower',
 'pink primrose',
 'pink-yellow dahlia?',
 'poinsettia',
 'primula',
 'prince of wales feathers',
 'purple coneflower',
 'red ginger',
 'rose',
 'ruby-lipped cattleya',
 'siam tulip',
 'silverbush',
 'snapdragon',
 'spear thistle',
 'spring crocus',
 'stemless gentian',
 'sunflower',
 'sweet pea',
 'sweet william',
 'sword lily',
 'thorn apple',
 'tiger lily',
 'toad lily',
 'tree mallow',
 'tree poppy',
 'trumpet creeper',
 'wallflower',
 'water lily',
 'watercress',
 'wild pansy',
 'windflower',
 'yellow iris']
path = Path(__file__).parent

app = Starlette()
app.add_middleware(CORSMiddleware, allow_origins=['*'], allow_headers=['X-Requested-With', 'Content-Type'])
app.mount('/static', StaticFiles(directory='app/static'))

async def download_file(url, dest):
    if dest.exists(): return
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            data = await response.read()
            with open(dest, 'wb') as f: f.write(data)

async def setup_learner():
    await download_file(export_file_url, path/export_file_name)
    try:
        learn = load_learner(path, export_file_name)
        return learn
    except RuntimeError as e:
        if len(e.args) > 0 and 'CPU-only machine' in e.args[0]:
            print(e)
            message = "\n\nThis model was trained with an old version of fastai and will not work in a CPU environment.\n\nPlease update the fastai library in your training environment and export your model again.\n\nSee instructions for 'Returning to work' at https://course.fast.ai."
            raise RuntimeError(message)
        else:
            raise

loop = asyncio.get_event_loop()
tasks = [asyncio.ensure_future(setup_learner())]
learn = loop.run_until_complete(asyncio.gather(*tasks))[0]
loop.close()

@app.route('/')
def index(request):
    html = path/'view'/'index.html'
    return HTMLResponse(html.open().read())

@app.route('/analyze', methods=['POST'])
async def analyze(request):
    data = await request.form()
    img_bytes = await (data['file'].read())
    img = open_image(BytesIO(img_bytes))
    prediction = learn.predict(img)[0]
    return JSONResponse({'result': str(prediction)})

if __name__ == '__main__':
    if 'serve' in sys.argv: uvicorn.run(app=app, host='0.0.0.0', port=5042)
