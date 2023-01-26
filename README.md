# transformers-collection
- A collection of transformer models built  using huggingface for various tasks. Training done using pytorch lightning.
- Datasets, models and tokenizers from hugging face.
- **Goal**: Get familiar with huggingface and pytorch lightning ecosystems.

## Get started
### Train Models using the library
- To train models, install using pip: `pip install transformers_collection`
- check installation: `transformers-collection version`

### Clone project and modify code
To play around with the code clone the repo:
- `git clone git@github.com:aadhithya/transformers-collection.git`
- Install poetry: `pip install poetry`
- Intsall dependencies: `poetry install`

**Note:** `poetry install` will create a new venv.
**Note**: `poetry/pip install` installs CPU version of pytorch if not available, please make sure to install CUDA version if needed.


## Train a model
- Create the yaml config file for the model (see configs/sentiment-clf.yml for example).
- train model using: `transformers-collection train /path/to/config.yml`

- For a list of supported models, see section Supported Models.



## Supported Models / Task
The following models are planned:
| Model                            |                      Dataset                       |  Status   | Checkpoint |
| :------------------------------- | :------------------------------------------------: | :-------: | ---------: |
| Sentiment/Emotion Classification | [emotion](https://huggingface.co/datasets/emotion) |     ✅     |        TBD |
| Text Summarization               |                                                    | 🗓️ Planned |        TBD |
