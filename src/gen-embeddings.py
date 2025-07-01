import torch
from thingsvision import get_extractor
from thingsvision.utils.storing import save_features
from thingsvision.utils.data import ImageDataset, DataLoader

model_name = "clip"
source = "custom"
device = "cuda" if torch.cuda.is_available() else "cpu"
model_parameters = {"variant": "RN50"}
batch_size = 32

extractor = get_extractor(
    model_name=model_name,
    source=source,
    device=device,
    pretrained=True,
    model_parameters=model_parameters,
)

dataset = ImageDataset(
    root="/Users/emdupre/Desktop/things-encode/cneuromod.things.stimuli/",
    out_path="/Users/emdupre/Desktop/things-encode/clip-features/",
    backend=extractor.get_backend(),
    transforms=extractor.get_transformations(),
)

batches = DataLoader(
    dataset=dataset,
    batch_size=batch_size,
    backend=extractor.get_backend(),  # backend framework of model
)

module_name = "visual"

features = extractor.extract_features(
    batches=batches,
    module_name=module_name,
    flatten_acts=True,  # flatten 2D feature maps from an early convolutional or attention layer
    output_type="ndarray",  # or "tensor" (only applicable to PyTorch models of which CLIP is one!)
)

# file_format can be set to "npy", "txt", "mat", "pt", or "hdf5"
save_features(
    features,
    out_path="/Users/emdupre/Desktop/things-encode/clip-features/",
    file_format="npy",
)
