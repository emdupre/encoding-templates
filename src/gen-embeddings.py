import click
import torch
from thingsvision import get_extractor
from thingsvision.utils.storing import save_features
from thingsvision.utils.data import ImageDataset, DataLoader


@click.command()
@click.option(
    "--input_dir",
    default="./things.stimuli/",
    help="Input directory with stimuli.",
)
@click.option(
    "--output_dir",
    default="./things.stimuli/stimuli.clip-features/",
    help="Output directory for CLIP embeddings",
)
def main(input_dir, output_dir):
    """
    Generate CLIP embeddings of provided stimuli
    """
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
        root=input_dir,
        out_path=output_dir,
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
        out_path=output_dir,
        file_format="npy",
    )
