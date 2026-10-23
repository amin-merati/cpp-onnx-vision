"""Convert a local JPEG/PNG to the runner's deliberately simple RGB PPM format."""
import argparse
from PIL import Image, ImageOps

parser = argparse.ArgumentParser()
parser.add_argument('input')
parser.add_argument('output')
args = parser.parse_args()
with Image.open(args.input) as image:
    ImageOps.exif_transpose(image).convert('RGB').save(args.output, format='PPM')
