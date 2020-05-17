import os
import json
import io

import responder
import numpy as np

from image2vec import load_img, Image2Vec


env = os.environ
DEBUG = env['DEBUG'] in ['1', 'True', 'true']
MODEL = env['MODEL']
INTERPOLATION = env['INTERPOLATION']
IMAGE_NET_TARGET_SIZE = (224, 224)

api = responder.API(debug=DEBUG)
image2vec = Image2Vec(MODEL)


def extract_features(bytes_io, interpolation=INTERPOLATION):
    img = load_img(
        bytes_io,
        target_size=IMAGE_NET_TARGET_SIZE,
        interpolation=interpolation
    )
    features = image2vec.extract_features(img)
    return features.tolist()


@api.route("/")
async def encode(req, resp):
    body = await req.content
    features = extract_features(io.BytesIO(body))
    resp_dict = dict(data=features, dim=len(features))
    resp.media = resp_dict


if __name__ == "__main__":
    api.run()