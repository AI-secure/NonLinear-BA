import numpy as np
from .base import Model
import os
from io import BytesIO
from PIL import Image
import time


class AzureDetectModel(Model):
    def __init__(self, bounds, src_img_detect, suffix, channel_axis=2):
        super(AzureDetectModel, self).__init__(bounds=bounds, channel_axis=channel_axis)

        from azure.cognitiveservices.vision.face import FaceClient
        from msrest.authentication import CognitiveServicesCredentials

        KEY = "YOUR_KEY"
        ENDPOINT = "YOUR_ENDPOINT"

        self.face_client = FaceClient(ENDPOINT, CognitiveServicesCredentials(KEY))

        self.src_img_detect = src_img_detect
        self.suffix = suffix

    def forward(self, inputs):
        assert len(inputs.shape) == 4
        preds = []
        for _i in range(inputs.shape[0]):
            x = inputs[_i, :]
            image = Image.fromarray(x.astype('uint8'), 'RGB')
            qry_img_path = './tmp_%s.jpg' %(self.suffix)
            image.save(qry_img_path)
            qry_img_fr = open(qry_img_path, 'rb')

            try:
                qry_faces = self.face_client.face.detect_with_stream(image=qry_img_fr, return_face_id=True,
                                                                 return_face_landmarks=False,
                                                                 return_face_attributes=None,
                                                                 recognition_model='recognition_01',
                                                                 return_recognition_model=False,
                                                                 detection_model='detection_01',
                                                                 custom_headers=None, raw=False, callback=None)
                qry_detect = bool(qry_faces)
                flag = (qry_detect == self.src_img_detect)
                preds.append((1-flag, flag))
            except Exception as e:
                preds.append((1, 0))
        return np.array(preds)

    def num_classes(self):
        return 2

    # Dummy
    def gradient_one(self, *args, **kwargs):
        return 0.0