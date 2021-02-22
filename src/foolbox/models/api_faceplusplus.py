import numpy as np
from .base import Model
import base64
from struct import unpack
import os
from PIL import Image
import requests
import base64
import json
import struct


class FacePlusPlusModel(Model):
    def __init__(self, bounds, src_img_path, suffix, account=0, channel_axis=2, simi_threshold=0.5):
        super(FacePlusPlusModel, self).__init__(bounds=bounds, channel_axis=channel_axis)
        self.src_img_path = src_img_path
        # the minimum confidence score needed to consider two imgs as similar;
        # the larger this score is, the stricter the criteria for two imgs to be similar.
        self.simi_threshold = simi_threshold
        self.suffix = suffix
        self.account = account

        if self.account == 0:
            # ilar
            self.key = "YOUR_KEY"
            self.secret = "YOUR_SECRET"
            self.http_url = 'https://api-us.faceplusplus.com/facepp/v3/compare'
        elif self.account == 1:
            self.key = "YOUR_KEY"
            self.secret = "YOUR_SECRET"
            self.http_url = 'https://api-us.faceplusplus.com/facepp/v3/compare'
        else:
            print("No account available")
            assert 0

    def forward(self, inputs):
        assert len(inputs.shape) == 4
        preds = []
        for _i in range(inputs.shape[0]):
            # first write input imgs to disk, then query
            x = inputs[_i, :].transpose(1, 2, 0) * 255
            assert x.shape[2] == 3
            image = Image.fromarray(x.astype('uint8'), 'RGB')
            qry_img_path = './tmp_%d_%s.jpg' %(self.account, self.suffix)
            image.save(qry_img_path)

            _img_fr = open(self.src_img_path, 'rb')
            tgt_img = base64.b64encode(_img_fr.read())
            qry_img_fr = open(qry_img_path, 'rb')
            qry_img = base64.b64encode(qry_img_fr.read())
            # qry_img = base64.encodebytes(struct.pack('<d', x)) # does not work since x is not a float value
            # qry_img = base64.b64encode(x)

            payload = {
                'api_key': self.key,
                'api_secret': self.secret,
                'image_base64_1': tgt_img,
                'image_base64_2': qry_img
            }
            _img_fr.close()
            qry_img_fr.close()
            try:
                res = requests.post(self.http_url, data=payload)
                res_json = json.loads(res.text)
                score = res_json['confidence']/100
                flag = int(score < self.simi_threshold)
                preds.append((flag, 1-flag))
            except Exception as e:
                # print('Error:')
                # print(e)
                preds.append((1, 0)) # if error, consider two imgs not similar
        return np.array(preds)

    def num_classes(self):
        return 2

    # Dummy
    def gradient_one(self, *args, **kwargs):
        return 0.0

