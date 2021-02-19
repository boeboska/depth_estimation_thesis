import os
import torch
from PIL import Image
import torchvision.transforms as T
from tqdm import tqdm
import argparse


def main(args):
    model, postprocessor = torch.hub.load('facebookresearch/detr', 'detr_resnet101_panoptic', pretrained=True,
                                          return_postprocessor=True, num_classes=250)
    model.eval()

    device = "cuda" if torch.cuda.is_available() else 'cpu'
    model.to(device)

    # 240 voor 50 detr
    # 220 voor 101 detr
    transform = T.Compose([
        T.Resize(220),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    paths = os.walk(args.kitti_path)

    for i, (path, directories, files) in tqdm(enumerate(paths)):

        if 'image_00' in path or 'image_01' in path:
            continue
        if len(files) > 0 and files[0].endswith('.png'):
            for file in files:
                img_path = os.path.join(path, file)

                attention_maps_path = img_path[:-4].replace('kitti' + os.path.sep,
                                                            'attention_masks_hidde' + os.path.sep)

                if os.path.exists(attention_maps_path):
                    continue

                sub_paths = attention_maps_path.split(os.path.sep)

                for i in range(len(sub_paths)):
                    sub_path = os.path.sep.join(sub_paths[:i + 1])

                    if not os.path.exists(sub_path):
                        os.mkdir(sub_path)

                img = Image.open(img_path)

                img = transform(img).unsqueeze(0).to(device)
                out = model(img)

                scores = out["pred_logits"].softmax(-1)[..., :-1].max(-1)[0].squeeze()
                out['pred_masks'] = out['pred_masks'].squeeze()

                for index, (score, mask) in enumerate(zip(scores, out['pred_masks'])):

                    min, max = mask.min(), mask.max()

                    if min < 0:
                        mask = mask - min
                    min, max = mask.min(), mask.max()
                    mask = (mask / max)

                    mask = T.ToPILImage()(mask)
                    img = mask.resize((640, 192))
                    img.save(os.path.join(attention_maps_path, '{}_{}.jpg'.format(index, round(score.item(), 3))))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--kitti_path",
                        type=str,
                        help="path to the kitti dataset folder")
    arguments = parser.parse_args()
    main(arguments)
