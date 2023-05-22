from models import fasterrcnn_resnet50_fpn_v2


def return_fasterrcnn_resnet50_fpn_v2(
    num_classes, pretrained=True, coco_model=False
):
    model = fasterrcnn_resnet50_fpn_v2.create_model(
        num_classes, pretrained=pretrained, coco_model=coco_model
    )
    return model


create_model = {
    'fasterrcnn_resnet50_fpn_v2': return_fasterrcnn_resnet50_fpn_v2,
}