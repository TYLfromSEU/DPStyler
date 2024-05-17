import os
import os.path as osp
from ..build import DATASET_REGISTRY
from ..base_dataset import Datum, DatasetBase, Datum_sf


@DATASET_REGISTRY.register()
class TerraIncognita_SF(DatasetBase):
    """Terra-Incognita.

    Statistics:
        - Around 24,330 images.

    """

    dataset_dir = "terra_incognita"
    domains = ["none", "location_38", "location_43", "location_46", "location_100"]

    def __init__(self, cfg, train_data):
        root = osp.abspath(osp.expanduser(cfg.DATASET.ROOT))
        self.dataset_dir = osp.join(root, self.dataset_dir)

        assert osp.exists(self.dataset_dir), f"{self.dataset_dir} dataset is not exist"

        self.check_input_domains(
            cfg.DATASET.SOURCE_DOMAINS, cfg.DATASET.TARGET_DOMAINS
        )

        test_datasets = []

        train = self._read_train_data(train_data)

        for domain in cfg.DATASET.TARGET_DOMAINS:
            test_datasets.append(self._read_data(self.dataset_dir, domain, train_data["classnames"]))

        super().__init__(train_x=train, test=test_datasets)

    def _read_data(self, dataset_dir, input_domain, classnames):
        items = []
        domain_dir_path = osp.join(dataset_dir, input_domain)
        for label_id, class_name in enumerate(classnames):
            domain_class_dir_path = osp.join(domain_dir_path, class_name)
            for img_name in os.listdir(domain_class_dir_path):
                img_path = osp.join(domain_class_dir_path, img_name)
                item = Datum(
                    impath=img_path,
                    label=label_id,
                    domain=input_domain,
                    classname=class_name
                )
                items.append(item)

        return items

    def _read_train_data(self, train_data):
        items = []
        classnames = train_data["classnames"]
        n_cls = train_data["n_cls"]
        n_style = train_data["n_style"]
        for idx_cls in range(n_cls):
            for idx_style in range(n_style):
                item = Datum_sf(
                    cls=idx_cls,
                    style=idx_style,
                    label=idx_cls,
                    classname=classnames[idx_cls]
                )
                items.append(item)
        return items
