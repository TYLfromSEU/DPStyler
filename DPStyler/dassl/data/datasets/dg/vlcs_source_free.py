import glob
import os.path as osp

from dassl.utils import listdir_nohidden

from ..build import DATASET_REGISTRY
from ..base_dataset import Datum, DatasetBase,Datum_sf


@DATASET_REGISTRY.register()
class VLCS_SF(DatasetBase):
    """VLCS.

    Statistics:
        - 4 domains: CALTECH, LABELME, PASCAL, SUN
        - 5 categories: bird, car, chair, dog, and person.

    Reference:
        - Torralba and Efros. Unbiased look at dataset bias. CVPR 2011.
    """

    dataset_dir = "VLCS"
    domains = ["none","CALTECH", "LABELME", "PASCAL", "SUN"]
    data_url = "https://drive.google.com/uc?id=1r0WL5DDqKfSPp9E3tRENwHaXNs1olLZd"

    def __init__(self, cfg,p_g):
        root = osp.abspath(osp.expanduser(cfg.DATASET.ROOT))
        self.dataset_dir = osp.join(root, self.dataset_dir)

        if not osp.exists(self.dataset_dir):
            dst = osp.join(root, "vlcs.zip")
            self.download_data(self.data_url, dst, from_gdrive=True)

        self.check_input_domains(
            cfg.DATASET.SOURCE_DOMAINS, cfg.DATASET.TARGET_DOMAINS
        )

        test_datasets=[]
        for domain in cfg.DATASET.TARGET_DOMAINS:
            test_datasets.append(self._read_data([domain], "test"))
        train=self._read_train_data(p_g)


        super().__init__(train_x=train, test=test_datasets)

    def _read_data(self, input_domains, split):
        items = []

        for domain, dname in enumerate(input_domains):
            dname = dname.upper()
            path = osp.join(self.dataset_dir, dname, split)
            folders = listdir_nohidden(path)
            folders.sort()

            for label, folder in enumerate(folders):
                impaths = glob.glob(osp.join(path, folder, "*.jpg"))

                for impath in impaths:
                    item = Datum(impath=impath, label=label, domain=domain)
                    items.append(item)

        return items

    def _read_train_data(self,p_g):
        items=[]
        classnames=p_g.classnames
        for idx_template in range(p_g.template_nums):
            for idx_cls in range(p_g.n_cls):
                for idx_style in range(p_g.n_style):
                    item = Datum_sf(
                        cls=idx_cls,
                        style=idx_style,
                        label=idx_cls,
                        template=idx_template,
                        classname=classnames[idx_cls]
                    )
                    items.append(item)
        return items