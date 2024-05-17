import os.path as osp
from ..build import DATASET_REGISTRY
from ..base_dataset import Datum, DatasetBase,Datum_sf


@DATASET_REGISTRY.register()
class PACS_SF(DatasetBase):
    """PACS.
    Statistics:
        - 4 domains: Photo (1,670), Art (2,048), Cartoon
        (2,344), Sketch (3,929).
        - 7 categories: dog, elephant, giraffe, guitar, horse,
        house and person.
    """

    dataset_dir = "pacs"
    domains = ["none","art_painting", "cartoon", "photo", "sketch"]
    data_url = "https://drive.google.com/uc?id=1m4X4fROCCXMO0lRLrr6Zz9Vb3974NWhE"
    # the following images contain errors and should be ignored
    _error_paths = ["sketch/dog/n02103406_4068-1.png"]

    def __init__(self, cfg,p_g):
        root = osp.abspath(osp.expanduser(cfg.DATASET.ROOT))
        self.dataset_dir = osp.join(root, self.dataset_dir)
        self.image_dir = osp.join(self.dataset_dir, "images")
        self.split_dir = osp.join(self.dataset_dir, "splits")

        if not osp.exists(self.dataset_dir):
            dst = osp.join(root, "pacs.zip")
            self.download_data(self.data_url, dst, from_gdrive=True)

        self.check_input_domains(
            cfg.DATASET.SOURCE_DOMAINS, cfg.DATASET.TARGET_DOMAINS
        )
        test_datasets=[]
        for domain in cfg.DATASET.TARGET_DOMAINS:
            test_datasets.append(self._read_data([domain], "test"))
        train=self._read_train_data(p_g)

        super().__init__(train_x=train,test=test_datasets)
    
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

        

    def _read_data(self, input_domains, split):
        items = []

        for domain, dname in enumerate(input_domains):
            file = osp.join(
                self.split_dir, dname + "_" + split + "_kfold.txt"
            )
            impath_label_list = self._read_split_pacs(file)

            for impath, label in impath_label_list:
                classname = impath.split("/")[-2]
                item = Datum(
                    impath=impath,
                    label=label,
                    domain=domain,
                    classname=classname
                )
                items.append(item)

        return items

    def _read_split_pacs(self, split_file):
        items = []

        with open(split_file, "r") as f:
            lines = f.readlines()

            for line in lines:
                line = line.strip()
                impath, label = line.split(" ")
                if impath in self._error_paths:
                    continue
                impath = osp.join(self.image_dir, impath)
                label = int(label) - 1
                items.append((impath, label))

        return items
