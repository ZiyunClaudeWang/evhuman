from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
from event_hpe.data_loader import TrackingDataloader
from event_hpe.our_data_loader import OursTrackingDataloader
from argparse import Namespace


class EventHPEDataModule(LightningDataModule):

    def __init__(self, data_dir: str,
                 batch_size: int = 32,
                 num_workers: int = 4,
                 target_action: str = None,
                 args: Namespace = None):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size

        our_data = args.our_data
        use_h5 = getattr(args, 'use_h5', False)

        if our_data and use_h5:
            from event_hpe.our_data_loader_h5 import OursTrackingDataloaderH5
            loader = OursTrackingDataloaderH5
            extra_kwargs = {'h5_dir': data_dir + '_h5'}
        elif our_data:
            loader = OursTrackingDataloader
            extra_kwargs = {}
        else:
            loader = TrackingDataloader
            extra_kwargs = {}

        if args.test_high_fps:
            assert our_data, "High FPS testing is only supported for BEAHM data"

        self.train_db = loader(self.data_dir,
                               mode="train",
                               use_hmr_feats=args.use_hmr_feats,
                               max_steps=args.max_steps,
                               skip=args.skip,
                               target_action=target_action,
                               event_folder=args.event_folder,
                               raw_events=args.contrast_loss > 0,
                               use_volumes=args.use_volumes,
                               **extra_kwargs)
        self.val_db = loader(self.data_dir,
                             mode='test',
                             use_hmr_feats=args.use_hmr_feats,
                             max_steps=args.max_steps,
                             skip=args.skip,
                             target_action=target_action,
                             event_folder=args.event_folder,
                             raw_events=args.contrast_loss > 0,
                             use_volumes=args.use_volumes,
                             test_high_fps=args.test_high_fps,
                             **extra_kwargs)

        self.num_workers = num_workers

    def train_dataloader(self):
        return DataLoader(self.train_db,
                          batch_size=self.batch_size,
                          shuffle=True,
                          num_workers=self.num_workers,
                          persistent_workers=self.num_workers > 0)

    def val_dataloader(self):
        return DataLoader(self.val_db,
                          batch_size=self.batch_size,
                          shuffle=False,
                          num_workers=self.num_workers,
                          persistent_workers=self.num_workers > 0)
