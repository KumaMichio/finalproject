"""
Re-Identification Module
Extracts features for cross-camera object matching.

Primary: OSNet (torchreid) pretrained on Market-1501 — proper ReID model.
Fallback: ResNet50 ImageNet if torchreid is not installed.
"""

import torch
import torch.nn.functional as F
from torchvision import transforms
import numpy as np
import logging

logger = logging.getLogger(__name__)

# Try to import torchreid (pip install torchreid)
try:
    import torchreid
    TORCHREID_AVAILABLE = True
except ImportError:
    TORCHREID_AVAILABLE = False
    logger.warning(
        "torchreid not found. Falling back to ResNet50-ImageNet (poor ReID accuracy). "
        "Install with: pip install torchreid"
    )


class ReIDExtractor:
    """
    Feature extractor for person/vehicle re-identification.

    Uses OSNet pretrained on Market-1501 when torchreid is available,
    otherwise falls back to ResNet50 with ImageNet weights.
    """

    def __init__(self, model_name='osnet_x1_0', pretrained=True, device=None):
        """
        Initialize ReID extractor.

        Args:
            model_name: torchreid model name ('osnet_x1_0', 'osnet_x0_75', 'osnet_x0_5').
                        Ignored when torchreid is unavailable (uses ResNet50).
            pretrained: Whether to use pretrained weights (Market-1501 via torchreid).
            device: 'cuda' | 'cpu' | None (auto-detect).
        """
        self.model_name = model_name if TORCHREID_AVAILABLE else 'resnet50_imagenet_fallback'
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')

        # OSNet output dim = 512; ResNet50 output dim = 2048
        self.feature_dim = 512 if TORCHREID_AVAILABLE else 2048

        # Gallery: {global_id: [feature_vectors]}
        self.gallery = {}

        # Standard ReID preprocessing (256x128)
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((256, 128)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

        self.model = None
        self._load_model(pretrained)

        logger.info(
            f"ReIDExtractor ready: model={self.model_name}, "
            f"device={self.device}, feature_dim={self.feature_dim}"
        )

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def _load_model(self, pretrained):
        if TORCHREID_AVAILABLE:
            self._load_torchreid_model(pretrained)
        else:
            self._load_resnet50_fallback()

    def _load_torchreid_model(self, pretrained):
        """Load OSNet pretrained on Market-1501 via torchreid."""
        try:
            self.model = torchreid.models.build_model(
                name=self.model_name,
                num_classes=751,     # Market-1501 has 751 identities
                loss='softmax',
                pretrained=pretrained
            )
            self.model.to(self.device)
            self.model.eval()
            logger.info(f"Loaded {self.model_name} (Market-1501 pretrained) via torchreid")
        except Exception as e:
            logger.error(f"Failed to load torchreid model: {e}")
            raise

    def _load_resnet50_fallback(self):
        """Fallback: ResNet50 with ImageNet weights (feature extraction only)."""
        try:
            import torchvision.models as tv_models
            backbone = tv_models.resnet50(pretrained=True)
            backbone.fc = torch.nn.Identity()   # remove classification head
            backbone.to(self.device)
            backbone.eval()
            self.model = backbone
            logger.warning("Using ResNet50-ImageNet fallback — ReID accuracy will be limited")
        except Exception as e:
            logger.error(f"Failed to load ResNet50 fallback: {e}")
            raise

    # ------------------------------------------------------------------
    # Feature extraction
    # ------------------------------------------------------------------

    def extract_feature(self, frame, box):
        """
        Extract ReID feature vector from a detected object crop.

        Args:
            frame: numpy array (H, W, 3) RGB image.
            box: [x1, y1, x2, y2] bounding box (integers).

        Returns:
            numpy array shape (1, feature_dim), L2-normalised, or None on failure.
        """
        try:
            x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])

            if x2 <= x1 or y2 <= y1:
                return None

            crop = frame[y1:y2, x1:x2]
            if crop.size == 0 or crop.shape[0] < 10 or crop.shape[1] < 10:
                return None

            tensor = self.transform(crop).unsqueeze(0).to(self.device)

            with torch.no_grad():
                feat = self.model(tensor)
                feat = F.normalize(feat, p=2, dim=1)

            return feat.cpu().numpy()   # shape (1, feature_dim)

        except Exception as e:
            logger.error(f"Feature extraction error: {e}")
            return None

    # ------------------------------------------------------------------
    # Gallery matching
    # ------------------------------------------------------------------

    def match_with_gallery(self, feature, threshold=0.5):
        """
        Match a feature vector against the gallery using cosine similarity.

        Because features are L2-normalised, dot product == cosine similarity.

        Args:
            feature: numpy array (1, feature_dim).
            threshold: minimum cosine similarity to accept a match [0, 1].

        Returns:
            int | None: matched global_id, or None if no match above threshold.
        """
        if not self.gallery:
            return None

        query = feature.flatten()
        best_id = None
        best_score = -1.0

        for global_id, feat_list in self.gallery.items():
            # Compare against all stored features; take the best score
            for gallery_feat in feat_list:
                score = float(np.dot(query, gallery_feat.flatten()))
                if score > best_score:
                    best_score = score
                    best_id = global_id

        if best_score >= threshold:
            logger.debug(f"ReID match → ID {best_id} (score={best_score:.3f})")
            return best_id

        return None

    # ------------------------------------------------------------------
    # Gallery management
    # ------------------------------------------------------------------

    def add_to_gallery(self, global_id, feature, max_per_id=10):
        """
        Add a feature vector to the gallery for a global ID.

        Keeps only the most recent `max_per_id` features per ID to bound memory.
        """
        if global_id not in self.gallery:
            self.gallery[global_id] = []

        self.gallery[global_id].append(feature)

        if len(self.gallery[global_id]) > max_per_id:
            self.gallery[global_id].pop(0)

        logger.debug(f"Gallery updated for ID {global_id} ({len(self.gallery[global_id])} features)")

    def get_gallery_size(self):
        return sum(len(fl) for fl in self.gallery.values())

    def get_gallery_ids(self):
        return list(self.gallery.keys())

    def clear_gallery(self):
        self.gallery.clear()
        logger.info("Gallery cleared")

    def save_gallery(self, filepath):
        try:
            np.savez(filepath, **{str(k): np.array(v) for k, v in self.gallery.items()})
            logger.info(f"Gallery saved → {filepath}")
        except Exception as e:
            logger.error(f"Failed to save gallery: {e}")

    def load_gallery(self, filepath):
        try:
            data = np.load(filepath, allow_pickle=True)
            self.gallery = {int(k): list(v) for k, v in data.items()}
            logger.info(f"Gallery loaded ← {filepath} ({self.get_gallery_size()} features)")
        except Exception as e:
            logger.error(f"Failed to load gallery: {e}")

    # ------------------------------------------------------------------
    # Info
    # ------------------------------------------------------------------

    def get_model_info(self):
        return {
            'model_name': self.model_name,
            'backend': 'torchreid' if TORCHREID_AVAILABLE else 'torchvision_fallback',
            'pretrained_dataset': 'Market-1501' if TORCHREID_AVAILABLE else 'ImageNet',
            'feature_dim': self.feature_dim,
            'device': str(self.device),
            'gallery_size': self.get_gallery_size(),
        }
