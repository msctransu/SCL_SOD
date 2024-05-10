import torch
import torch.nn as nn
from .Transformer import Transformer
from .Transformer import token_Transformer
from .Decoder import Decoder
from .t2t_vit import T2t_vit_t_14


class ConrastDepthImageNet(nn.Module):
    def __init__(self, args):
        super(ConrastDepthImageNet, self).__init__()

        # Encoder
        self.rgb_backbone = T2t_vit_t_14(pretrained=True,args=args)

        # projector
        embed_dim = self.rgb_backbone.embed_dim
        # patches_num = self.rgb_backbone.tokens_to_token.num_patches
        prev_dim = embed_dim
        self.projector = nn.Sequential(
            nn.Linear(in_features=prev_dim, out_features=prev_dim, bias=False),
            nn.BatchNorm1d(num_features=prev_dim),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=prev_dim, out_features=prev_dim, bias=False),
            nn.BatchNorm1d(num_features=prev_dim)
        )

        # predictor
        self.predictor = nn.Sequential(
            nn.Linear(in_features=prev_dim, out_features=prev_dim*2),
            nn.BatchNorm1d(prev_dim*2),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=prev_dim*2, out_features=prev_dim)
        )

        # VST Convertor
        self.transformer = Transformer(embed_dim=384, depth=4, num_heads=6, mlp_ratio=3.)

        # VST Decoder
        self.token_trans = token_Transformer(embed_dim=384, depth=4, num_heads=6, mlp_ratio=3.)
        self.decoder = Decoder(embed_dim=384, token_dim=64, depth=2, img_size=args.img_size)

    def forward(self, image1, image2, image3):
        # output things
        # output, pred_1, proj_1, pred_2, proj_2
        bs = image1.size()[0]
        # feature 16 can be convert as the embedding
        rgb_fea_1_16, rgb_fea_1_8, rgb_fea_1_4, _ = self.rgb_backbone(image3)

        # Contrast Part
        _, _, _, image1_rgb_fea_1_16 = self.rgb_backbone(image1)
        _, _, _, image2_rgb_fea_1_16 = self.rgb_backbone(image2)

        # VST Decoder
        saliency_fea_1_16, fea_1_16, saliency_tokens, contour_fea_1_16, contour_tokens = self.token_trans(rgb_fea_1_16)
        outputs = self.decoder(saliency_fea_1_16, fea_1_16, saliency_tokens, contour_fea_1_16, contour_tokens, rgb_fea_1_8, rgb_fea_1_4)

        # Contrast Backbone
        image1_rgb_fea_1_16 = image1_rgb_fea_1_16.view(bs, -1)
        z1 = self.projector(image1_rgb_fea_1_16)
        p1 = self.predictor(z1)

        z2 = self.projector(image2_rgb_fea_1_16)
        p2 = self.predictor(z2)

        return outputs, p1, p2, z1.detach(), z2.detach()
