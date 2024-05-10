import torch.nn as nn
from .t2t_vit import T2t_vit_t_14
from .Transformer import Transformer
from .Transformer import token_Transformer
from .Decoder import Decoder


class ImageDepthNet(nn.Module):
    def __init__(self, args):
        super(ImageDepthNet, self).__init__()

        # VST Encoder
        self.rgb_backbone = T2t_vit_t_14(pretrained=True, args=args)


        # VST Convertor
        self.transformer = Transformer(embed_dim=384, depth=4, num_heads=6, mlp_ratio=3.)

        # VST Decoder
        self.token_trans = token_Transformer(embed_dim=384, depth=4, num_heads=6, mlp_ratio=3.)
        self.decoder = Decoder(embed_dim=384, token_dim=64, depth=2, img_size=args.img_size)

    def forward(self, image_Input):

        B, _, _, _ = image_Input.shape
        # VST Encoder
        rgb_fea_1_16, rgb_fea_1_8, rgb_fea_1_4, _ = self.rgb_backbone(image_Input)
        # print("rgb_backbone is as followed")
        # # for name, p in self.rgb_backbone.blocks.named_parameters():
        #     # print("name_p: ", name, "p: ", p.shape)
        # # print(self.rgb_backbone.blocks.named_parameters())
        # print(self.rgb_backbone.tokens_to_token.num_patches)
        # print(self.rgb_backbone.embed_dim)
        # print("rgb_fea_1_16 is as followed")
        # print(rgb_fea_1_16.shape)

        # VST Convertor
        rgb_fea_1_16 = self.transformer(rgb_fea_1_16)
        # rgb_fea_1_16 [B, 14*14, 384]

        # VST Decoder
        saliency_fea_1_16, fea_1_16, saliency_tokens, contour_fea_1_16, contour_tokens = self.token_trans(rgb_fea_1_16)
        # saliency_fea_1_16 [B, 14*14, 384]
        # fea_1_16 [B, 1 + 14*14 + 1, 384]
        # saliency_tokens [B, 1, 384]
        # contour_fea_1_16 [B, 14*14, 384]
        # contour_tokens [B, 1, 384]

        outputs = self.decoder(saliency_fea_1_16, fea_1_16, saliency_tokens, contour_fea_1_16, contour_tokens, rgb_fea_1_8, rgb_fea_1_4)

        return outputs
