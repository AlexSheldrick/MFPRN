#2d
class GaussianFourierFeatureTransform(torch.nn.Module):
    #Given an input of size [batches, num_input_channels, width, height],
    #returns a tensor of size [batches, mapping_size*2, width, height].

    def __init__(self, num_input_channels, mapping_size=256, scale=10):
        super().__init__()

        self._num_input_channels = num_input_channels
        self._mapping_size = mapping_size
        self._B = torch.randn((num_input_channels, mapping_size)) * scale

    def forward(self, x):
        assert x.dim() == 4, 'Expected 4D input (got {}D input)'.format(x.dim())

        batches, channels, width, height = x.shape

        assert channels == self._num_input_channels,\
            "Expected input to have {} channels (got {} channels)".format(self._num_input_channels, channels)

        # Make shape compatible for matmul with _B.
        # From [B, C, W, H] to [(B*W*H), C].
        x = x.permute(0, 2, 3, 1).reshape(batches * width * height, channels)

        x = x @ self._B.to(x.device)

        # From [(B*W*H), C] to [B, W, H, C]
        x = x.view(batches, width, height, self._mapping_size)
        # From [B, W, H, C] to [B, C, W, H]
        x = x.permute(0, 3, 1, 2)

        x = 2 * np.pi * x
        return torch.cat([torch.sin(x), torch.cos(x)], dim=1)

#3d
class GaussianFourierFeatureTransform3D_vox(torch.nn.Module):
    #Given an input of size [batches, num_input_channels, width, height, depth],
    #returns a tensor of size [batches, mapping_size*2, width, height, depth].

    def __init__(self, num_input_channels, mapping_size=256, scale=10):
        super().__init__()

        self._num_input_channels = num_input_channels
        self._mapping_size = mapping_size
        self._B = torch.randn((num_input_channels, mapping_size)) * scale

    def forward(self, x):
        assert x.dim() == 5, 'Expected 5D input (got {}D input)'.format(x.dim())

        batches, channels, width, height, depth = x.shape

        assert channels == self._num_input_channels,\
            "Expected input to have {} channels (got {} channels)".format(self._num_input_channels, channels)

        # Make shape compatible for matmul with _B.
        # From [B, C, W, H, D] to [(B*W*H*D), C].
        x = x.permute(0, 2, 3, 4, 1).reshape(batches * width * height * depth, channels)

        x = x @ self._B.to(x.device)

        # From [(B*W*H*D), C] to [B, W, H, D, C]
        x = x.view(batches, width, height, depth, self._mapping_size)
        # From [B, W, H, D, C] to [B, C, W, H, D]
        x = x.permute(0, 4, 1, 2, 3)

        x = 2 * np.pi * x
        return torch.cat([torch.sin(x), torch.cos(x)], dim=1)

def make_2d_grid(bb_min, bb_max, shape):
    size = shape[0] * shape[1]
    pxs = torch.linspace(bb_min[0], bb_max[0], shape[0])
    pys = torch.linspace(bb_min[1], bb_max[1], shape[1])

    pxs = pxs.view(-1, 1).expand(*shape).contiguous().view(size)
    pys = pys.view(1, -1).expand(*shape).contiguous().view(size)
    p = torch.stack([pxs, pys], dim=1)
    return p

def evaluate_network_on_2d_grid(network, x, resolution):
    points_batch_size = args.num_points * args.batch_size #55000 #num_points * batch_size
    pointsf = make_2d_grid(
        (-1,)*2, (1,)*2, resolution
    )
    p_split = torch.split(pointsf, points_batch_size)
    values = []
    for pi in p_split:
        pi = pi.unsqueeze(0).to(x.device)
        with torch.no_grad():
            sdf, rgb = network(x, pi)[:,0], network(x, pi)[:,1:]
        values.append(sdf.squeeze(0).detach().cpu())
    value = torch.cat(values, dim=0).numpy()
    value_grid = value.reshape(3, resolution[0], resolution[1])
    return value_grid
    
def implicit_to_img(network, x, resolution, output_path):
    value_grid = evaluate_network_on_2d_grid(network, x, resolution)
    visualize_implicit_rgb(value_grid, output_path)

def project_perspective(points):
    points[...,0] = points[...,0] / -points[...,2]
    points[...,1] = points[...,1] / -points[...,2]
    return points[..., :2]
