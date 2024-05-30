import torch.nn as nn

# Config format:
#      |h_size:int|l_config:List[Tuple[nn.activation_func, int]]
# name:

SYS_TEST_CONFIGS = {
	'TEST_h5_3x3_ReLU': {
		'h_size'  : 5,
		'l_config': [(nn.ReLU, 3), (nn.ReLU, 3), (nn.ReLU, 3), (nn.ReLU, 1)]
	},
	'TEST_h10_3i3i1_ReLU': {
		'h_size'  : 10,
		'l_config': [(nn.ReLU, 3), (nn.ReLU, 3), (nn.ReLU, 1)]
	}
}
