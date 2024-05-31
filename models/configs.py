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

DEF_CONFIGS = {
	'h5_3i3i1_ReLU': {
		'h_size'  : 5,
		'l_config': [(nn.ReLU, 3), (nn.ReLU, 3), (nn.ReLU, 1)]
	},
	'h16_8i1': {
		'h_size'  : 16,
		'l_config': [(nn.ReLU, 8), (nn.ReLU, 1)]
	},
	'h32_8i1': {
		'h_size'  : 32,
		'l_config': [(nn.ReLU, 8), (nn.ReLU, 1)]
	},
	'h64_8i1': {
		'h_size'  : 64,
		'l_config': [(nn.ReLU, 8), (nn.ReLU, 1)]
	},
	'h32_8i4i1': {
		'h_size'  : 32,
		'l_config': [(nn.ReLU, 8), (nn.ReLU, 4), (nn.ReLU, 1)]
	},
	'h32_16i8i4i1': {
		'h_size'  : 32,
		'l_config': [(nn.ReLU, 16), (nn.ReLU, 8), (nn.ReLU, 4), (nn.ReLU, 1)]
	},
	'h64_16i16i8i8i1': {
		'h_size'  : 64,
		'l_config': [(nn.ReLU, 16), (nn.ReLU, 16), (nn.ReLU, 8), (nn.ReLU, 8), (nn.ReLU, 1)]
	}
}